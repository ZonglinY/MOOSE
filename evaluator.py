import os, time, re
import torch
import openai
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from evaluate_utils import prompts_for_evaluator_modules, pick_score, load_ground_truth_hypotheses


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name
        self.root_data_dir = args.root_data_dir
        self.output_dir = args.output_dir
        self.prev_api_usage_time = 0
        if args.if_azure_api == 0:
            openai.api_key = ""
        else:
            openai.api_type = ""
            openai.api_base = ""
            openai.api_version = ""
            openai.api_key = ""
        assert openai.api_key != ""
        # self.hypotheses is a sub-element of self.result
        self.result = None
        self.hypotheses = None
        self.scores = None
        self.score_reasons = None
        # only use self.previous_scores when self.args.prev_eval_output_dir is not ""
        self.previous_scores = []
        if self.model_name == "gpt4":
            print("Warning: using gpt4")

    def read_from_checkpoint(self):
        # self.background, self.hypotheses
        if self.args.if_groundtruth_hypotheses == 0:
            chkp_dir = os.path.join(self.output_dir, "background_inspiration_hypotheses.pt")
            self.result = torch.load(chkp_dir)
            self.background = self.result[2]
            print("len(self.background): ", len(self.background))
            self.hypotheses = self.result[8]
        elif self.args.if_groundtruth_hypotheses == 1:
            dataset_dir = os.path.join(self.root_data_dir, "business_research.xlsx")
            self.background, self.hypotheses = load_ground_truth_hypotheses(dataset_dir)
            # print("self.background: ", self.background)
            # print("self.hypotheses: ", self.hypotheses)
        else:
            raise NotImplementedError
        # start_id and end_id
        if self.args.start_id != -1 or self.args.end_id != -1:
            assert self.args.start_id != -1 and self.args.end_id != -1
            print("start_id: {}; end_id: {}".format(self.args.start_id, self.args.end_id))
            if len(self.background) < self.args.end_id:
                print("Warning: length of self.background is less than self.end_id.")
                if len(self.background) == self.args.end_id - self.args.start_id:
                    # self.background is itself
                    print("Warning: using current full background for evaluation. len(self.background): {}".format(len(self.background)))
                else:
                    raise Exception("Can't decide which background to evaluate. len(self.background): {}".format(len(self.background)))
            else:
                self.background = self.background[self.args.start_id : self.args.end_id]
        else:
            print("Using full ckpt for evaluation")
        print("len(self.background): ", len(self.background))
        # self.args.prev_eval_output_dir
        if self.args.prev_eval_output_dir != "":
            assert len(self.previous_scores) == 0
            with open(self.args.prev_eval_output_dir, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("id: "):
                        cur_id = re.findall(r'id: \d+', line)
                        assert len(cur_id) == 1
                        cur_id = int(cur_id[0].strip("id: "))
                        cur_score = re.findall(r'cur_score: \[.*\]', line)
                        assert len(cur_score) == 1
                        cur_score = cur_score[0].strip("cur_score: [").strip(']').split(',')
                        assert len(cur_score) == 3
                        self.previous_scores.append([])
                        assert self.previous_scores[cur_id] == []
                        for s in cur_score:
                            s = int(s.strip().strip("\'"))
                            self.previous_scores[cur_id].append(s)
            print("len(self.previous_scores): ", len(self.previous_scores))




    # FUNCTION:
    #   For each hypothesis, evaluate from three aspects (validness, novelty, helpfulness)
    #   Not adding significance metric now, since it's hard to evaluate. If in need, we could evaluate it seperately
    def evaluate(self):
        scores = {}
        score_reasons = {}
        cnt_finished = 0
        if self.args.if_groundtruth_hypotheses == 0:
            for cur_id_bkg, cur_bkg_ori in enumerate(self.background):
                if cur_bkg_ori not in scores:
                    cur_bkg = cur_bkg_ori
                    assert cur_bkg not in score_reasons
                    scores[cur_bkg] = []
                    score_reasons[cur_bkg] = []
                    cur_bkg = cur_bkg_ori
                    cur_hyp_for_cur_bkg = self.hypotheses[cur_bkg_ori]
                    # in case a bkg has more than one data item in our dataset
                    if len(cur_hyp_for_cur_bkg) > 1:
                        cur_hyp_for_cur_bkg = cur_hyp_for_cur_bkg[:1]
                else:
                    # raise Exception("repeated key in scores: {}; cur_bkg: {}".format(scores, cur_bkg))
                    cur_bkg = cur_bkg_ori + " "
                    assert cur_bkg not in score_reasons
                    scores[cur_bkg] = []
                    score_reasons[cur_bkg] = []
                    cur_hyp_for_cur_bkg = self.hypotheses[cur_bkg_ori][1:2]
                if cur_id_bkg == 0:
                    print("len(cur_hyp_for_cur_bkg): ", len(cur_hyp_for_cur_bkg))
                for cur_id_hyp_direct_or_indirect , cur_hyp_direct_or_indirect in enumerate(cur_hyp_for_cur_bkg):
                    scores[cur_bkg].append([])
                    score_reasons[cur_bkg].append([])
                    for cur_id_hyp_all_itr, cur_hyp_all_itr in enumerate(cur_hyp_direct_or_indirect):
                        scores[cur_bkg][cur_id_hyp_direct_or_indirect].append([])
                        score_reasons[cur_bkg][cur_id_hyp_direct_or_indirect].append([])
                        assert len(cur_hyp_all_itr) == self.args.num_CoLM_feedback_times + 1
                        for cur_id_hyp, cur_hyp in enumerate(cur_hyp_all_itr):
                            scores[cur_bkg][cur_id_hyp_direct_or_indirect][cur_id_hyp_all_itr].append([])
                            score_reasons[cur_bkg][cur_id_hyp_direct_or_indirect][cur_id_hyp_all_itr].append([])
                            if "\n\nRefined hypothesis:" not in cur_hyp and "\n\nReasoning process:" not in cur_hyp and "\n\nHypothesis:" not in cur_hyp:
                                if cnt_finished <= len(self.previous_scores)-1:
                                    cur_score = self.previous_scores[cnt_finished]
                                    cur_score_reason = ""
                                else:
                                    pre_prompt, post_prompt = prompts_for_evaluator_modules()
                                    input_txt = pre_prompt + cur_hyp + post_prompt
                                    cur_generation = self.llm_generation(input_txt)
                                    # print("cur_generation: ", cur_generation)
                                    # cur_score: [validness score, novelty score, helpfulness score]
                                    cur_score, cur_score_reason, if_matched = pick_score(cur_generation, input_txt)
                                    assert if_matched == True
                                # add cur_score to self.scores
                                scores[cur_bkg][cur_id_hyp_direct_or_indirect][cur_id_hyp_all_itr][cur_id_hyp].append(cur_score)
                                score_reasons[cur_bkg][cur_id_hyp_direct_or_indirect][cur_id_hyp_all_itr][cur_id_hyp].append(cur_score_reason)
                                print("id: {}; cur_score: {}".format(cnt_finished, cur_score))
                                cnt_finished += 1
        else:
            score_list = []
            for cur_id_hyp, cur_hyp in enumerate(self.hypotheses):
                if cnt_finished <= len(self.previous_scores)-1:
                    cur_score = self.previous_scores[cnt_finished]
                    cur_score_reason = ""
                else:
                    pre_prompt, post_prompt = prompts_for_evaluator_modules()
                    input_txt = pre_prompt + cur_hyp + post_prompt
                    cur_generation = self.llm_generation(input_txt)
                    cur_score, cur_score_reason, if_matched = pick_score(cur_generation, input_txt)
                    assert if_matched == True
                if cur_hyp in scores or cur_hyp in score_reasons:
                    print("cur_hyp: ", cur_hyp)
                    raise Exception
                # scores
                scores[cur_hyp] = cur_score
                # score_reasons
                score_reasons[cur_hyp] = cur_score_reason
                # score_list
                cur_score_int = [int(i) for i in cur_score]
                score_list.append(cur_score_int)
                print("id: {}; cur_score: {}".format(cnt_finished, cur_score))
                cnt_finished += 1
            score_list = np.array(score_list)
            print("score_list.shape: ", score_list.shape)
            ave_score = np.mean(score_list, axis=0)
            print("ave_score: ", ave_score)


        # save Important variables
        assert self.scores == None
        self.scores = scores
        self.score_reasons = score_reasons
        if self.args.start_id != -1 or self.args.end_id != -1:
            torch.save(self.scores, os.path.join(self.output_dir, "automatic_evaluation_hypotheses_{}_{}_{}.pt".format(self.model_name, self.args.start_id, self.args.end_id)))
            torch.save(self.score_reasons, os.path.join(self.output_dir, "automatic_evaluation_hypotheses_reasons_{}_{}_{}.pt".format(self.model_name, self.args.start_id, self.args.end_id)))
        else:
            torch.save(self.scores, os.path.join(self.output_dir, "automatic_evaluation_hypotheses_{}.pt".format(self.model_name)))
            torch.save(self.score_reasons, os.path.join(self.output_dir, "automatic_evaluation_hypotheses_reasons_{}.pt".format(self.model_name)))




    def llm_init(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # it should be 4096, but we tend to make it cheaper
        self.model_input_len = 2048

    def llm_generation(self, input_txt):
        if self.model_name == 'chatgpt':
            api_model_name = 'gpt-3.5-turbo'
            temperature = 0.00
            sleep_time = 0.25
            assert self.args.if_azure_api == 0
        elif self.model_name == 'gpt4':
            if self.args.if_azure_api == 0:
                api_model_name = 'gpt-4-0613'
            else:
                api_model_name = "GPT4"
            temperature = 0.00
            sleep_time = 0.35
        else:
            raise NotImplementedError
        # the while loop is used to avoid the rate limit set by openai
        while (time.time() - self.prev_api_usage_time) <= sleep_time:
            time.sleep(sleep_time/2)
        # openai.api_key = self.api_key
        max_tokens = 320
        # To prevent api error
        if_api_completed = False
        while if_api_completed == False:
            try:
                if self.args.if_azure_api == 0:
                    response = openai.ChatCompletion.create(
                    model=api_model_name,
                    messages=[{"role": "user", "content": input_txt}],
                    top_p=0.90,
                    temperature=temperature,
                    max_tokens=max_tokens)
                    reply = response["choices"][0]['message']['content']
                    if_api_completed = True
                else:
                    response = openai.ChatCompletion.create(
                    engine=api_model_name,
                    messages=[{"role": "user", "content": input_txt}],
                    top_p=0.90,
                    temperature=temperature,
                    max_tokens=max_tokens)
                    reply = response["choices"][0]['message']['content']
                    if_api_completed = True
            except:
                print("OpenAI reach its rate limit")
                time.sleep(sleep_time*2)
        self.prev_api_usage_time = time.time()
        return reply
