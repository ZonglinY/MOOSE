import os, argparse, logging, sys, random, datetime, math, time, shutil, csv
import numpy as np
import openai
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from rank_bm25 import BM25Okapi
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
from utils import chunk_passage, find_passages_with_titles, prompts_for_tomato_modules, unify_feedbacks_to_format, match_existing_title_given_title_generation, pick_from_generation, print_nvidia_smi, load_variables_for_debug, find_simi_score_using_BM25

# if deadlocked, try to set it to "false" (TOKENIZERS_PARALLELISM == true may lead to deadlocks)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tomato(object):
    # self.corpus/background_corpus/inspiration_corpus: [['title0', 'passage0'], ...]; used as input to develop hypothesis;
    #   corpus = background_corpus + inspiration_corpus
    # existing_literature: [['title0', 'existing_literature0'], ...]; used to check the novelty of developed hypothesis
    def __init__(self, args, corpus, background_corpus, inspiration_corpus, background_golden, inspiration_golden, existing_literature):
        assert len(background_corpus) + len(inspiration_corpus) == len(corpus)
        assert len(background_golden) == len(inspiration_golden)
        ## other variables
        self.args = args
        self.model_name = args.model_name
        self.num_CoLM_feedback_times = args.num_CoLM_feedback_times
        self.num_background_for_hypotheses = args.num_background_for_hypotheses
        self.if_indirect_feedback = args.if_indirect_feedback
        self.if_only_indirect_feedback = args.if_only_indirect_feedback
        self.if_close_domain = args.if_close_domain
        self.if_ban_selfeval = args.if_ban_selfeval
        self.if_baseline = args.if_baseline
        # For each module, whether self_eval (for hypothesis_generator, if_with_eval does not mean self_eval, but self_present_reasoning_process)
        if self.if_ban_selfeval == 0:
            if self.model_name == 'chatgpt':
                self.if_self_eval_module = {'background_finder': True, 'inspiration_title_retriever': True, 'inspiration_passage_retriever': True, 'background_evaluator': False, 'hypothesis_suggstor': False, 'hypothesis_generator': True, 'deductive_consistency_evaluator': False, 'indiscriminate_confirmation_handler': False, 'generalization_checker': False, 'novelty_detector': True, 'specification_detector': True, 'background_changer': False, 'inspiration_title_suggestor': True, 'inspiration_title_changer': True}
            else:
                self.if_self_eval_module = {'background_finder': False, 'inspiration_title_retriever': True, 'inspiration_passage_retriever': False, 'background_evaluator': False, 'hypothesis_suggstor': False, 'hypothesis_generator': False, 'deductive_consistency_evaluator': False, 'indiscriminate_confirmation_handler': False, 'generalization_checker': False, 'novelty_detector': False, 'specification_detector': False, 'background_changer': False, 'inspiration_title_suggestor': True, 'inspiration_title_changer': True}
        else:
            self.if_self_eval_module = {'background_finder': False, 'inspiration_title_retriever': False, 'inspiration_passage_retriever': False, 'background_evaluator': False, 'hypothesis_suggstor': False, 'hypothesis_generator': False, 'deductive_consistency_evaluator': False, 'indiscriminate_confirmation_handler': False, 'generalization_checker': False, 'novelty_detector': False, 'specification_detector': False, 'background_changer': False, 'inspiration_title_suggestor': False, 'inspiration_title_changer': False}
        self.keyword_key_generation = {'background_finder': 'Background:', 'inspiration_title_retriever': 'Title:', 'inspiration_passage_retriever': 'Inspiration:', 'background_evaluator': None, 'hypothesis_suggstor': 'Suggestion:', 'hypothesis_generator_first_with_future_fdbk': 'Hypothesis:', 'hypothesis_generator_refine_with_future_fdbk': 'Refined hypothesis:', 'hypothesis_generator_first_without_future_fdbk': 'Hypothesis:', 'hypothesis_generator_refine_without_future_fdbk': 'Refined hypothesis:', 'deductive_consistency_evaluator': 'Feedback:', 'indiscriminate_confirmation_handler': 'Feedback:', 'generalization_checker': 'Feedback:', 'novelty_detector': 'Feedback:', 'specification_detector': 'Feedback:', 'background_changer': 'Feedback:', 'inspiration_title_suggestor': 'Problem:', 'inspiration_title_changer': 'Title:'}
        self.keyword_key_generation_eval = {'background_finder': 'Evaluation:', 'inspiration_title_retriever': 'Evaluation:', 'inspiration_passage_retriever': 'Evaluation:', 'background_evaluator': None, 'hypothesis_suggstor': None, 'hypothesis_generator_first_with_future_fdbk': 'Reasoning process:', 'hypothesis_generator_refine_with_future_fdbk': 'Reasoning process:', 'hypothesis_generator_first_without_future_fdbk': 'Reasoning process:', 'hypothesis_generator_refine_without_future_fdbk': 'Reasoning process:', 'deductive_consistency_evaluator': 'Evaluation:', 'indiscriminate_confirmation_handler': 'Evaluation:', 'generalization_checker': 'Evaluation:', 'novelty_detector': 'Suggestion:', 'specification_detector': 'Suggestion:', 'background_changer': 'Evaluation:', 'inspiration_title_suggestor': 'Suggestion:', 'inspiration_title_changer': 'Evaluation:'}
        self.api_key = None
        # note the previous api usage time to avoid rate limit by openai
        self.prev_api_usage_time = 0
        ## initial variable
        # self.corpus / background_corpus / inspiration_corpus: [['title0', 'passage0'], ...]
        self.corpus = corpus
        # corpus = background_corpus + inspiration_corpus
        if self.args.if_insp_pasg_for_bkg_and_bkg_pasg_included_in_insp == 0:
            self.background_corpus = background_corpus
            self.inspiration_corpus = inspiration_corpus
        else:
            self.background_corpus = inspiration_corpus
            self.inspiration_corpus = background_corpus + inspiration_corpus
        # background_golden / inspiration_golden: [[bkg0, bkg1](for line 0), ...]
        self.background_golden = background_golden
        self.inspiration_golden = inspiration_golden
        # self.corpus_chunk: ['title0, passage0_chunk0', 'title0, passage0_chunk1', ...]
        self.corpus_chunk = None
        # self.bkg_corpus_chunk_noter: note which corpus chunk has been investigated for background; 0 <= self.bkg_corpus_chunk_noter <= len(self.corpus_chunk) - 1
        self.bkg_corpus_chunk_noter = args.bkg_corpus_chunk_noter
        # self.title: ['title0', 'title1', ...]; used for find inspiration
        self.title = None
        # self.title_chunk: ['title_chunk0', 'title_chunk1', ...]; used for find inspiration
        self.title_chunk = None
        # self.inspiration_title: ['title0', 'title1', ...]; used for find inspiration (only titles from inspiration passages)
        self.inspiration_title = None
        # self.inspiration_title_chunk: ['title_chunk0', 'title_chunk1', ...]; used for find inspiration (only titles from inspiration passages)
        self.inspiration_title_chunk = None
        # self.existing_literature: [['title0', 'existing_literature0'], ...]
        self.existing_literature = existing_literature
        # self.existing_literature_chunk: ['title0, passage0_chunk0', 'title0, passage0_chunk1', ...]
        self.existing_literature_chunk = None
        ## intermediate variables
        # self.background: [bg0, bg1, ...]
        self.background = []
        # self.background_self_eval: [bg0_eval, bg1_eval, ...]
        self.background_self_eval = []
        # self.selected_titles: {bg0: [[ttl0, ttl1, ...], [](results from inspiration_changer)]}
        self.selected_titles = {}
        # self.selected_titles_self_eval:{bg0: [[ttl0_eval, ttl1_eval, ...], [](results from inspiration_changer)]}
        self.selected_titles_self_eval = {}
        # self.inspiration: {bg0: [[i0, i1, ...], [](results from inspiration_changer)]}
        self.inspiration = {}
        # self.inspiration_self_eval: {bg0: [[i0_eval, i1_eval, ...], [](results from inspiration_changer)]}
        self.inspiration_self_eval = {}
        self.suggestion = {}
        ## output variable
        # self.hypothesis: {bg0: [['hypothesis_round0', 'hypothesis_round1', ...], [](results from inspiration_changer)]}
        self.hypothesis = {}
        # self.hypothesis_reasoning_process: {bg0: [['hypothesis_reasoning_round0', 'hypothesis_reasoning_round1', ...], [](results from inspiration_changer)]}
        self.hypothesis_reasoning_process = {}
        # feedbacks
        # self.hypothesis_CoLM_internal_feedback: feedbacks from consistency checker, reality checker, and novelty checker
        #   {'bg0': [{'hp0': ['consistent feedback', 'reality feedback', 'novelty feedback'], 'hp1':...}, {}(results from inspiration_changer)], 'bg1':...}
        self.hypothesis_CoLM_internal_feedback = {}
        # self.hypothesis_CoLM_external_feedback: feedbacks to inspiration finder and background finder
        self.hypothesis_CoLM_external_feedback = {}
        ## model
        self.tokenizer = None
        self.model = None
        self.model_input_len = None
        print("self.model_name: {}; self.num_background_for_hypotheses: {}; self.num_CoLM_feedback_times: {}; self.bkg_corpus_chunk_noter: {}; self.args.max_chunks_each_passage: {}; self.if_indirect_feedback: {}; self.if_close_domain: {}; self.if_ban_selfeval: {}; self.if_baseline: {}; self.args.if_novelty_module_have_access_to_surveys: {}; self.args.if_insp_pasg_for_bkg_and_bkg_pasg_included_in_insp: {}".format(self.model_name, self.num_background_for_hypotheses, self.num_CoLM_feedback_times, self.bkg_corpus_chunk_noter, self.args.max_chunks_each_passage, self.if_indirect_feedback, self.if_close_domain, self.if_ban_selfeval, self.if_baseline, self.args.if_novelty_module_have_access_to_surveys, self.args.if_insp_pasg_for_bkg_and_bkg_pasg_included_in_insp))

    def save_important_variables(self):
        # torch.save([self.args.model_name, self.bkg_corpus_chunk_noter, self.background, self.inspiration, self.hypothesis, self.hypothesis_reasoning_process, self.hypothesis_CoLM_internal_feedback, self.hypothesis_CoLM_external_feedback, self.args.max_chunks_each_passage, self.corpus_chunk], os.path.join(self.args.output_dir, "background_inspiration_hypotheses.pt"))
        torch.save([self.args.model_name, self.bkg_corpus_chunk_noter, self.background, self.background_self_eval, self.selected_titles, self.selected_titles_self_eval, self.inspiration, self.inspiration_self_eval, self.hypothesis, self.hypothesis_reasoning_process, self.hypothesis_CoLM_internal_feedback, self.hypothesis_CoLM_external_feedback, self.args.max_chunks_each_passage, self.corpus_chunk, self.args, self.suggestion, self.if_baseline, self.args.if_novelty_module_have_access_to_surveys, self.args.if_insp_pasg_for_bkg_and_bkg_pasg_included_in_insp, self.args.if_hypothesis_suggstor], os.path.join(self.args.output_dir, "background_inspiration_hypotheses.pt"))
        print("Important variables saved successfully")

    # Function: init self.model (possibly with vicuna) and self.tokenizer
    def llm_init(self):
        if self.model_name != "chatgpt":
            MODEL_CLASSES = {
            # "bart-base": (BartForConditionalGeneration, BartTokenizer, BartConfig, "facebook/bart-base"),
            # I know here should be 1024, but use 512 to prevent cuda error (gpt2 is used for debug anyway)
            "gpt2": (GPT2LMHeadModel, GPT2Tokenizer, 450, "gpt2"),
            # "gptj": (AutoModelForCausalLM, AutoTokenizer, GPTJConfig, 'EleutherAI/gpt-j-6B'),
            "llama": (AutoModelForCausalLM, LlamaTokenizer, 2048, "decapoda-research/llama-7b-hf"),
            "vicuna": (AutoModelForCausalLM, AutoTokenizer, 2048, "eachadea/vicuna-7b-1.1"),
            "vicuna13": (AutoModelForCausalLM, AutoTokenizer, 2048, "eachadea/vicuna-13b-1.1"),
            "falcon": (AutoModelForCausalLM, AutoTokenizer, 2048, "tiiuae/falcon-40b-instruct")
            # "vicuna-delta": (AutoModelForCausalLM, AutoTokenizer, 2048, "lmsys/vicuna-7b-delta-v1.1")
            # "mpt": (AutoModelForCausalLM, GPTNeoXTokenizerFast, None, "mosaicml/mpt-7b")
            }
            Generator_Model, Generator_Tokenizer, self.model_input_len, Generator_Model_Name = MODEL_CLASSES[self.model_name]
            ## load tokenizer
            if "mpt" in Generator_Model_Name:
                tokenizer = Generator_Tokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            else:
                tokenizer = Generator_Tokenizer.from_pretrained(Generator_Model_Name)
            self.tokenizer = tokenizer
            ## load model
            if "falcon" in Generator_Model_Name:
                model = Generator_Model.from_pretrained(Generator_Model_Name, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
                # model = Generator_Model.from_pretrained(Generator_Model_Name, trust_remote_code=True, device_map="auto", load_in_8bit=True)
            else:
                model = Generator_Model.from_pretrained(Generator_Model_Name, device_map="auto", torch_dtype=torch.float16)
            model.eval()
            # half all models to save gpu memory
            # if self.model_name == "vicuna13" or self.model_name == "vicuna" or self.model_name == "llama":
            #     model = model.half()
            # model.to(device1)
            self.model = model
            print("Loaded self.tokenizer and self.model, {} is initialized successfully".format(self.model_name))
            print_nvidia_smi()
        else:
            self.api_key = self.args.api_key
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            # it should be 4096, but we tend to make it cheaper
            self.model_input_len = 2048
            assert self.api_key != ""


    # Function: split the corpus into multiple chunks with proper size to be used as input for LLMs
    #   func0: self.corpus --> self.title + self.title_chunk + self.corpus_chunk
    #   func1: self.existing_literature --> self.existing_literature_chunk (if self.existing_literature != None)
    #   self.corpus: [['title0', 'passage0'], ...]
    #   self.corpus_chunk: ['title0, passage0_chunk0', 'title0, passage0_chunk1', ...]
    #   self.title: ['title0', 'title1', ...]
    #   self.title_chunk: ['title_chunk0', 'title_chunk1', ...]
    #   self.existing_literature: [['title0', 'existing_literature0'], ...]
    #   self.existing_literature_chunk: ['title0, passage0_chunk0', 'title0, passage0_chunk1', ...]
    def corpus_chunking_init(self):
        ## self.corpus --> self.corpus_chunk + self.title + self.title_chunk
        # self.title
        self.title = [self.corpus[i][0] for i in range(len(self.corpus))]
        assert len(self.title) == len(self.corpus)
        self.inspiration_title = [self.inspiration_corpus[i][0] for i in range(len(self.inspiration_corpus))]
        assert len(self.inspiration_title) == len(self.inspiration_corpus)
        if self.model_name == 'chatgpt':
            # Previous: add another "/ 2" since the input length of chatgpt is 4096, twice as vicuna
            # Now: no need to add "/ 2" since we are using "try except"
            word_limit_weight_background = 3/8
            word_limit_weight_inspiration = 3/8
            word_limit_weight_title = 5/16
            word_limit_weight_literature = 1/8
        else:
            word_limit_weight_background = 1/4
            word_limit_weight_inspiration = 3/16
            word_limit_weight_title = 3/16
            word_limit_weight_literature = 1/8
        # self.corpus_chunk, self.title_chunk, and self.existing_literature_chunk
        # only self.corpus_chunk is restricted with max_chunks_each_passage (default value for max_chunks_each_passage is 30)
        # self.corpus_chunk is mainly used for extracting backgroud (relatively coarse)
        self.corpus_chunk = chunk_passage(self.corpus, self.model_input_len, max_chunks_each_passage=self.args.max_chunks_each_passage, if_title_chunk=False, if_with_title=True, word_limit_weight=word_limit_weight_background)
        # self.background_corpus_chunk is mainly used in background_finder()
        self.background_corpus_chunk = chunk_passage(self.background_corpus, self.model_input_len, max_chunks_each_passage=self.args.max_chunks_each_passage, if_title_chunk=False, if_with_title=True, word_limit_weight=word_limit_weight_background)
        # self.inspiration_corpus_chunk is mainly used for extracting inspirations (relatively refined)
        self.inspiration_corpus_chunk = chunk_passage(self.inspiration_corpus, self.model_input_len, max_chunks_each_passage=self.args.max_chunks_each_passage, if_title_chunk=False, if_with_title=True, word_limit_weight=word_limit_weight_inspiration)
        # title_chunk probably is more intricate, so use less word_limit_weight
        # here we concat all titles (from all corpus instead of only background_corpus_chunk or inspiration_corpus_chunk) into one passage and then chunk it --- we don't want to miss any title so we set a large max_chunks_each_passage
        self.title_chunk = chunk_passage(self.title, self.model_input_len, max_chunks_each_passage=10000, if_title_chunk=True, if_with_title=False, word_limit_weight=word_limit_weight_title)
        self.inspiration_title_chunk = chunk_passage(self.inspiration_title, self.model_input_len, max_chunks_each_passage=10000, if_title_chunk=True, if_with_title=False, word_limit_weight=word_limit_weight_title)
        # here we use as many chunks for existing literature as possible (since BM25 is not computationally expensive)
        # if_with_title == False since the accuracy of extracting title from pdf is low
        self.existing_literature_chunk = chunk_passage(self.existing_literature, self.model_input_len, max_chunks_each_passage=10000, if_title_chunk=False, if_with_title=False, word_limit_weight=word_limit_weight_literature)
        print("len(self.existing_literature_chunk): ", len(self.existing_literature_chunk))


    # Function: the general usage of llm.generate()
    # input_txt: 'text'; reply: 'text'
    def llm_generation(self, input_txt, module_name=""):
        if self.model_name != "chatgpt":
            # gpt2 is used for finding bugs
            if self.model_name == 'gpt2':
                min_new_tokens = 5
                max_new_tokens = 15
            elif "hypothesis_generator_first" in module_name:
                min_new_tokens = 15
                max_new_tokens = 256
            # Even with vicuna-13b, feedback modules are higly possible to generate non-sense;
            #   in such case, if not min_new_tokens = 0, the 48 of '\n' do not help with the regeneration process
            elif module_name == "deductive_consistency_evaluator" or module_name == "indiscriminate_confirmation_handler" or module_name == "generalization_checker" or module_name == "novelty_detector":
                min_new_tokens = 0
                max_new_tokens = 192
            else:
                min_new_tokens = 15
                max_new_tokens = 192
            # print("input_txt: ", input_txt)
            input = self.tokenizer(input_txt, return_tensors="pt")
            input = input.to(0)
            # input = input.to(device1)
            # input.to(device1)
            generate_ids = self.model.generate(input.input_ids, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens, temperature=0.7, do_sample=True, top_p=1.0)
            reply = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        else:
            sleep_time = 0.25
            # the while loop is used to avoid the rate limit set by openai
            while (time.time() - self.prev_api_usage_time) <= sleep_time:
                time.sleep(sleep_time/2)
            openai.api_key = self.api_key
            if "hypothesis_generator" in module_name:
                max_tokens = 1220
            elif module_name == "novelty_detector":
                max_tokens = 288
            elif module_name == "inspiration_title_suggestor":
                max_tokens = 320
            elif module_name == "inspiration_passage_retriever" or module_name == "hypothesis_suggstor":
                max_tokens = 512
            else:
                max_tokens = 288
            # To prevent api error
            if_api_completed = False
            while if_api_completed == False:
                try:
                    response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[{"role": "user", "content": input_txt}],
                    top_p=0.90,
                    temperature=0.90,
                    max_tokens=max_tokens)
                    reply = response["choices"][0]['message']['content']
                    if_api_completed = True
                except:
                    print("OpenAI reach its rate limit")
                    time.sleep(sleep_time)
            self.prev_api_usage_time = time.time()
        return reply


    # Function: control when to start/stop to use which module
    #   control the usage of bkg_insp_controller() and CoLM_controller()
    def main_controller(self):
        # the max number of backgroud to find is len(self.background_corpus_chunk)
        if self.if_close_domain == 0:
            max_steps = min(self.num_background_for_hypotheses, len(self.background_corpus_chunk))
        else:
            max_steps = min(self.num_background_for_hypotheses, len(self.background_golden))
        print("max_steps: ", max_steps)
        for cur_id_background in range(max_steps):
            # if debug indirect feedbacks, skip the former parts
            ## Perform tomato-base
            if self.if_only_indirect_feedback == 0:
                # cur_background: 'background'
                # cur_background_eval: 'background_eval'
                # cur_inspirations: ['inspiration0', 'inspiration1', ...]
                # cur_inspirations_eval: ['inspiration0_eval', 'inspiration1_eval', ...]
                if self.if_baseline == 0:
                    cur_background, cur_background_eval, cur_title_matched, cur_title_matched_self_eval, cur_inspirations, cur_inspirations_eval = self.bkg_insp_controller()
                elif self.if_baseline == 1 or self.if_baseline == 2 or self.if_baseline == 3:
                    cur_background, cur_background_eval, cur_title_matched, cur_title_matched_self_eval, cur_inspirations, cur_inspirations_eval = self.bkg_insp_controller_baseline()
                else:
                    raise NotImplementedError
                # self.save_important_variables()
                # cur_hypotheses / cur_hypotheses_reasoning_process: [[hyp0_itr0, hyp0_itr1, hyp0_itr2, ...], [hyp1_itr0, hyp1_itr1, hyp1_itr2, ...], ...]
                # cur_feedbacks_hypotheses: {'hypothesis0': ['consistent feedback', 'reality feedback', 'novelty feedback'], ...}
                cur_hypotheses, cur_hypotheses_reasoning_process, cur_feedbacks_hypotheses = self.CoLM_controller(cur_background, cur_background_eval, cur_inspirations, cur_inspirations_eval)
                # self.save_important_variables()
                # # Lets think how to utilize these feedbacks later
                # cur_feedback_inspirations = self.inspiration_changer(cur_background, cur_inspirations, cur_hypotheses, cur_feedbacks_hypotheses)
                # cur_feedback_background = self.background_changer(cur_background, cur_inspirations, cur_hypotheses, cur_feedbacks_hypotheses)
                if self.model_name != "chatgpt":
                    print_nvidia_smi()
                self.save_important_variables()
            ## Do NOT perform tomato-base because tomato-base has been performed in this checkpoint
            elif self.if_only_indirect_feedback == 1:
                cur_background, cur_background_eval, cur_title_matched, cur_title_matched_self_eval, cur_inspirations, cur_inspirations_eval, cur_hypotheses, cur_hypotheses_reasoning_process, cur_feedbacks_hypotheses = load_variables_for_debug(self, self.args.output_dir, cur_id_background)
            ## Do NOT perform tomato-base, but at least tomato-base + past feedback
            elif self.if_only_indirect_feedback == 2:
                assert self.if_baseline == 0
                assert self.if_close_domain == 0
                assert self.if_indirect_feedback == 1
                cur_background, cur_background_eval, if_selfEval_matched_bkg = self.background_finder_wrapper()
                cur_title_matched, cur_title_matched_self_eval = [''], ['']
                cur_inspirations, cur_inspirations_eval, cur_hypotheses, cur_hypotheses_reasoning_process, cur_feedbacks_hypotheses = None, None, None, None, None
            else:
                raise NotImplementedError
                # print("cur_feedbacks_hypotheses: ", cur_feedbacks_hypotheses)
            ## indirect feedbacks
            if self.if_indirect_feedback:
                # inspiration changer
                cur_hypotheses_newInsp, cur_hypotheses_reasoning_process_newInsp, cur_feedbacks_hypotheses_newInsp = self.inspiration_changer_controller(cur_background, cur_background_eval, cur_title_matched, cur_title_matched_self_eval, cur_inspirations, cur_inspirations_eval, cur_hypotheses, cur_hypotheses_reasoning_process, cur_feedbacks_hypotheses)
                # background changer
                self.save_important_variables()
        print("main_controller finished!")


    # INPUT:
    #   cur_background: 'background'
    #   cur_background_eval: 'background_eval'
    #   cur_title_matched: ['existing title0', 'existing title1', ...]
    #   cur_title_matched_self_eval: ['existing title0 eval', 'existing title1 eval', ...]
    #   cur_inspirations: ['inspiration0', 'inspiration1', ...]
    #   cur_inspirations_eval: ['inspiration0_eval', 'inspiration1_eval', ...]
    #   cur_hypotheses / cur_hypotheses_reasoning_process: [[hyp0_itr0, hyp0_itr1, hyp0_itr2, ...], [hyp1_itr0, hyp1_itr1, hyp1_itr2, ...], ...]
    #   cur_feedbacks_hypotheses: {'hypothesis0': ['consistent feedback', 'reality feedback', 'novelty feedback'], ...}
    # OUTPUT:
    #   cur_changed_hypotheses: 'hypothesis-latest'
    #   cur_changed_hypotheses_reasoning_process: 'hypothesis_reasoning-lastest'
    #   cur_changed_feedbacks_hypotheses: {'hypothesis-latest': ['consistent feedback', 'reality feedback', 'novelty feedback']}
    def inspiration_changer_controller(self, cur_background, cur_background_eval, cur_title_matched, cur_title_matched_self_eval, cur_inspirations, cur_inspirations_eval, cur_hypotheses, cur_hypotheses_reasoning_process, cur_feedbacks_hypotheses):
        # cur_title_problems: ['problem0', 'problem1', ...]
        # cur_title_suggestions: ['suggest0', 'suggest1', ...]
        cur_title_problems, cur_title_suggestions, if_title_problem_suggestion_matched = self.inspiration_title_suggestor(cur_background, cur_background_eval, cur_title_matched, cur_title_matched_self_eval, cur_hypotheses, cur_hypotheses_reasoning_process, cur_feedbacks_hypotheses)
        # cur_changed_title_matched: ['existing title0', 'existing title1', ...]
        # cur_changed_title_matched_self_eval: ['existing title0 eval', 'existing title1 eval', ...]
        cur_changed_title_matched, cur_changed_title_matched_self_eval, if_changed_title_eval_matched = self.inspiration_title_changer(cur_background, cur_background_eval, cur_title_matched, cur_title_matched_self_eval, cur_hypotheses, cur_hypotheses_reasoning_process, cur_feedbacks_hypotheses, cur_title_problems, cur_title_suggestions)
        print("cur_changed_title_matched: ", cur_changed_title_matched)
        print("cur_changed_title_matched_self_eval: ", cur_changed_title_matched_self_eval)
        # cur_inspirations: ['inspiration0', 'inspiration1', ...]
        # cur_inspirations_self_eval: ['inspiration0_eval', 'inspiration1_eval', ...]
        cur_changed_inspirations, cur_changed_inspirations_self_eval = self.inspiration_passage_retriever(cur_changed_title_matched, cur_changed_title_matched_self_eval, cur_background, cur_background_eval, prompt_mode=1)
        # cur_changed_hypotheses / cur_changed_hypotheses_reasoning_process: [[hyp0_itr0, hyp0_itr1, hyp0_itr2, ...], [hyp1_itr0, hyp1_itr1, hyp1_itr2, ...], ...]
        # cur_changed_feedbacks_hypotheses: {'hypothesis0': ['consistent feedback', 'reality feedback', 'novelty feedback'], ...}
        cur_changed_hypotheses, cur_changed_hypotheses_reasoning_process, cur_changed_feedbacks_hypotheses = self.CoLM_controller(cur_background, cur_background_eval, cur_changed_inspirations, cur_changed_inspirations_self_eval)
        return cur_changed_hypotheses, cur_changed_hypotheses_reasoning_process, cur_changed_feedbacks_hypotheses


    # COMMENT:
    #   Current title_suggestor can not provide helpful suggestion on 'Problem:' and 'Suggestion:';
    #       Therefore we manually provide 'Problem:' and 'Suggestion:' now to give helpful suggestion (the manual feedback is according to the previous generated hypotheses and their feedbacks)
    # INPUT:
    #   cur_background: 'background'
    #   cur_background_eval: 'background_eval'
    #   cur_title_matched: ['existing title0', 'existing title1', ...]
    #   cur_title_matched_self_eval: ['existing title0 eval', 'existing title1 eval', ...]
    #   cur_hypotheses / cur_hypotheses_reasoning_process: [[hyp0_itr0, hyp0_itr1, hyp0_itr2, ...], [hyp1_itr0, hyp1_itr1, hyp1_itr2, ...], ...]
    #   cur_feedbacks_hypotheses: {'hypothesis0': ['reality feedback', 'novelty feedback', 'clarity feedback'], ...}
    # OUTPUT:
    #   cur_problem: ['problem0', 'problem1', ...] if if_selfEval_matched == True else [raw_generation]
    #   cur_suggestion: ['suggest0', 'suggest1', ...] if if_selfEval_matched == True else [""]
    #   if_selfEval_matched: True or False
    # def inspiration_title_suggestor(self, cur_background, cur_background_eval, cur_title_matched, cur_title_matched_self_eval, cur_hypotheses, cur_hypotheses_reasoning_process, cur_feedbacks_hypotheses):
    #     if_with_eval = self.if_self_eval_module['inspiration_title_suggestor']
    #     assert if_with_eval == True or if_with_eval == False
    #     # prompts
    #     pre_prompt, mid_prompt, post_prompt = prompts_for_tomato_modules(self.model_name, 'inspiration_title_suggestor', if_with_eval=if_with_eval)
    #     assert len(mid_prompt) == 2
    #     # previous_titles_for_input
    #     if self.if_self_eval_module['inspiration_title_retriever'] == True:
    #         assert len(cur_title_matched) == len(cur_title_matched_self_eval)
    #         assert len(cur_title_matched) >= 1
    #         previous_titles_for_input = ''
    #         for cur_title_id in range(len(cur_title_matched)):
    #             previous_titles_for_input += cur_title_matched[cur_title_id] + '. ' + cur_title_matched_self_eval[cur_title_id]
    #     else:
    #         previous_titles_for_input = '. '.join(cur_title_matched) + '.'
    #     # input_txt
    #     cur_hypotheses_and_feedbacks = ''
    #
    #     # only use one hyp and hyp_eval, otherwise it's a bit lengthy --> but it will cause every feedback to be "other titles not relevant to the specific hypothesis"; to prevent being lengthy we only adopt one aspect of hypothesis feedbacks (valid / novel / clear) (which could possibly also stress on the weak aspects)
    #     # for i in range(1):
    #     for i in range(len(cur_hypotheses)):
    #         if self.if_ban_selfeval:
    #             cur_hypotheses_and_feedbacks += cur_hypotheses[i][-2] + "\n"
    #         else:
    #             # cur_hypotheses_and_feedbacks += cur_hypotheses[i][-2] + "\n" + '\n'.join(cur_feedbacks_hypotheses[cur_hypotheses[i][-2]]) + "\n"
    #             # To prevent lenthy we only select novelty feedback now
    #             cur_hypotheses_and_feedbacks += cur_hypotheses[i][-2] + "\n" + cur_feedbacks_hypotheses[cur_hypotheses[i][-2]][1] + "\n"
    #     input_txt = pre_prompt + cur_background + cur_background_eval + mid_prompt[0] + previous_titles_for_input + mid_prompt[1] + cur_hypotheses_and_feedbacks + post_prompt
    #     print("input_txt for inspiration_title_suggestor: \n", input_txt)
    #     cur_generation = self.llm_generation(input_txt, 'inspiration_title_suggestor')
    #     # cur_problem: ['problem0', 'problem1', ...] if if_selfEval_matched == True; else ['raw_generation']
    #     # cur_suggestion: ['suggest0', 'suggest1', ...] if if_selfEval_matched == True; else [""]
    #     cur_problem, cur_suggestion, if_selfEval_matched = pick_from_generation(self.model_name, cur_generation, post_prompt, if_with_eval=if_with_eval, keyword_key_generation=self.keyword_key_generation['inspiration_title_suggestor'], keyword_key_generation_eval=self.keyword_key_generation_eval['inspiration_title_suggestor'])
    #     print("cur_problem: \n", cur_problem)
    #     print("cur_suggestion: \n", cur_suggestion)
    #     print("if_selfEval_matched: ", if_selfEval_matched)
    #     return cur_problem, cur_suggestion, if_selfEval_matched

    def inspiration_title_suggestor(self, cur_background, cur_background_eval, cur_title_matched, cur_title_matched_self_eval, cur_hypotheses, cur_hypotheses_reasoning_process, cur_feedbacks_hypotheses):
        if_with_eval = self.if_self_eval_module['inspiration_title_suggestor']
        assert if_with_eval == True or if_with_eval == False
        cur_problem = ['Previous selected titles might have resulted in a less novel hypothese generation. Because inspirations that are used to generate hypotheses are extracted from the business reports with the corresponding selected titles. In general, if the inspirations are less directly related to the given academic background, then the resulting generated hypothese are more likely to be novel. Therefore if the selected titles are very related to the research background, then trivial (less novel) hypotheses might be generated.']
        if if_with_eval:
            cur_suggestion = ['When selecting the titles, maybe try to select those that are less directly related to the given background.']
        else:
            cur_suggestion = ['']
        if_selfEval_matched = True
        return cur_problem, cur_suggestion, if_selfEval_matched


    # INPUT:
    #   cur_background: 'background'
    #   cur_background_eval: 'background_eval'
    #   cur_title_matched: ['existing title0', 'existing title1', ...]
    #   cur_title_matched_self_eval: ['existing title0 eval', 'existing title1 eval', ...]
    #   cur_hypotheses / cur_hypotheses_reasoning_process: [[hyp0_itr0, hyp0_itr1, hyp0_itr2, ...], [hyp1_itr0, hyp1_itr1, hyp1_itr2, ...], ...]
    #   cur_feedbacks_hypotheses: {'hypothesis0': ['consistent feedback', 'reality feedback', 'novelty feedback'], ...}
    #   cur_title_problems: ['problem0', 'problem1', ...] if if_selfEval_matched == True else [raw_generation]
    #   cur_title_suggestions: ['suggest0', 'suggest1', ...] if if_selfEval_matched == True else [""]
    # OUTPUT:
    #   title_collections: ['existing title0', 'existing title1', ...]
    #   title_collections_eval: ['existing title0 eval', 'existing title1 eval', ...]
    def inspiration_title_changer(self, cur_background, cur_background_eval, cur_title_matched, cur_title_matched_self_eval, cur_hypotheses, cur_hypotheses_reasoning_process, cur_feedbacks_hypotheses, cur_title_problems, cur_title_suggestions):
        if_with_eval = self.if_self_eval_module['inspiration_title_changer']
        assert if_with_eval == True or if_with_eval == False
        # prompts
        pre_prompt, mid_prompt, post_prompt = prompts_for_tomato_modules(self.model_name, 'inspiration_title_changer', if_with_eval=if_with_eval)
        assert len(mid_prompt) == 2
        # previous_titles_for_input
        if self.if_self_eval_module['inspiration_title_retriever'] == True:
            assert len(cur_title_matched) == len(cur_title_matched_self_eval)
            assert len(cur_title_matched) >= 1
            previous_titles_for_input = ''
            for cur_title_id in range(len(cur_title_matched)):
                previous_titles_for_input += cur_title_matched[cur_title_id] + '. ' + cur_title_matched_self_eval[cur_title_id]
        else:
            previous_titles_for_input = '. '.join(cur_title_matched) + '.'
        # previous_titles_problems_suggestions_for_input
        assert len(cur_title_problems) >= 1
        previous_titles_problems_suggestions_for_input = ''
        if self.if_self_eval_module['inspiration_title_suggestor'] == True:
            assert len(cur_title_problems) == len(cur_title_suggestions)
            for cur_title_problem_id in range(len(cur_title_problems)):
                previous_titles_problems_suggestions_for_input += cur_title_problems[cur_title_problem_id] + " " + cur_title_suggestions[cur_title_problem_id]
        else:
            for cur_title_problem_id in range(len(cur_title_problems)):
                previous_titles_problems_suggestions_for_input += cur_title_problems[cur_title_problem_id]
        # title_collections: ['existing title0', 'existing title1', ...]
        title_collections = []
        title_collections_eval = []
        # # cur_hypotheses_and_feedbacks
        # cur_hypotheses_and_feedbacks = ''
        # for i in range(len(cur_hypotheses)):
        # only use one hyp and hyp_eval, otherwise it's a bit lengthy
        # for i in range(1):
        #     if self.if_ban_selfeval:
        #         cur_hypotheses_and_feedbacks += cur_hypotheses[i][-2] + "\n"
        #     else:
        #         cur_hypotheses_and_feedbacks += cur_hypotheses[i][-2] + "\n" + '\n'.join(cur_feedbacks_hypotheses[cur_hypotheses[i][-2]]) + "\n"
        for cur_title_chunk_id in range(len(self.inspiration_title_chunk)):
            cur_title_chunk = self.inspiration_title_chunk[cur_title_chunk_id]
            # input_txt = pre_prompt + cur_background + cur_background_eval + mid_prompt[0] + previous_titles_for_input + mid_prompt[1] + cur_hypotheses_and_feedbacks + mid_prompt[2] + previous_titles_problems_suggestions_for_input + mid_prompt[3] + cur_title_chunk + post_prompt
            # input_txt = pre_prompt + cur_background + mid_prompt[0] + previous_titles_problems_suggestions_for_input + mid_prompt[1] + cur_title_chunk + post_prompt
            input_txt = pre_prompt + cur_background + cur_background_eval + mid_prompt[0] + previous_titles_problems_suggestions_for_input + mid_prompt[1] + cur_title_chunk + post_prompt
            if cur_title_chunk_id == 0:
                print("input_txt for inspiration_title_changer: \n", input_txt)
            cur_generation = self.llm_generation(input_txt, 'inspiration_title_changer')
            cur_title_split, cur_title_eval_split, if_selfEval_matched = pick_from_generation(self.model_name, cur_generation, post_prompt, if_with_eval=if_with_eval, keyword_key_generation=self.keyword_key_generation['inspiration_title_changer'], keyword_key_generation_eval=self.keyword_key_generation_eval['inspiration_title_changer'], module_name='inspiration_title_changer')
            cur_title_matched, cur_title_matched_eval = match_existing_title_given_title_generation(cur_title_split, cur_title_eval_split, if_selfEval_matched, if_with_eval, self.title)
            # find_passages_with_titles() in self.inspiration_passage_retriever() will filter repeated titles, so no need to worry about repetiton issues here
            title_collections += cur_title_matched
            title_collections_eval += cur_title_matched_eval
        if if_with_eval == True:
            assert len(title_collections) == len(title_collections_eval)
        return title_collections, title_collections_eval, if_selfEval_matched




    # Function: control the usage of background_finder() and inspiration_passage_retriever()
    #   given self.corpus_chunk, find the next one possible background (also note self.bkg_corpus_chunk_noter) and all possible inspirations for the background
    # Output:
    #   cur_background: 'background'
    #   cur_inspirations: ['inspiration0', 'inspiration1', ...]
    #   cur_title_matched: ['existing title0', 'existing title1', ...]
    #   cur_title_matched_self_eval: ['existing title0 eval', 'existing title1 eval', ...]
    def bkg_insp_controller(self):
        if self.if_close_domain == 0:
            # find background
            cur_background, cur_background_self_eval, if_selfEval_matched_bkg = self.background_finder_wrapper()
            print("\ncur_background: \n", cur_background)
            print("\ncur_background_self_eval: \n", cur_background_self_eval)
            # find inspiration (only titles of possible passages that possibly contain inspirations)
            # cur_title_matched: ['existing title0', 'existing title1', ...]
            # cur_title_matched_self_eval: ['existing title0 eval', 'existing title1 eval', ...] or ['', '', ...]
            cur_title_matched, cur_title_matched_self_eval = self.inspiration_title_retriever(cur_background, cur_background_self_eval)
            print("\ncur_title_matched: \n", cur_title_matched)
            print("\ncur_title_matched_self_eval: \n", cur_title_matched_self_eval)
            # cur_inspirations: ['inspiration0', 'inspiration1', ...]
            # cur_inspirations_self_eval: ['inspiration0_eval', 'inspiration1_eval', ...]
            cur_inspirations, cur_inspirations_self_eval = self.inspiration_passage_retriever(cur_title_matched, cur_title_matched_self_eval, cur_background, cur_background_self_eval)
            print("\ncur_inspirations: \n", cur_inspirations)
            print("\ncur_inspirations_self_eval: \n", cur_inspirations_self_eval)
            # assume cur_inspirations is not none
            assert len(cur_inspirations) > 0
            print("Background and inspiration are found successfully")
        else:
            cur_background, cur_background_self_eval, cur_title_matched, cur_title_matched_self_eval, cur_inspirations, cur_inspirations_self_eval = self.close_domain_bkg_insp_wrapper()
        return cur_background, cur_background_self_eval, cur_title_matched, cur_title_matched_self_eval, cur_inspirations, cur_inspirations_self_eval


    # Output:
    #   cur_background: 'background'
    #   cur_inspirations: ['inspiration0', 'inspiration1', ...]
    #   cur_title_matched: ['existing title0', 'existing title1', ...]
    #   cur_title_matched_self_eval: ['existing title0 eval', 'existing title1 eval', ...]
    def bkg_insp_controller_baseline(self):
        assert self.if_baseline == 1 or self.if_baseline == 2 or self.if_baseline == 3
        print("bkg_insp_controller_baseline() runs")
        # cur_background
        cur_bkg_chunk = self.background_corpus_chunk[self.bkg_corpus_chunk_noter]
        cur_bkg_chunk_split = cur_bkg_chunk.split('.')
        lucky_bkg_index = np.random.randint(0, len(cur_bkg_chunk_split)-2)
        cur_background = cur_bkg_chunk_split[lucky_bkg_index] + ". " + cur_bkg_chunk_split[lucky_bkg_index+1] + ". " + cur_bkg_chunk_split[lucky_bkg_index+2] + "."
        cur_background_eval = ''
        if self.if_baseline == 2:
            num_insp_for_baseline = 0
            cur_title_matched, cur_title_matched_self_eval, cur_inspirations, cur_inspirations_eval = [""], [""], [""], [""]
        elif self.if_baseline == 3:
            num_insp_for_baseline = 6
            lucky_title_index = np.random.choice(np.arange(len(self.title)), num_insp_for_baseline, replace=False)
            # cur_title_matched, cur_title_matched_self_eval
            cur_title_matched = []
            for cur_id in lucky_title_index:
                cur_title_matched.append(self.title[cur_id])
            cur_title_matched_self_eval = ["" for i in range(len(cur_title_matched))]
            assert len(cur_title_matched) == num_insp_for_baseline
            # cur_inspirations
            cur_inspirations = []
            passage_collections_chunks, passage_collections_chunks_title_eval = find_passages_with_titles(cur_title_matched, cur_title_matched_self_eval, self.corpus_chunk)
            if not len(passage_collections_chunks) == num_insp_for_baseline:
                print("Warning: len(passage_collections_chunks): {}; num_insp_for_baseline: {}".format(len(passage_collections_chunks), num_insp_for_baseline))
            for cur_pasg in passage_collections_chunks:
                cur_splitted_pasge_chunk = cur_pasg.split(".")
                cur_insp = ""
                cnt_while_loop = 0
                while len(cur_insp) <= 90 and cnt_while_loop < 50:
                    lucky_pasg_sent_index = np.random.randint(0, len(cur_splitted_pasge_chunk))
                    cur_insp = cur_splitted_pasge_chunk[lucky_pasg_sent_index] + '.'
                    cnt_while_loop += 1
                cur_inspirations.append(cur_insp)
            cur_inspirations_eval = ["" for i in range(len(cur_inspirations))]
            if not len(cur_inspirations) == num_insp_for_baseline:
                print("Warning: len(cur_inspirations): {}; num_insp_for_baseline: {}".format(len(cur_inspirations), num_insp_for_baseline))
        elif self.if_baseline == 1:
            # cur_title_matched
            num_insp_for_baseline = 5
            cur_title_matched = find_simi_score_using_BM25(cur_background, self.title, num_insp_for_baseline)
            cur_title_matched_self_eval = ["" for i in range(len(cur_title_matched))]
            assert len(cur_title_matched) == num_insp_for_baseline
            # cur_inspirations
            passage_collections_chunks, passage_collections_chunks_title_eval = find_passages_with_titles(cur_title_matched, cur_title_matched_self_eval, self.corpus_chunk)
            if not len(passage_collections_chunks) == num_insp_for_baseline:
                print("passage_collections_chunks: ", passage_collections_chunks)
                print("len(passage_collections_chunks): ", len(passage_collections_chunks))
                raise Exception
            cur_inspirations = []
            for cur_id in range(len(passage_collections_chunks)):
                cur_splitted_pasge_chunk = passage_collections_chunks[cur_id].split('.')
                cur_insp = find_simi_score_using_BM25(cur_background, cur_splitted_pasge_chunk, 1)
                assert len(cur_insp) == 1
                cur_inspirations.append(cur_insp[0] + ". ")
            assert len(cur_inspirations) == num_insp_for_baseline
            cur_inspirations_eval = ["" for i in range(len(cur_inspirations))]
        else:
            raise NotImplementedError
        # self.bkg_corpus_chunk_noter
        self.bkg_corpus_chunk_noter += 1
        # save valuables
        self.background.append(cur_background)
        self.background_self_eval.append(cur_background_eval)
        if cur_background not in self.selected_titles:
            self.selected_titles[cur_background] = [cur_title_matched]
            assert cur_background not in self.selected_titles_self_eval
            self.selected_titles_self_eval[cur_background] = [cur_title_matched_self_eval]
        else:
            self.selected_titles[cur_background].append(cur_title_matched)
            assert cur_background in self.selected_titles_self_eval
            self.selected_titles_self_eval[cur_background].append(cur_title_matched_self_eval)
        if cur_background not in self.inspiration:
            self.inspiration[cur_background] = [cur_inspirations]
            assert cur_background not in self.inspiration_self_eval
            self.inspiration_self_eval[cur_background] = [cur_inspirations_eval]
        else:
            self.inspiration[cur_background].append(cur_inspirations)
            assert cur_background in self.inspiration_self_eval
            self.inspiration_self_eval[cur_background].append(cur_inspirations_eval)
        return cur_background, cur_background_eval, cur_title_matched, cur_title_matched_self_eval, cur_inspirations, cur_inspirations_eval


    # read background and inspiration from golden annotation and save object variables
    def close_domain_bkg_insp_wrapper(self):
        assert len(self.background_golden) == len(self.inspiration_golden)
        # cur_background, cur_background_self_eval
        cur_background, cur_background_self_eval = "", ""
        cur_background_list = self.background_golden[self.bkg_corpus_chunk_noter]
        assert len(cur_background_list) >= 1
        for i in range(len(cur_background_list)):
            cur_background += "\nBackground {}: \n".format(i+1) + cur_background_list[i]
        # cur_title_matched, cur_title_matched_self_eval
        cur_title_matched, cur_title_matched_self_eval = [], []
        # cur_inspirations, cur_inspirations_self_eval
        cur_inspirations = self.inspiration_golden[self.bkg_corpus_chunk_noter]
        cur_inspirations_self_eval = ["" for i in range(len(cur_inspirations))]
        # self.bkg_corpus_chunk_noter
        self.bkg_corpus_chunk_noter += 1
        # save valuables
        self.background.append(cur_background)
        self.background_self_eval.append(cur_background_self_eval)
        if cur_background not in self.selected_titles:
            self.selected_titles[cur_background] = [cur_title_matched]
            assert cur_background not in self.selected_titles_self_eval
            self.selected_titles_self_eval[cur_background] = [cur_title_matched_self_eval]
        else:
            self.selected_titles[cur_background].append(cur_title_matched)
            assert cur_background in self.selected_titles_self_eval
            self.selected_titles_self_eval[cur_background].append(cur_title_matched_self_eval)
        if cur_background not in self.inspiration:
            self.inspiration[cur_background] = [cur_inspirations]
            assert cur_background not in self.inspiration_self_eval
            self.inspiration_self_eval[cur_background] = [cur_inspirations_self_eval]
        else:
            self.inspiration[cur_background].append(cur_inspirations)
            assert cur_background in self.inspiration_self_eval
            self.inspiration_self_eval[cur_background].append(cur_inspirations_self_eval)
        return cur_background, cur_background_self_eval, cur_title_matched, cur_title_matched_self_eval, cur_inspirations, cur_inspirations_self_eval


    # Function: iterate over self.corpus_chunk from self.bkg_corpus_chunk_noter to find one not-none backgroud for further usage
    # Output:
    #   cur_background/cur_background_self_eval: 'text' (if find background else None: cannot check whether backgroud counts yet)
    #   if_selfEval_matched_bkg: True / False
    def background_finder_wrapper(self):
        for cur_chunk_id in range(self.bkg_corpus_chunk_noter, len(self.background_corpus_chunk)):
            cur_chunk = self.background_corpus_chunk[cur_chunk_id]
            cur_background, cur_background_self_eval, if_selfEval_matched_bkg = self.background_finder(cur_chunk)
            # select a backgroud and eval from the list (cur_background and cur_background_self_eval)
            if self.if_self_eval_module['background_finder'] == True:
                assert len(cur_background) == len(cur_background_self_eval)
            if len(cur_background) > 1:
                print("Warning: multiple background generated. \ncur_background:", cur_background)
                assert if_selfEval_matched_bkg == True
            # Q: currently when len(cur_background) > 1 we just select the first background and leave the left alone
            cur_background, cur_background_self_eval = cur_background[0], cur_background_self_eval[0]
            # TD: possibly add an evaluator for cur_background and check whether continue by the backgroud evaluator
            if cur_background != None:
                self.bkg_corpus_chunk_noter = cur_chunk_id+1
                self.background.append(cur_background)
                self.background_self_eval.append(cur_background_self_eval)
                break
            else:
                raise Exception("cur_background == None")
        return cur_background, cur_background_self_eval, if_selfEval_matched_bkg


    # Function: (one step, consume one chunk of corpus) use self.model to find possible background to add to self.background
    # Input: cur_chunk: 'chunk'
    # Output: cur_background/cur_background_self_eval: 'text' (if find background else None: cannot check whether backgroud counts yet)
    # To consider: maybe also evaluate the generated background
    def background_finder(self, cur_chunk):
        if_with_eval = self.if_self_eval_module['background_finder']
        assert if_with_eval == True or if_with_eval == False
        pre_prompt, mid_prompt, post_prompt = prompts_for_tomato_modules(self.model_name, 'background_finder', if_with_eval=if_with_eval)
        input_txt = pre_prompt + cur_chunk + post_prompt
        # print("\ninput_txt in background_finder: \n", input_txt)
        cur_generation = self.llm_generation(input_txt, 'background_finder')
        # cur_background: ['key_item0', 'key_item1', ...] if if_selfEval_matched == True; else ['raw_generation']
        # cur_background_self_eval: ['key_item0_eval', 'key_item1_eval', ...] or ["", "", ...]
        cur_background, cur_background_self_eval, if_selfEval_matched = pick_from_generation(self.model_name, cur_generation, post_prompt, if_with_eval=if_with_eval, keyword_key_generation=self.keyword_key_generation['background_finder'], keyword_key_generation_eval=self.keyword_key_generation_eval['background_finder'], module_name='background_finder')
        return cur_background, cur_background_self_eval, if_selfEval_matched


    # Function: (one step or multiple steps, consume all titles which might contain multiple chunks) given self.model, self.background and all titles, to find possible titles whose corresponding passage might contain inspirations
    # Input:
    #   cur_background/cur_background_eval: 'text'
    #   self.inspiration_title_chunk: ['title_chunk0', 'title_chunk1', ...]
    # Output:
    #   title_collections: ['existing title0', 'existing title1', ...]
    #   title_collections_eval: ['existing title0 eval', 'existing title1 eval', ...] or ['', '', ...] (if if_with_eval == False)
    def inspiration_title_retriever(self, cur_background, cur_background_eval):
        if_with_eval = self.if_self_eval_module['inspiration_title_retriever']
        assert if_with_eval == True or if_with_eval == False
        pre_prompt, mid_prompt, post_prompt = prompts_for_tomato_modules(self.model_name, 'inspiration_title_retriever', if_with_eval=if_with_eval)
        # title_collections: ['existing title0', 'existing title1', ...]
        title_collections = []
        title_collections_eval = []
        for cur_title_chunk_id in range(len(self.inspiration_title_chunk)):
            cur_title_chunk = self.inspiration_title_chunk[cur_title_chunk_id]
            input_txt = pre_prompt + cur_background + cur_background_eval + mid_prompt + cur_title_chunk + post_prompt
            cur_generation = self.llm_generation(input_txt, 'inspiration_title_retriever')
            cur_title_split, cur_title_eval_split, if_selfEval_matched = pick_from_generation(self.model_name, cur_generation, post_prompt, if_with_eval=if_with_eval, keyword_key_generation=self.keyword_key_generation['inspiration_title_retriever'], keyword_key_generation_eval=self.keyword_key_generation_eval['inspiration_title_retriever'], module_name='inspiration_title_retriever')
            cur_title_matched, cur_title_matched_eval = match_existing_title_given_title_generation(cur_title_split, cur_title_eval_split, if_selfEval_matched, if_with_eval, self.title)
            title_collections += cur_title_matched
            title_collections_eval += cur_title_matched_eval
        # save variables
        if cur_background not in self.selected_titles:
            self.selected_titles[cur_background] = [title_collections]
            assert cur_background not in self.selected_titles_self_eval
            self.selected_titles_self_eval[cur_background] = [title_collections_eval]
        else:
            self.selected_titles[cur_background].append(title_collections)
            assert cur_background in self.selected_titles_self_eval
            self.selected_titles_self_eval[cur_background].append(title_collections_eval)
        return title_collections, title_collections_eval


    # Function: (one step or multiple steps, consume one passage which might contain multiple chunks) given self.model, self.background and multiple passages, to find possible inspirations for the background
    #   better to control the total length of the returned value (cur_inspirations) -- not too long and not too short
    # Input:
    #   title_collections: ['existing title0', 'existing title1', ...]
    #   title_collections_eval: ['existing title0 eval', 'existing title1 eval', ...] or ['', '', ...]
    #   background/background_eval: 'text'
    #   prompt_mode: 0: no past feedbacks; 1: within past feedbacks
    #   self.corpus_chunk: ['title0, passage0_chunk0', 'title0, passage0_chunk1', ...]
    # Intermediate:
    #   passage_collections_chunks: ['title0, passage0_chunk0', 'title0, passage0_chunk1', ...], where the length of chunks follows self.model_input_len
    #   passage_collections_chunks_title_eval: ['corresponding title eval', ...], some 'eval' could be '' if no enough confidence
    # Output:
    #   cur_inspirations: ['inspiration0', 'inspiration1', ...]
    #   cur_inspirations_eval: ['inspiration0_eval', 'inspiration1_eval', ...]
    def inspiration_passage_retriever(self, title_collections, title_collections_eval, background, background_eval, prompt_mode=0):
        # if_with_eval
        if_with_eval = self.if_self_eval_module['inspiration_passage_retriever']
        assert if_with_eval == True or if_with_eval == False
        # passage_collections_chunks, passage_collections_chunks_title_eval
        # here the title in passage_collections_chunks_title_eval can be titles from both background and inspiration webpage
        passage_collections_chunks, passage_collections_chunks_title_eval = find_passages_with_titles(title_collections, title_collections_eval, self.corpus_chunk)
        assert len(passage_collections_chunks) > 0
        # cur_inspirations, cur_inspirations_eval
        cur_inspirations, cur_inspirations_eval = [], []
        # only use one hyp and hyp_eval, otherwise it's a bit lengthy
        for cur_chk_id, cur_chk in enumerate(passage_collections_chunks):
            cur_chk_ttl = passage_collections_chunks[cur_chk_id].split('.\n')[0]
            cur_chk_ttl_eval = passage_collections_chunks_title_eval[cur_chk_id]
            pre_prompt, mid_prompt, post_prompt = prompts_for_tomato_modules(self.model_name, 'inspiration_passage_retriever', if_with_eval=if_with_eval, prompt_mode=prompt_mode)
            if self.if_ban_selfeval:
                assert background_eval == "" and cur_chk_ttl_eval == ""
            input_txt = pre_prompt + background + background_eval + mid_prompt[0] + cur_chk + mid_prompt[1] + cur_chk_ttl_eval + post_prompt
            cur_generation = self.llm_generation(input_txt, 'inspiration_passage_retriever')
            # TD: not using if_selfEval_matched
            cur_insp_split, cur_insp_eval_split, if_selfEval_matched = pick_from_generation(self.model_name, cur_generation, post_prompt, if_with_eval=if_with_eval, keyword_key_generation=self.keyword_key_generation['inspiration_passage_retriever'], keyword_key_generation_eval=self.keyword_key_generation_eval['inspiration_passage_retriever'], module_name='inspiration_passage_retriever')
            # print("\ncur_chk_ttl: ", cur_chk_ttl)
            # print("cur_chk_ttl_eval: ", cur_chk_ttl_eval)
            # print("cur_insp_split from cur_chk_ttl", cur_insp_split)
            # print("cur_insp_eval_split from cur_chk_ttl", cur_insp_eval_split)
            cur_inspirations += cur_insp_split
            cur_inspirations_eval += cur_insp_eval_split
        # save valuables
        if background not in self.inspiration:
            self.inspiration[background] = [cur_inspirations]
            assert background not in self.inspiration_self_eval
            self.inspiration_self_eval[background] = [cur_inspirations_eval]
        else:
            self.inspiration[background].append(cur_inspirations)
            assert background in self.inspiration_self_eval
            self.inspiration_self_eval[background].append(cur_inspirations_eval)
        print("New inspirations are updated successfully")
        return cur_inspirations, cur_inspirations_eval


    # Function: control the flow (and possble feedbacks) within CoLM
    # Input:
    #   cur_background: 'background'
    #   cur_background_eval: 'background_eval'
    #   cur_inspirations: ['inspiration0', 'inspiration1', ...]
    #   cur_inspirations_eval: ['inspiration0_eval', 'inspiration1_eval', ...]
    # Output:
    #   hypotheses_collections / hypotheses_reasoning_collections: [[hyp0_itr0, hyp0_itr1, hyp0_itr2, ...], [hyp1_itr0, hyp1_itr1, hyp1_itr2, ...], ...]
    #   hypotheses_CoLM_feedbacks: {'hypothesis0': ['consistent feedback', 'reality feedback', 'novelty feedback'], ...}
    def CoLM_controller(self, cur_background, cur_background_eval, cur_inspirations, cur_inspirations_eval):
        if self.args.if_hypothesis_suggstor == 1:
            cur_suggestions, cur_suggestions_eval, if_hypothesis_suggstor_matched = self.hypothesis_suggstor(cur_background, cur_background_eval, cur_inspirations, cur_inspirations_eval)
        elif self.args.if_hypothesis_suggstor == 0:
            cur_suggestions = None
        else:
            raise NotImplementedError
        # cur_hypotheses: ['hypothesis_generation0', 'hypothesis_generation1', ...]
        cur_hypotheses, cur_hypotheses_reasoning_process, if_hypothesis_reasoning_matched = self.hypothesis_generator(cur_background, cur_background_eval, cur_inspirations, cur_inspirations_eval, cur_suggestions)
        if self.if_self_eval_module['hypothesis_generator'] == True:
            assert len(cur_hypotheses) == len(cur_hypotheses_reasoning_process)
        if if_hypothesis_reasoning_matched == False:
            assert len(cur_hypotheses) == 1 and cur_hypotheses_reasoning_process == [""]
        print("\ncur_hypotheses: \n", cur_hypotheses)
        print("\ncur_hypotheses_reasoning_process: \n", cur_hypotheses_reasoning_process)
        # hypotheses_collections: just to note down all iteration of generated hypotheses in case they would be used
        # hypotheses_collections: [[hyp0_itr0, hyp0_itr1, hyp0_itr2, ...], [hyp1_itr0, hyp1_itr1, hyp1_itr2, ...], ...]
        hypotheses_collections = [[cur_hypotheses[id]] for id in range(len(cur_hypotheses))]
        hypotheses_reasoning_collections = [[cur_hypotheses_reasoning_process[id]] for id in range(len(cur_hypotheses_reasoning_process))]
        hypotheses_CoLM_feedbacks = {}
        # hypotheses_collections += cur_hypotheses
        # hypotheses_reasoning_collections += cur_hypotheses_reasoning_process
        for cur_hyp_id in range(len(cur_hypotheses)):
            for cur_cnt_feedback in range(self.num_CoLM_feedback_times):
                print("############## round {} ##############".format(cur_cnt_feedback))
                # cur_*_feedback: {'hypothesis0': *_feedback0, 'hypothesis1': *_feedback1, ...}
                if cur_cnt_feedback == 0:
                    hypothesis_to_focus_for_feedback = cur_hypotheses[cur_hyp_id]
                else:
                    # we assume there's only one returned hypothesis given feedbacks
                    hypothesis_to_focus_for_feedback = cur_regene_hypotheses[0]
                # cur_consistency_feedback = self.deductive_consistency_evaluator(cur_background, cur_inspirations, hypothesis_to_focus_for_feedback)
                cur_reality_feedback = self.indiscriminate_confirmation_handler(hypothesis_to_focus_for_feedback)
                cur_novelty_feedback = self.novelty_detector(cur_inspirations, hypothesis_to_focus_for_feedback)
                cur_specification_feedback = self.specification_detector(hypothesis_to_focus_for_feedback)
                cur_feedbacks_hypotheses = unify_feedbacks_to_format([cur_reality_feedback, cur_novelty_feedback, cur_specification_feedback])
                # hypotheses_CoLM_feedbacks[hypothesis_to_focus_for_feedback] = cur_feedbacks_hypotheses
                hypotheses_CoLM_feedbacks.update(cur_feedbacks_hypotheses)
                print("\ncur_feedbacks_hypotheses: \n", cur_feedbacks_hypotheses)
                cur_regene_hypotheses, cur_regene_hypotheses_reasoning_process, if_hypothesis_reasoning_matched = self.hypothesis_generator(cur_background, cur_background_eval, cur_inspirations, cur_inspirations_eval, cur_suggestions, hypothesis_to_focus_for_feedback, cur_feedbacks_hypotheses[hypothesis_to_focus_for_feedback])
                if not (len(cur_regene_hypotheses) == 1 and len(cur_regene_hypotheses_reasoning_process) == 1):
                    print("Warning: multiple hypotheses generated given feedbacks, cur_regene_hypotheses:\n {}; cur_regene_hypotheses_reasoning_process:\n {}\n, if_hypothesis_reasoning_matched: {}".format(cur_regene_hypotheses, cur_regene_hypotheses_reasoning_process, if_hypothesis_reasoning_matched))
                else:
                    print("\ncur_hypotheses: (cur_regene_hypotheses) \n", cur_regene_hypotheses)
                    print("\ncur_regene_hypotheses_reasoning_process: \n", cur_regene_hypotheses_reasoning_process)
                # In re-generation of hypothesis, only collect the first hypothesis to avoid complexity
                hypotheses_collections[cur_hyp_id].append(cur_regene_hypotheses[0])
                hypotheses_reasoning_collections[cur_hyp_id].append(cur_regene_hypotheses_reasoning_process[0])
        # self.hypothesis
        # assert cur_background not in self.hypothesis
        if self.if_self_eval_module['hypothesis_generator'] == True:
            assert len(hypotheses_collections) == len(hypotheses_reasoning_collections)
        ## save variables
        # hypotheses_collections: [[hyp0_itr0, hyp0_itr1, hyp0_itr2, ...], [hyp1_itr0, hyp1_itr1, hyp1_itr2, ...], ...]
        # hypotheses_CoLM_feedbacks: {'hyp0_itr0': ['consistency_feedback0', 'reality_feedback', 'novelty_feedback'], ...}
        if cur_background in self.hypothesis:
            self.hypothesis[cur_background].append(hypotheses_collections)
            assert cur_background in self.hypothesis_reasoning_process
            self.hypothesis_reasoning_process[cur_background].append(hypotheses_reasoning_collections)
            assert cur_background in self.hypothesis_CoLM_internal_feedback
            self.hypothesis_CoLM_internal_feedback[cur_background].append(hypotheses_CoLM_feedbacks)
        else:
            self.hypothesis[cur_background] = [hypotheses_collections]
            assert cur_background not in self.hypothesis_reasoning_process
            self.hypothesis_reasoning_process[cur_background] = [hypotheses_reasoning_collections]
            assert cur_background not in self.hypothesis_CoLM_internal_feedback
            self.hypothesis_CoLM_internal_feedback[cur_background] = [hypotheses_CoLM_feedbacks]
        print("hypotheses and feedbacks are found successfully")
        return hypotheses_collections, hypotheses_reasoning_collections, hypotheses_CoLM_feedbacks


    # cur_suggestions: ['cur_suggestions']
    # cur_suggestions_eval = ['']
    # if_selfEval_matched = False
    def hypothesis_suggstor(self, cur_background, cur_background_eval, cur_inspirations, cur_inspirations_eval):
        module_name = 'hypothesis_suggstor'
        if_with_eval = self.if_self_eval_module[module_name]
        assert if_with_eval == True or if_with_eval == False
        pre_prompt, mid_prompt, post_prompt = prompts_for_tomato_modules(self.model_name, module_name, if_with_eval=if_with_eval)
        # cur_inspirations_for_input
        cur_inspirations_for_input = ''
        for cur_insp_id in range(len(cur_inspirations)):
            cur_inspirations_for_input += "\nInspiration {}: \n".format(cur_insp_id+1) + cur_inspirations[cur_insp_id] + '\n'
        # input_txt
        input_txt = pre_prompt + cur_background + mid_prompt + cur_inspirations_for_input + post_prompt
        cur_generation = self.llm_generation(input_txt, module_name)
        cur_suggestions, cur_suggestions_eval, if_selfEval_matched = pick_from_generation(self.model_name, cur_generation, post_prompt, if_with_eval=if_with_eval, keyword_key_generation=self.keyword_key_generation[module_name], keyword_key_generation_eval=self.keyword_key_generation_eval[module_name], module_name=module_name)
        # save valuables
        if cur_background not in self.suggestion:
            self.suggestion[cur_background] = [cur_suggestions]
        else:
            self.suggestion[cur_background].append(cur_suggestions)
        print("New inspirations are updated successfully")
        return cur_suggestions, cur_suggestions_eval, if_selfEval_matched



    # Function: given self.moel, self.background, and self.inspiration, to generate possible hypotheses
    # Input:
    #   cur_background: 'background'
    #   cur_background_eval: 'background_eval'
    #   cur_inspirations: ['inspirations_from_chunk0', 'inspirations_from_chunk1', ...]
    #   cur_inspirations_eval: ['', '', ...]
    #   cur_suggestions: ['suggestions']
    #   pre_hypotheses: 'hypothesis_generation0'
    #   pre_hypotheses_feedbacks: ['consistent feedback', 'reality feedback', 'novelty feedback'] for 'hypothesis_generation0'
    # Output:
    #   cur_hypotheses: ['hypothesis_generation0', 'hypothesis_generation1', ...]
    #   cur_hypotheses_eval: ['hypothesis_generation0_eval', 'hypothesis_generation1_eval', ...]
    #   if_selfEval_matched: True or False, normally False
    def hypothesis_generator(self, cur_background, cur_background_eval, cur_inspirations, cur_inspirations_eval, cur_suggestions=None, pre_hypotheses=None, pre_hypotheses_feedbacks=None):
        if_with_eval = self.if_self_eval_module['hypothesis_generator']
        assert if_with_eval == True or if_with_eval == False
        ## prompt
        if self.args.if_hypothesis_suggstor == 1:
            module_name = 'hypothesis_generator_first_with_future_fdbk' if pre_hypotheses == None else 'hypothesis_generator_refine_with_future_fdbk'
            assert len(cur_suggestions) == 1
            cur_suggestions = cur_suggestions[0]
            assert isinstance(cur_suggestions, str)
        elif self.args.if_hypothesis_suggstor == 0:
            module_name = 'hypothesis_generator_first_without_future_fdbk' if pre_hypotheses == None else 'hypothesis_generator_refine_without_future_fdbk'
            assert cur_suggestions == None
        else:
            raise NotImplementedError
        pre_prompt, mid_prompt, post_prompt = prompts_for_tomato_modules(self.model_name, module_name, if_with_eval=if_with_eval, if_baseline=self.if_baseline)
        ## cur_inspirations_for_input
        cur_inspirations_for_input = ''
        # when if_baseline == 2, no inspirations are needed
        if self.if_baseline != 2:
            # this self_eval is in low quality, we use suggestions instead
            # if self.if_self_eval_module['inspiration_passage_retriever'] == True:
            #     assert len(cur_inspirations) == len(cur_inspirations_eval)
            #     assert len(cur_inspirations) >= 1
            #     for cur_insp_id in range(len(cur_inspirations)):
            #         cur_inspirations_for_input += "\nInspiration {}: \n".format(cur_insp_id+1) + cur_inspirations[cur_insp_id] + ' ' + cur_inspirations_eval[cur_insp_id] + '\n'
            # else:
            for cur_insp_id in range(len(cur_inspirations)):
                cur_inspirations_for_input += "\nInspiration {}: \n".format(cur_insp_id+1) + cur_inspirations[cur_insp_id] + '\n'
            ## preventing from input larger than input length limit
            if self.model_name == "gpt2":
                cur_inspirations_for_input = cur_inspirations_for_input[:200]
        ## input_txt
        if pre_hypotheses == None:
            assert pre_hypotheses_feedbacks == None
            if self.args.if_hypothesis_suggstor == 1:
                input_txt = pre_prompt + cur_background + mid_prompt[0] + cur_inspirations_for_input + mid_prompt[1] + cur_suggestions + post_prompt
            else:
                input_txt = pre_prompt + cur_background + mid_prompt + cur_inspirations_for_input + post_prompt
        else:
            assert pre_hypotheses_feedbacks != None
            pre_hypotheses_feedbacks = '\n\n'.join(pre_hypotheses_feedbacks)
            if self.args.if_hypothesis_suggstor == 1:
                input_txt = pre_prompt + cur_background + mid_prompt[0] + cur_inspirations_for_input + mid_prompt[1] + cur_suggestions + mid_prompt[2] + pre_hypotheses + mid_prompt[3] + pre_hypotheses_feedbacks + post_prompt
            else:
                # input_txt = pre_prompt + cur_background + mid_prompt[0] + cur_inspirations_for_input + mid_prompt[1] + pre_hypotheses + post_prompt
                input_txt = pre_prompt + cur_background + mid_prompt[0] + cur_inspirations_for_input + mid_prompt[1] + pre_hypotheses + mid_prompt[2] + pre_hypotheses_feedbacks + post_prompt
        ## cur_generation
        print("\ninput_txt for hypothesis: \n", input_txt)
        cur_generation = self.llm_generation(input_txt, 'hypothesis_generator')
        cur_hypotheses, cur_hypotheses_eval, if_selfEval_matched = pick_from_generation(self.model_name, cur_generation, post_prompt, if_with_eval=if_with_eval, keyword_key_generation=self.keyword_key_generation[module_name], keyword_key_generation_eval=self.keyword_key_generation_eval[module_name], module_name=module_name)
        print("len(cur_hypotheses): {}; len(cur_hypotheses_eval): {}".format(len(cur_hypotheses), len(cur_hypotheses_eval)))
        # print("cur_hypotheses: {};\ncur_hypotheses_eval: {}".format(cur_hypotheses, cur_hypotheses_eval))
        return cur_hypotheses, cur_hypotheses_eval, if_selfEval_matched


    # Function: given self.background, self.inspiration, self.hypothesis, to evaluate
    # Input:
    #   whether the hypotheses is consistent with given evidence (background, inspiration, and possible other source)
    #   cur_background: 'background'
    #   cur_inspirations: ['inspiration0', 'inspiration1', ...]
    #   cur_hypotheses: 'hypothesis0' (I think there should only be one string)
    # Output:
    #   consistency_feedback: {'hypothesis0': consistency_feedback0}
    def deductive_consistency_evaluator(self, cur_background, cur_inspirations, cur_hypotheses):
        if_with_eval = self.if_self_eval_module['deductive_consistency_evaluator']
        assert if_with_eval == True or if_with_eval == False
        pre_prompt, mid_prompt, post_prompt = prompts_for_tomato_modules(self.model_name, 'deductive_consistency_evaluator', if_with_eval=if_with_eval)
        cur_inspirations = '\n\n'.join(cur_inspirations)
        input_txt = pre_prompt + cur_background + mid_prompt[0] + cur_inspirations + mid_prompt[1] + cur_hypotheses + post_prompt
        cur_generation = self.llm_generation(input_txt, 'deductive_consistency_evaluator')
        cur_insp_split, cur_insp_eval_split, if_selfEval_matched = pick_from_generation(self.model_name, cur_generation, post_prompt, if_with_eval=if_with_eval, keyword_key_generation=self.keyword_key_generation['deductive_consistency_evaluator'], keyword_key_generation_eval=self.keyword_key_generation_eval['deductive_consistency_evaluator'], module_name='deductive_consistency_evaluator')
        if if_with_eval == True:
            raise NotImplementedError
        else:
            assert len(cur_insp_split) == 1
            consistency_feedback = {cur_hypotheses: cur_insp_split[0]}
            return consistency_feedback


    # Function: given self.hypothesis, to evaluate
    #   whether the hypothesis reflects the reality
    # Input:
    #   cur_hypotheses: 'hypothesis0'
    # Output:
    #   reality_feedback: {'hypothesis0': reality_feedback0}
    def indiscriminate_confirmation_handler(self, cur_hypotheses):
        if_with_eval = self.if_self_eval_module['indiscriminate_confirmation_handler']
        assert if_with_eval == True or if_with_eval == False
        pre_prompt, mid_prompt, post_prompt = prompts_for_tomato_modules(self.model_name, 'indiscriminate_confirmation_handler', if_with_eval=if_with_eval)
        input_txt = pre_prompt + cur_hypotheses + post_prompt
        cur_generation = self.llm_generation(input_txt, 'indiscriminate_confirmation_handler')
        cur_insp_split, cur_insp_eval_split, if_selfEval_matched = pick_from_generation(self.model_name, cur_generation, post_prompt, if_with_eval=if_with_eval, keyword_key_generation=self.keyword_key_generation['indiscriminate_confirmation_handler'], keyword_key_generation_eval=self.keyword_key_generation_eval['indiscriminate_confirmation_handler'], module_name='indiscriminate_confirmation_handler')
        if if_with_eval == True:
            raise NotImplementedError
        else:
            assert len(cur_insp_split) == 1
            reality_feedback = {cur_hypotheses: cur_insp_split[0]}
            return reality_feedback


    # # Function: given self.hypothesis, to evaluate
    # #   whether the hypothesis can be more general and thus has larger coverage scope
    # def generalization_checker(self):


    # Function: given self.hypothesis, to evaluate
    #   whether the hypothesis is novel and not been discovered before (maybe compare with relevant survey)
    # Input:
    #   cur_inspirations: ['inspiration0', 'inspiration1', ...]
    #   cur_hypotheses: 'hypothesis0'
    # Output:
    #   novelty_feedback: {'hypothesis0': novelty_feedback0}
    def novelty_detector(self, cur_inspirations, cur_hypotheses):
        if self.args.if_novelty_module_have_access_to_surveys == 1:
            ## find cur_matched_survey_chunk using BM25 to possibly help check novelty
            tokenized_all_survey = [survey.split(" ") for survey in self.existing_literature_chunk]
            bm25 = BM25Okapi(tokenized_all_survey)
            tokenized_cur_hypotheses = cur_hypotheses.split(" ")
            simi_scores = bm25.get_scores(tokenized_cur_hypotheses)
            assert len(simi_scores) == len(self.existing_literature_chunk)
            cur_survey_index = np.argmax(simi_scores)
            print("simi_score for cur_survey_paragraph: ", simi_scores[cur_survey_index])
            cur_matched_survey_chunk = self.existing_literature_chunk[cur_survey_index]
            print("cur_matched_survey_chunk: ", cur_matched_survey_chunk)
        ## normal module steps
        if_with_eval = self.if_self_eval_module['novelty_detector']
        assert if_with_eval == True or if_with_eval == False
        pre_prompt, mid_prompt, post_prompt = prompts_for_tomato_modules(self.model_name, 'novelty_detector', if_with_eval=if_with_eval, prompt_mode=self.args.if_novelty_module_have_access_to_surveys)
        cur_inspirations = '\n\n'.join(cur_inspirations)
        if self.args.if_novelty_module_have_access_to_surveys == 1:
            input_txt = pre_prompt + cur_hypotheses + mid_prompt[0] + cur_inspirations + mid_prompt[1] + cur_matched_survey_chunk + post_prompt
        else:
            input_txt = pre_prompt + cur_hypotheses + mid_prompt[0] + cur_inspirations + post_prompt
        # print("input_txt for novelty_detector: ", input_txt)
        cur_generation = self.llm_generation(input_txt, 'novelty_detector')
        cur_insp_split, cur_insp_eval_split, if_selfEval_matched = pick_from_generation(self.model_name, cur_generation, post_prompt, if_with_eval=if_with_eval, keyword_key_generation=self.keyword_key_generation['novelty_detector'], keyword_key_generation_eval=self.keyword_key_generation_eval['novelty_detector'], module_name='novelty_detector')
        # novelty_feedback
        cur_insp_concat = [cur_insp_split[i] + cur_insp_eval_split[i] for i in range(len(cur_insp_split))]
        novelty_feedback = {cur_hypotheses: "\n".join(cur_insp_concat)}
        return novelty_feedback

    # Function:
    # cur_hypotheses: 'hypothesis0'
    def specification_detector(self, cur_hypotheses):
        if_with_eval = self.if_self_eval_module['specification_detector']
        assert if_with_eval == True or if_with_eval == False
        pre_prompt, mid_prompt, post_prompt = prompts_for_tomato_modules(self.model_name, 'specification_detector', if_with_eval=if_with_eval)
        input_txt = pre_prompt + cur_hypotheses + post_prompt
        cur_generation = self.llm_generation(input_txt, 'specification_detector')
        cur_specific_feedback_split, cur_specific_feedback_eval_split, if_selfEval_matched = pick_from_generation(self.model_name, cur_generation, post_prompt, if_with_eval=if_with_eval, keyword_key_generation=self.keyword_key_generation['specification_detector'], keyword_key_generation_eval=self.keyword_key_generation_eval['specification_detector'], module_name='specification_detector')
        # specification_feedback
        cur_insp_concat = [cur_specific_feedback_split[i] + cur_specific_feedback_eval_split[i] for i in range(len(cur_specific_feedback_split))]
        specification_feedback = {cur_hypotheses: "\n".join(cur_insp_concat)}
        return specification_feedback

    # Function: given self.hypothesis and feedbacks for the hypothesis, to find feedbacks for the possible change of background
    # Input:
    #   cur_background: 'background'
    #   cur_inspirations: ['inspiration0', 'inspiration1', ...]
    #   cur_hypotheses: ['hypothesis0', 'hypothesis1', ...]
    #   cur_feedbacks_hypotheses: {'hypothesis0': ['consistent feedback', 'reality feedback', 'novelty feedback'], ...}
    # Output:
    #   TD: cur_feedback_background: {'hypothesis0': 'feedback_background', ...}
    def background_changer(self, cur_background, cur_inspirations, cur_hypotheses, cur_feedbacks_hypotheses):
        pass

        # return cur_feedback_background
