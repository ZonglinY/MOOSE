import pandas as pd


def prompts_for_evaluator_modules():
    # pre_prompt = "Given a not yet peer reviewed research hypothesis in business domain, try to evaluate the hypothesis from three aspects and give score according to evaluation guidelines provided below. All three aspects should be evaluated in a 4 point scale. \nAspect 1: Validness. \n4 points: the hypothesis totally reflects the reality; 3 points: The hypothesis should be minorly revised to totally reflect the reality; 2 points: The hypothesis should be majorly revised to totally reflect the reality; 1 point: the hypothesis should be edited completely to reflect the reality.\nAspect 2: Novelty. \n4 points: the hypothesis is completely novel and has not been proposed by any existing literature; 3 points: the hypothesis is mostly novel; 2 points: a small part of the hypothesis is novel; 1 point: the hypothesis is not novel at all and should have been proposed by existing business literature. \nAspect 3: Helpfulness. \n4 points: the hypothesis can be directly adopted by human researcher; 3 points: the hypothesis can be adopted after minor changes; 2 points: only a small part of the hypothesis is inspiring for human researchers to develop a new hypothesis; 1 point: the hypothesis is not helpful at all. \nThe hypothesis is:\n"
    pre_prompt = "Given a not yet peer reviewed research hypothesis in business domain, try to evaluate the research hypothesis from three research aspects and give score according to evaluation guidelines provided below. All three aspects should be evaluated in a 5 point scale." + "\nAspect 1: Validness. \n5 points: the hypothesis completely reflects the reality; 4 points: the hypothesis almost completely reflects the reality, but has only one or two minor conflictions that can be easily modified; 3 points: the hypothesis has at least one moderate conflict or several minor conflicts; 2 points: the hypothesis has at least one major confliction with the reality or only establishes in very rare circumstances that are not mentioned in this hypothesis; 1 point: the hypothesis completely violates the reality. " + "\nAspect 2: Novelty. \n5 points: the hypothesis is completely novel and has not been proposed by any existing literature; 4 points: the main argument or several sub-arguments of the hypothesis are novel; 3 points: the main argument is not novel, only one or two sub-arguments appear to be novel; 2 points: the full hypothesis is not novel, but the way it combines the topics can be inspiring for human researchers; 1 point: the hypothesis is not novel at all and not inspiring for human researchers. " + "\nAspect 3: Helpfulness. \n5 points: the hypothesis is novel, valid, clear and specific enough that it is itself a matural research hypothesis, and human researchers can directly adopt it for publication with no modifications needed; 4 points: the hypothesis is novel enough and can be directly adopted by human researcher for publication after minor modifications; 3 points: the hypothesis should be largely modified or reconstructed by human researcher to adopt it; 2 points: modifying this hypothesis might not deserve the efforts, but a small part of this hypothesis is inspiring for human researchers to develop a new hypothesis; 1 point: the hypothesis is not helpful and not inspiring at all. \nThe hypothesis is:\n"
    post_prompt = "\nPlease give a response to the initial question on scoring the hypothesis from three aspects. (response format: 'Validness score: \nConcise reason: \nNovelty score: \nConcise reason: \nHelpfulness score: \nConcise reason: \n')."
    return pre_prompt, post_prompt


# OUTPUT:
#   score_collection: ['score0', 'score1', 'score2']
#   score_reason_collection: ['reason0', 'reason1', 'reason2']
#   if_successful: True or False
def pick_score(cur_generation, input_txt):
    score_format = ['Validness score:', 'Novelty score:', 'Helpfulness score:']
    reason_format = 'Concise reason:'
    potential_scores = ['1', '2', '3', '4', '5']
    # score_collection, score_reason_collection
    cur_generation_split = cur_generation.split('\n')
    score_collection, score_reason_collection = [], []
    # format_mode: 0: Validness score: 2\nConcise Reason:; 1: Validness score:\n2 points\nConcise Reason:
    if_mode1_next_is_reason = 0
    for cur_sent in cur_generation_split:
        cur_if_succeed = 0
        # format_mode 1 reason
        if if_mode1_next_is_reason == 1:
            cur_sent = cur_sent.replace(reason_format, "").strip()
            if len(cur_sent) > 0:
                score_reason_collection.append(cur_sent)
                if_mode1_next_is_reason = 0
            else:
                raise Exception("Can't find reason for score: ", cur_generation_split)
        else:
            # normal reason
            if reason_format in cur_sent:
                cur_sent = cur_sent.replace(reason_format, "").strip()
                if len(cur_sent) > 0:
                    score_reason_collection.append(cur_sent)
                else:
                    if_mode1_next_is_reason = 1
            else:
                # normal score
                for cur_score_format in score_format:
                    if cur_score_format in cur_sent:
                        cur_score = cur_sent.replace(cur_score_format, "").replace("points", "").replace("point", "").strip()
                        if cur_score in potential_scores:
                            score_collection.append(cur_score)
                            cur_if_succeed = 1
                # format_mode 1 score
                if cur_if_succeed == 0:
                    cur_score_temp = cur_sent.replace(cur_score_format, "").replace("points", "").replace("point", "").strip()
                    # format_mode 1 score
                    if cur_score_temp in potential_scores:
                        score_collection.append(cur_score_temp)
    # if_successful
    if len(score_collection) == len(score_reason_collection) and len(score_collection) == 3:
        if_successful = True
    else:
        if_successful = False
        print("input_txt: ", input_txt)
        print("score_collection: ", score_collection)
        print("len(score_collection): ", len(score_collection))
        print("len(score_reason_collection): ", len(score_reason_collection))
        print("cur_generation: ", cur_generation)
        raise Exception()
    return score_collection, score_reason_collection, if_successful


# OUTPUT:
#   background_golden: [bkg0, bkg1, ...]
#   hypothese_golden: [hyp0, hyp1, ...]
def load_ground_truth_hypotheses(dataset_dir):
    raw_corpus = pd.read_excel(dataset_dir)
    # background_golden and hypothese_golden
    background_golden = []
    hypothese_golden = []
    for cur_data_id in range(len(raw_corpus)):
        # background_golden
        cur_data_bkg_golden = []
        for cur_ctnt in ['background_1', 'background_2']:
            cur_data_golden = raw_corpus['{}_golden'.format(cur_ctnt)][cur_data_id]
            if isinstance(cur_data_golden, str):
                if cur_data_golden.strip() != "":
                    cur_data_golden = cur_data_golden.strip()
                    cur_data_bkg_golden.append(cur_data_golden)
        cur_data_bkg_golden = "\n".join(cur_data_bkg_golden)
        background_golden.append(cur_data_bkg_golden)
        # hypothese_golden
        cur_data_hyp = raw_corpus['Main hypotheis'][cur_data_id]
        assert cur_data_hyp not in hypothese_golden
        hypothese_golden.append(cur_data_hyp)
    assert len(background_golden) == len(hypothese_golden)
    return background_golden, hypothese_golden
