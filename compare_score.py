import os
import torch
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def read_score_file(model_name, start_id, end_id, ckpt_addr1):
    # h1
    # ckpt_dir = "/export/home/zonglin001/Checkpoints/Tomato/"
    ckpt_dir = "./Checkpoints/"
    hyp_file = "background_inspiration_hypotheses.pt"
    hyp_file_addr1_full = os.path.join(ckpt_dir, ckpt_addr1, hyp_file)
    h1 = torch.load(hyp_file_addr1_full)
    # s1_raw
    try:
        score_file = "automatic_evaluation_hypotheses_{}_{}_{}.pt".format(model_name, start_id, end_id)
        score_file_addr1_full = os.path.join(ckpt_dir, ckpt_addr1, score_file)
        s1_raw = torch.load(score_file_addr1_full)
    except:
        try:
            score_file = "automatic_evaluation_hypotheses_{}.pt".format(model_name)
            score_file_addr1_full = os.path.join(ckpt_dir, ckpt_addr1, score_file)
            s1_raw = torch.load(score_file_addr1_full)
        except:
            score_file = "automatic_evaluation_hypotheses.pt"
            score_file_addr1_full = os.path.join(ckpt_dir, ckpt_addr1, score_file)
            s1_raw = torch.load(score_file_addr1_full)
    return h1, s1_raw

# all_score: [[valid, novel, helpful], ...] (a numpy array)
def find_score(h, s, num_CoLM_feedback_times, start_id, end_id):
    background = h[2]
    all_score = []
    all_score_without_indirect, all_score_with_indirect = [], []
    all_score_without_indirect_itrs = [[] for i in range(num_CoLM_feedback_times+1)]
    all_score_with_indirect_itrs = [[] for i in range(num_CoLM_feedback_times+1)]
    # final_start_id, final_end_id
    if start_id != -1 and end_id != -1:
        if end_id - start_id == len(background):
            # print("Warning: adjust start_id and end_id to 0 and len(background)")
            final_start_id, final_end_id = 0, len(background)
        else:
            final_start_id, final_end_id = start_id, end_id
    else:
        final_start_id, final_end_id = 0, len(background)
    # begin the loop
    # print("final_start_id: {}; final_end_id: {}； len(background): {}".format(final_start_id, final_end_id, len(background)))
    cnt_not_fully_splitted_hyp = 0
    for cur_id_bkg in range(final_start_id,final_end_id):
        s_bkg = s[background[cur_id_bkg]]
        for cur_id_s_direct, cur_s_direct in enumerate(s_bkg):
            for cur_id_s_hyp, cur_s_hyp in enumerate(cur_s_direct):
                for cur_id_s_itr, cur_s_itr in enumerate(cur_s_hyp):
                    if cur_id_s_itr == num_CoLM_feedback_times+1:
                        # print("break: ", cur_id_s_itr)
                        break
                    # cur_score
                    if not (len(cur_s_itr) == 1 and len(cur_s_itr[0]) == 3):
                        # there's no score here because the hyp is not fully splitted (probably contain "\n\nReasoning process")
                        assert len(cur_s_itr) == 0
                        cnt_not_fully_splitted_hyp += 1
                        # continue
                        cur_score = [np.nan, np.nan, np.nan]
                    else:
                        # str to int
                        cur_score = []
                        for cur_str_score in cur_s_itr[0]:
                            cur_score.append(int(cur_str_score))
                    all_score.append(cur_score)
                    if cur_id_s_direct == 0:
                        all_score_without_indirect.append(cur_score)
                        all_score_without_indirect_itrs[cur_id_s_itr].append(cur_score)
                    elif cur_id_s_direct == 1:
                        all_score_with_indirect.append(cur_score)
                        all_score_with_indirect_itrs[cur_id_s_itr].append(cur_score)
                    else:
                        raise Exception
    # print("cnt_not_fully_splitted_hyp: ", cnt_not_fully_splitted_hyp)
    all_score_without_indirect = np.array(all_score_without_indirect)
    all_score_with_indirect = np.array(all_score_with_indirect)
    all_score_without_indirect_itrs = np.array(all_score_without_indirect_itrs)
    all_score_with_indirect_itrs = np.array(all_score_with_indirect_itrs)
    all_score = np.array(all_score)
    return all_score_without_indirect, all_score_with_indirect, all_score_without_indirect_itrs, all_score_with_indirect_itrs, all_score


# start_end_id_1: [[0,5], [5,25], [25,50]]
def read_file_find_score_concat_score(model_name, start_end_id_1, num_CoLM_feedback_times_1, ckpt_addr1_full):
    assert len(start_end_id_1) == len(ckpt_addr1_full)
    for cur_id in range(len(ckpt_addr1_full)):
        cur_ckpt_addr = ckpt_addr1_full[cur_id]
        cur_start_id, cur_end_id = start_end_id_1[cur_id]
        cur_h1, cur_s1_raw = read_score_file(model_name, cur_start_id, cur_end_id, cur_ckpt_addr)
        cur_score1_wo_ind, cur_score1_w_ind, cur_score1_wo_ind_itrs, cur_score1_w_ind_itrs, cur_score1_all = find_score(cur_h1, cur_s1_raw, num_CoLM_feedback_times_1, cur_start_id, cur_end_id)
        if cur_id == 0:
            concat_score1_wo_ind, concat_score1_w_ind, concat_score1_wo_ind_itrs, concat_score1_w_ind_itrs, concat_score1_all = cur_score1_wo_ind, cur_score1_w_ind, cur_score1_wo_ind_itrs, cur_score1_w_ind_itrs, cur_score1_all
        else:
            concat_score1_wo_ind = np.concatenate((concat_score1_wo_ind, cur_score1_wo_ind), axis=0)
            concat_score1_w_ind = np.concatenate((concat_score1_w_ind, cur_score1_w_ind), axis=0)
            concat_score1_wo_ind_itrs = np.concatenate((concat_score1_wo_ind_itrs, cur_score1_wo_ind_itrs), axis=1)
            concat_score1_w_ind_itrs = np.concatenate((concat_score1_w_ind_itrs, cur_score1_w_ind_itrs), axis=1)
            concat_score1_all = np.concatenate((concat_score1_all, cur_score1_all), axis=0)
    return concat_score1_wo_ind, concat_score1_w_ind, concat_score1_wo_ind_itrs, concat_score1_w_ind_itrs, concat_score1_all


def find_hyperparameter_for_display_results(model_name, method_name):
    assert method_name == "MOOSE_base" or method_name == "MOOSE" or method_name == "rand_background_baseline" or method_name == "rand_background_rand_inspiration_baseline" or method_name == "rand_background_BM25_inspiration_baseline" or method_name == "gpt35_background_gpt35_inspiration" or method_name == "groundtruth_background_groundtruth_inspiration" or method_name == "MOOSE_wo_ff1" or method_name == "MOOSE_wo_ff2" or method_name == "MOOSE_wo_survey" or method_name == "MOOSE_w_random_corpus"

    ## baseline ckpts
    ckpt_baseline1_0_50 = "chatgpt_50bkg_0itr_bkgnoter0_indirect0_onlyindirect0_close0_ban1_baseline1_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ckpt_baseline2_0_50 = "chatgpt_50bkg_0itr_bkgnoter0_indirect0_onlyindirect0_close0_ban1_baseline2_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ckpt_baseline3_0_50 = "chatgpt_50bkg_0itr_bkgnoter0_indirect0_onlyindirect0_close0_ban1_baseline3_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ## golden ckpt
    ckpt_golden_0_50 = "chatgpt_50bkg_4itr_bkgnoter0_indirect0_onlyindirect0_close1_ban1_baseline0_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ## Tomato-base ckpts
    ckpt_tomato_base_0_25 = "chatgpt_25bkg_4itr_bkgnoter0_indirect0_onlyindirect0_close0_ban1_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ckpt_tomato_base_25_50 = "chatgpt_25bkg_4itr_bkgnoter25_indirect0_onlyindirect0_close0_ban1_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ckpt_tomato_base_0_50_present_feedback_using_feedback =     "chatgpt_50bkg_4itr_bkgnoter0_indirect0_onlyindirect0_close0_ban1_baseline0_survey1_bkgInspPasgSwap0_hypSuggestor0_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ## Tomato-past-future ckpts
    ckpt_tomato_pf_0_25 = "chatgpt_25bkg_4itr_bkgnoter0_indirect1_onlyindirect0_close0_ban0_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ckpt_tomato_pf_25_50 = "chatgpt_25bkg_4itr_bkgnoter25_indirect1_onlyindirect0_close0_ban0_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ## Tomato-past-future variations ckpts
    ckpt_tomato_pf_0_50_with_selfeval_without_hypSuggestor =     "chatgpt_50bkg_4itr_bkgnoter0_indirect1_onlyindirect2_close0_ban0_baseline0_survey1_bkgInspPasgSwap0_hypSuggestor0_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ckpt_tomato_pf_0_50_without_selfeval_with_hypSuggestor =    "chatgpt_50bkg_4itr_bkgnoter0_indirect1_onlyindirect2_close0_ban1_baseline0_survey1_bkgInspPasgSwap0_hypSuggestor1_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ckpt_tomato_pf_0_50_noSurvey = "chatgpt_50bkg_4itr_bkgnoter0_indirect1_onlyindirect2_close0_ban0_baseline0_survey0_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ckpt_tomato_pf_0_50_bkg_insp_pasg_swap = "chatgpt_50bkg_4itr_bkgnoter0_indirect1_onlyindirect2_close0_ban0_baseline0_survey1_bkgInspPasgSwap1_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"

    if method_name == "MOOSE_base":
        start_end_id = [[0,5], [5,25], [25,50]]
        num_CoLM_feedback_times = 4
        ckpt_addr_full = [ckpt_tomato_base_0_25, ckpt_tomato_base_0_25, ckpt_tomato_base_25_50]
    elif method_name == "MOOSE":
        start_end_id = [[0,5], [5,25], [25,50]]
        num_CoLM_feedback_times = 4
        ckpt_addr_full = [ckpt_tomato_pf_0_25, ckpt_tomato_pf_0_25, ckpt_tomato_pf_25_50]
    elif method_name == "rand_background_baseline":
        start_end_id = [[0, 50]]
        num_CoLM_feedback_times = 0
        ckpt_addr_full = [ckpt_baseline2_0_50]
    elif method_name == "rand_background_rand_inspiration_baseline":
        start_end_id = [[0, 50]]
        num_CoLM_feedback_times = 0
        ckpt_addr_full = [ckpt_baseline3_0_50]
    elif method_name == "rand_background_BM25_inspiration_baseline":
        start_end_id = [[0, 50]]
        num_CoLM_feedback_times = 0
        ckpt_addr_full = [ckpt_baseline1_0_50]
    elif method_name == "gpt35_background_gpt35_inspiration":
        start_end_id = [[0,5], [5,25], [25,50]]
        num_CoLM_feedback_times = 4
        ckpt_addr_full = [ckpt_tomato_base_0_25, ckpt_tomato_base_0_25, ckpt_tomato_base_25_50]
    elif method_name == "groundtruth_background_groundtruth_inspiration":
        start_end_id = [[0, 50]]
        num_CoLM_feedback_times = 0
        ckpt_addr_full = [ckpt_golden_0_50]
    elif method_name == "MOOSE_wo_ff1":
        start_end_id = [[0, 50]]
        num_CoLM_feedback_times = 4
        ckpt_addr_full = [ckpt_tomato_pf_0_50_without_selfeval_with_hypSuggestor]
    elif method_name == "MOOSE_wo_ff2":
        start_end_id = [[0, 50]]
        num_CoLM_feedback_times = 4
        ckpt_addr_full = [ckpt_tomato_pf_0_50_with_selfeval_without_hypSuggestor]
    elif method_name == "MOOSE_wo_survey":
        start_end_id = [[0, 50]]
        num_CoLM_feedback_times = 4
        ckpt_addr_full = [ckpt_tomato_pf_0_50_noSurvey]
    elif method_name == "MOOSE_w_random_corpus":
        start_end_id = [[0, 50]]
        num_CoLM_feedback_times = 4
        ckpt_addr_full = [ckpt_tomato_pf_0_50_bkg_insp_pasg_swap]
    else:
        raise NotImplementedError

    return start_end_id, num_CoLM_feedback_times, ckpt_addr_full


def main():
    ## hyper-parameters
    # 'chatgpt' or 'gpt4'
    model_name = 'gpt4'
    # "MOOSE_base", "rand_background_baseline", "rand_background_rand_inspiration_baseline", "rand_background_BM25_inspiration_baseline", "gpt35_background_gpt35_inspiration", "MOOSE_wo_ff1", "MOOSE_wo_ff2", "MOOSE_wo_survey", "MOOSE_w_random_corpus"
    method_name1 = "MOOSE_base"
    # "MOOSE"
    method_name2 = "MOOSE"
    ## load data and find score
    start_end_id_1, num_CoLM_feedback_times_1, ckpt_addr1_full = find_hyperparameter_for_display_results(model_name, method_name1)
    start_end_id_2, num_CoLM_feedback_times_2, ckpt_addr2_full = find_hyperparameter_for_display_results(model_name, method_name2)
    score1_wo_ind, score1_w_ind, score1_wo_ind_itrs, score1_w_ind_itrs, score1_all = read_file_find_score_concat_score(model_name, start_end_id_1, num_CoLM_feedback_times_1, ckpt_addr1_full)
    score2_wo_ind, score2_w_ind, score2_wo_ind_itrs, score2_w_ind_itrs, score2_all = read_file_find_score_concat_score(model_name, start_end_id_2, num_CoLM_feedback_times_2, ckpt_addr2_full)

    ## result processing
    print("score1_wo_ind.shape: ", score1_wo_ind.shape)
    # print("score1_w_ind.shape: ", score1_w_ind.shape)
    print("score2_wo_ind.shape: ", score2_wo_ind.shape)
    print("score2_w_ind.shape: ", score2_w_ind.shape)

    ave_score1_wo_ind = np.nanmean(score1_wo_ind, axis=0)
    # ave_score1_w_ind = np.nanmean(score1_w_ind, axis=0)
    ave_score2_wo_ind = np.nanmean(score2_wo_ind, axis=0)
    ave_score2_w_ind = np.nanmean(score2_w_ind, axis=0)
    print("\nave_score1_wo_ind: ", ave_score1_wo_ind)
    # print("ave_score1_w_ind: ", ave_score1_w_ind)
    print("ave_score2_wo_ind: ", ave_score2_wo_ind)
    print("ave_score2_w_ind: ", ave_score2_w_ind)

    # score_all_itrs
    if method_name1 == "MOOSE_base" and method_name2 == "MOOSE":
        score_all_itrs = np.concatenate((score1_wo_ind_itrs, score2_wo_ind_itrs, score2_w_ind_itrs), axis=1)
        print("\nscore_all_itrs: ", score_all_itrs.shape)
        ave_score_all_itrs = np.nanmean(score_all_itrs, axis=1)
        print("ave_score_all_itrs: \n", ave_score_all_itrs)

    # # score_each_itrs
    if method_name1 == "gpt35_background_gpt35_inspiration":
        ave_score1_wo_ind_itrs = np.nanmean(score1_wo_ind_itrs, axis=1)
        # ave_score1_w_ind_itrs = np.nanmean(score1_w_ind_itrs, axis=1)
        ave_score2_wo_ind_itrs = np.nanmean(score2_wo_ind_itrs, axis=1)
        ave_score2_w_ind_itrs = np.nanmean(score2_w_ind_itrs, axis=1)

        print("\nscore1_wo_ind_itrs: ", score1_wo_ind_itrs.shape)
        # print("score1_w_ind_itrs: ", score1_w_ind_itrs.shape)
        print("score2_wo_ind_itrs: ", score2_wo_ind_itrs.shape)
        print("score2_w_ind_itrs: ", score2_w_ind_itrs.shape)

        print("\nave_score1_wo_ind_itrs: \n", ave_score1_wo_ind_itrs)
        # print("\nave_score1_w_ind_itrs: \n", ave_score1_w_ind_itrs)
        print("\nave_score2_wo_ind_itrs: \n", ave_score2_wo_ind_itrs)
        print("\nave_score2_w_ind_itrs: \n", ave_score2_w_ind_itrs)




if __name__ == "__main__":
    main()
    print("finished")
