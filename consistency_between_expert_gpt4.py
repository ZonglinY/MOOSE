import os
import torch
import numpy as np
import pandas as pd


def load_normal_order_expert_scores():
    root_data_dir = "./Checkpoints/expert_evaluation/"
    # read from annotated random file
    raw_corpus = pd.read_excel(os.path.join(root_data_dir, 'expert_evaluation_normal_order.xlsx'))
    full_list_of_hyp = []
    full_list_of_validness, full_list_of_novelty, full_list_of_helpfulness = [], [], []
    for cur_data_id in range(len(raw_corpus)):
        cur_data_hyp = raw_corpus["Hypothesis"][cur_data_id]
        cur_data_val = raw_corpus["Validness"][cur_data_id]
        cur_data_nov = raw_corpus["Novelty"][cur_data_id]
        cur_data_hep = raw_corpus["Helpfulness"][cur_data_id]
        full_list_of_hyp.append(cur_data_hyp)
        full_list_of_validness.append(cur_data_val)
        full_list_of_novelty.append(cur_data_nov)
        full_list_of_helpfulness.append(cur_data_hep)
    assert len(full_list_of_hyp) == len(full_list_of_validness)
    assert len(full_list_of_hyp) == len(full_list_of_novelty)
    assert len(full_list_of_hyp) == len(full_list_of_helpfulness)
    return full_list_of_hyp, full_list_of_validness, full_list_of_novelty, full_list_of_helpfulness


# start_end_id = [[0,5], [5,25], [25,50]]
# direct_or_indirect: 0 or 1
# ckpt_files: [file0, file1]
# itrs: [0] or [0,2,4]
# the order inside ckpt_files and start_end_id should from small to large
def load_gpt4_scores(ckpt_files, start_end_id, direct_or_indirect, itrs):
    ckpt_root_dir = "./Checkpoints/"
    hyp_file = "background_inspiration_hypotheses.pt"
    gpt4_score_files = ["automatic_evaluation_hypotheses_gpt4_{}_{}.pt".format(i, j) for i, j in start_end_id]
    ## gpt4_scores
    gpt4_scores = {}
    for cur_f in gpt4_score_files:
        # cur_gpt4_scores
        cur_gpt4_scores = None
        for cur_ckpt_file_id in range(len(ckpt_files)):
            try:
                cur_score_file = os.path.join(ckpt_root_dir, ckpt_files[cur_ckpt_file_id], cur_f)
                cur_gpt4_scores = torch.load(cur_score_file)
                break
            except:
                continue
        assert cur_gpt4_scores != None
        gpt4_scores.update(cur_gpt4_scores)
    ## picked_hyp_id_file: picked_hyp_ids
    picked_hyp_id_file = os.path.join(ckpt_root_dir, ckpt_files[0], "picked_hyp_id_{}.pt".format(direct_or_indirect))
    picked_hyp_ids = torch.load(picked_hyp_id_file)
    assert len(picked_hyp_ids) == len(gpt4_scores)
    ## backgrounds = []
    backgrounds = []
    for cur_file_id in range(len(ckpt_files)):
        # cur_data
        cur_file = os.path.join(ckpt_root_dir, ckpt_files[cur_file_id], hyp_file)
        cur_data = torch.load(cur_file)
        backgrounds += cur_data[2]
    assert len(backgrounds) == len(gpt4_scores)
    ## picked_gpt4_scores = []
    picked_gpt4_scores = []
    for bkg_id, bkg in enumerate(backgrounds):
        # cur_score: [[valid, novel, helpful], [valid, novel, helpful], [valid, novel, helpful]] or [[valid, novel, helpful]]
        cur_score = []
        for cur_i in itrs:
            cur_single_score = gpt4_scores[bkg][direct_or_indirect][picked_hyp_ids[bkg_id]][cur_i]
            # print("cur_single_score: ", cur_single_score)
            if len(cur_single_score) != 1 or len(cur_single_score[0]) != 3:
                print("Warning: cur_single_score: ", cur_single_score)
                cur_score.append(cur_single_score)
            else:
                cur_score.append(cur_single_score[0])
        assert len(cur_score) == 1 or len(cur_score) == 3
        picked_gpt4_scores.append(cur_score)
    assert len(picked_gpt4_scores) == len(gpt4_scores)
    return picked_gpt4_scores


# list_scores: [scores0, scores1, ...]; should be in order of expert evaluation .xlsx file
# OUTPUT:
#   full_list_of_validness, full_list_of_novelty, full_list_of_helpfulness
def unify_gpt4_scores(list_scores):
    assert len(list_scores) == 4
    full_list_of_validness, full_list_of_novelty, full_list_of_helpfulness = [], [], []
    for cur_bkg_id in range(len(list_scores[0])):
        for cur_scores in list_scores:
            for cur_data_id in range(len(cur_scores[cur_bkg_id])):
                if cur_scores[cur_bkg_id][cur_data_id] != []:
                    full_list_of_validness.append(cur_scores[cur_bkg_id][cur_data_id][0])
                    full_list_of_novelty.append(cur_scores[cur_bkg_id][cur_data_id][1])
                    full_list_of_helpfulness.append(cur_scores[cur_bkg_id][cur_data_id][2])
                else:
                    full_list_of_validness.append(np.nan)
                    full_list_of_novelty.append(np.nan)
                    full_list_of_helpfulness.append(np.nan)
    return full_list_of_validness, full_list_of_novelty, full_list_of_helpfulness


def consistency(list1, list2, if_hard_consistency):
    assert len(list1) == len(list2)
    assert if_hard_consistency == 0 or if_hard_consistency == 1
    consistency_score = []
    for cur_id in range(len(list1)):
        s1 = float(list1[cur_id])
        s2 = float(list2[cur_id])
        # print("s1: {}; s2: {}".format(s1, s2))
        if np.isnan(s1) or np.isnan(s2):
            continue
        abs_diff = abs(s1-s2)
        if if_hard_consistency == 0:
            if abs_diff == 0:
                consistency_score.append(1)
            elif abs_diff == 1:
                consistency_score.append(0.75)
            elif abs_diff == 2:
                consistency_score.append(0.50)
            elif abs_diff == 3:
                consistency_score.append(0.25)
            elif abs_diff == 4:
                consistency_score.append(0.0)
            else:
                raise Exception("s1: {}; s2: {}".format(s1, s2))
        else:
            if abs_diff == 0:
                consistency_score.append(1)
            else:
                consistency_score.append(0)
    print("abs(len(consistency_score)): ", abs(len(consistency_score)))
    assert abs(len(consistency_score) - 400) <= 5
    ave_consistency_score = sum(consistency_score) / len(consistency_score)
    return ave_consistency_score


    return consistency_score


def read_expert_scores():
    root_data_dir = "./Checkpoints/expert_evaluation/"
    # read from annotated random file
    raw_corpus = pd.read_excel(os.path.join(root_data_dir, 'expert_evaluation_normal_order.xlsx'))
    full_list_of_hyp = []
    full_list_of_validness, full_list_of_novelty, full_list_of_helpfulness = [], [], []
    for cur_data_id in range(len(raw_corpus)):
        cur_data_hyp = raw_corpus["Hypothesis"][cur_data_id]
        cur_data_val = raw_corpus["Validness"][cur_data_id]
        cur_data_nov = raw_corpus["Novelty"][cur_data_id]
        cur_data_hep = raw_corpus["Helpfulness"][cur_data_id]
        full_list_of_hyp.append(cur_data_hyp)
        full_list_of_validness.append(cur_data_val)
        full_list_of_novelty.append(cur_data_nov)
        full_list_of_helpfulness.append(cur_data_hep)
    assert len(full_list_of_hyp) == len(full_list_of_validness)
    assert len(full_list_of_hyp) == len(full_list_of_novelty)
    assert len(full_list_of_hyp) == len(full_list_of_helpfulness)
    return full_list_of_validness, full_list_of_novelty, full_list_of_helpfulness

def main():
    # if_hard_consistency: 0/1
    if_hard_consistency = 0
    # expert_hyp / expert_validness / expert_novelty / expert_helpfulness: []
    expert_hyp, expert_validness, expert_novelty, expert_helpfulness = load_normal_order_expert_scores()
    ## baseline ckpt
    ckpt_baseline2_0_50 = "chatgpt_50bkg_0itr_bkgnoter0_indirect0_onlyindirect0_close0_ban1_baseline2_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ## Tomato-base ckpts
    ckpt_tomato_base_0_25 = "chatgpt_25bkg_4itr_bkgnoter0_indirect0_onlyindirect0_close0_ban1_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ckpt_tomato_base_25_50 = "chatgpt_25bkg_4itr_bkgnoter25_indirect0_onlyindirect0_close0_ban1_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ## Tomato-past-future ckpts
    ckpt_tomato_pf_0_25 = "chatgpt_25bkg_4itr_bkgnoter0_indirect1_onlyindirect0_close0_ban0_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ckpt_tomato_pf_25_50 = "chatgpt_25bkg_4itr_bkgnoter25_indirect1_onlyindirect0_close0_ban0_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"

    gpt4_scores_baseline2 = load_gpt4_scores([ckpt_baseline2_0_50], [[0,50]], 0, [0])
    gpt4_scores_tomato_base = load_gpt4_scores([ckpt_tomato_base_0_25, ckpt_tomato_base_25_50], [[0,5], [5,25],[25,50]], 0, [0,2,4])
    gpt4_scores_tomato_pf_onlyf = load_gpt4_scores([ckpt_tomato_pf_0_25, ckpt_tomato_pf_25_50], [[0,5], [5,25],[25,50]], 0, [4])
    gpt4_scores_tomato_pf_bothpf = load_gpt4_scores([ckpt_tomato_pf_0_25, ckpt_tomato_pf_25_50], [[0,5], [5,25],[25,50]], 1, [0,2,4])
    assert len(gpt4_scores_baseline2) == 50
    assert len(gpt4_scores_tomato_base) == 50
    assert len(gpt4_scores_tomato_pf_onlyf) == 50
    assert len(gpt4_scores_tomato_pf_bothpf) == 50

    full_list_of_validness_gpt4, full_list_of_novelty_gpt4, full_list_of_helpfulness_gpt4 = unify_gpt4_scores([gpt4_scores_baseline2, gpt4_scores_tomato_base, gpt4_scores_tomato_pf_onlyf, gpt4_scores_tomato_pf_bothpf])
    print("len(full_list_of_validness_gpt4): ", len(full_list_of_validness_gpt4))
    full_list_of_validness_expert, full_list_of_novelty_expert, full_list_of_helpfulness_expert = read_expert_scores()

    consist_valid = consistency(full_list_of_validness_gpt4, full_list_of_validness_expert, if_hard_consistency)
    consist_novel = consistency(full_list_of_novelty_gpt4, full_list_of_novelty_expert, if_hard_consistency)
    consist_helpf = consistency(full_list_of_helpfulness_gpt4, full_list_of_helpfulness_expert, if_hard_consistency)

    print("if_hard_consistency: ", if_hard_consistency)
    print("consist_valid: {}; consist_novel: {}; consist_helpf: {}".format(consist_valid, consist_novel, consist_helpf))























if __name__ == "__main__":
    main()
    print("finished")
