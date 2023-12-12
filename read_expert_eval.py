import os
import torch
import numpy as np
import pandas as pd
np.set_printoptions(precision=3)

# annotated_score: [n, 8, 3]
def load_annotated_data(root_data_dir, expert_file):
    raw_corpus = pd.read_excel(os.path.join(root_data_dir, expert_file))
    annotated_score = []
    metrics = ['Validness', 'Novelty', 'Helpfulness']
    if_annotated = 1
    for cur_data_id in range(len(raw_corpus)):
        if if_annotated == 0:
            break
        for cur_id_ctnt, cur_ctnt in enumerate(metrics):
            cur_score = raw_corpus[cur_ctnt][cur_data_id]
            # print("cur_score: ", np.isnan(cur_score))
            if cur_id_ctnt == 0:
                # print("cur_score: ", cur_score)
                if np.isnan(cur_score) == False:
                    if cur_data_id % 8 == 0:
                        annotated_score.append([])
                    annotated_score[-1].append([cur_score])
                    assert cur_score >= 1 and cur_score <= 5
                else:
                    if_annotated = 0
                    break
            else:
                assert np.isnan(cur_score) == False
                assert cur_score >= 1 and cur_score <= 5
                annotated_score[-1][-1].append(cur_score)
    # print("annotated_score: ", annotated_score)
    annotated_score = np.array(annotated_score)
    print("annotated_score.shape: ", annotated_score.shape)
    # annotated_score: [n, 8, 3]
    return annotated_score


# annotated_score: [[n(num_data), 8, 3], ...]
# rand_order: [50, 4]
def final_result(annotated_score, rand_order):
    def final_result_single_file(annotated_score, rand_order):
        # baseline, tomato_base, tomato_future, tomato_fp
        all_scores = [[], [], [], []]
        for cur_bkg_id in range(annotated_score.shape[0]):
            # [2, 1, 0, 3]
            cur_rand_order = rand_order[cur_bkg_id]
            cur_cnted_score_id = 0
            for cur_rand_order_id in cur_rand_order:
                # three successive scores
                if cur_rand_order_id == 1 or cur_rand_order_id == 3:
                    # cur_scores: [3, 3]
                    cur_scores = annotated_score[cur_bkg_id][cur_cnted_score_id: cur_cnted_score_id+3]
                    cur_cnted_score_id += 3
                # one score
                else:
                    # cur_scores: [1, 3]
                    cur_scores = annotated_score[cur_bkg_id][cur_cnted_score_id: cur_cnted_score_id+1]
                    cur_cnted_score_id += 1
                all_scores[cur_rand_order_id].append(cur_scores)
                assert cur_cnted_score_id <= 8
        return all_scores

    # Aggregate scores from multiple files
    all_scores_across_files = [[], [], [], []]
    for cur_file_id in range(len(annotated_score)):
        cur_all_scores = final_result_single_file(annotated_score[cur_file_id], rand_order)
        for cur_model_id in range(len(all_scores_across_files)):
            all_scores_across_files[cur_model_id] += cur_all_scores[cur_model_id]

    baseline_score = np.array(all_scores_across_files[0])
    tomato_base_score = np.array(all_scores_across_files[1])
    tomato_future_score = np.array(all_scores_across_files[2])
    tomato_fp = np.array(all_scores_across_files[3])
    print("baseline_score.shape: ", baseline_score.shape)
    print("tomato_base_score.shape: ", tomato_base_score.shape)
    print("tomato_future_score.shape: ", tomato_future_score.shape)
    print("tomato_fp.shape: ", tomato_fp.shape)

    ave_baseline_score = np.mean(baseline_score, axis=0).squeeze()
    ave_tomato_base_score = np.mean(tomato_base_score, axis=0).squeeze()
    ave_tomato_future_score = np.mean(tomato_future_score, axis=0).squeeze()
    ave_tomato_fp = np.mean(tomato_fp, axis=0).squeeze()
    print("\nave_baseline_score: \n", ave_baseline_score)
    print("\nave_tomato_base_score: \n", ave_tomato_base_score)
    print("\nave_tomato_future_score: \n", ave_tomato_future_score)
    print("\nave_tomato_fp: \n", ave_tomato_fp)


def main():
    root_data_dir = "./Checkpoints/expert_evaluation/"
    expert_file_0 = "picked_hypotheses_for_expert_evaluation_rand_order_for_each_group_Junxian_labeled.xlsx"
    expert_file_1 = "expert_1_2.xlsx"
    annotated_score_0 = load_annotated_data(root_data_dir, expert_file_0)
    annotated_score_1 = load_annotated_data(root_data_dir, expert_file_1)
    # num_group_annotated = annotated_score.shape[0]
    rand_id_dir = "./Checkpoints/expert_evaluation/rand_order_for_each_group.pt"
    rand_order = torch.load(rand_id_dir)
    final_result([annotated_score_0, annotated_score_1], rand_order)














if __name__ == "__main__":
    main()
    print("finished")
