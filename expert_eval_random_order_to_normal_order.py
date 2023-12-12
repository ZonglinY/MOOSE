import os
import torch
import numpy as np
import pandas as pd





def main(expert_file_rand_order="picked_hypotheses_for_expert_evaluation_rand_order_for_each_group_Junxian_labeled.xlsx", if_save_output_file=False):
    root_data_dir = "./Checkpoints/expert_evaluation/"
    # read from annotated random file
    raw_corpus = pd.read_excel(os.path.join(root_data_dir, expert_file_rand_order))
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
    # recover original order
    rand_order = torch.load("/export/home/zonglin001/Autonomous_Open_Domain_Hypothetical_Induction/Checkpoints/expert_evaluation/rand_order_for_each_group.pt")
    assert len(full_list_of_hyp) == len(rand_order) * 8
    full_list_of_hyp_normal_order = []
    full_list_of_validness_normal_order, full_list_of_novelty_normal_order, full_list_of_helpfulness_normal_order = [], [], []
    for cur_group_id in range(len(rand_order)):
        cur_hyp_rand_order = full_list_of_hyp[cur_group_id*8:(cur_group_id+1)*8]
        cur_val_rand_order = full_list_of_validness[cur_group_id*8:(cur_group_id+1)*8]
        cur_nov_rand_order = full_list_of_novelty[cur_group_id*8:(cur_group_id+1)*8]
        cur_hep_rand_order = full_list_of_helpfulness[cur_group_id*8:(cur_group_id+1)*8]
        cur_rand_order = rand_order[cur_group_id]
        assert len(cur_rand_order) == 4
        cur_hyp_normal_order = ["" for i in range(len(cur_hyp_rand_order))]
        cur_val_normal_order = ["" for i in range(len(cur_val_rand_order))]
        cur_nov_normal_order = ["" for i in range(len(cur_nov_rand_order))]
        cur_hep_normal_order = ["" for i in range(len(cur_hep_rand_order))]
        cur_group_hyp_id = 0
        for cur_id_order, cur_order in enumerate(cur_rand_order):
            # only 1 hyp
            if cur_order == 0:
                cur_hyp_normal_order[0:1] = cur_hyp_rand_order[cur_group_hyp_id:cur_group_hyp_id+1]
                cur_val_normal_order[0:1] = cur_val_rand_order[cur_group_hyp_id:cur_group_hyp_id+1]
                cur_nov_normal_order[0:1] = cur_nov_rand_order[cur_group_hyp_id:cur_group_hyp_id+1]
                cur_hep_normal_order[0:1] = cur_hep_rand_order[cur_group_hyp_id:cur_group_hyp_id+1]
                cur_group_hyp_id += 1
            elif cur_order == 1:
                cur_hyp_normal_order[1:4] = cur_hyp_rand_order[cur_group_hyp_id:cur_group_hyp_id+3]
                cur_val_normal_order[1:4] = cur_val_rand_order[cur_group_hyp_id:cur_group_hyp_id+3]
                cur_nov_normal_order[1:4] = cur_nov_rand_order[cur_group_hyp_id:cur_group_hyp_id+3]
                cur_hep_normal_order[1:4] = cur_hep_rand_order[cur_group_hyp_id:cur_group_hyp_id+3]
                cur_group_hyp_id += 3
            elif cur_order == 2:
                cur_hyp_normal_order[4:5] = cur_hyp_rand_order[cur_group_hyp_id:cur_group_hyp_id+1]
                cur_val_normal_order[4:5] = cur_val_rand_order[cur_group_hyp_id:cur_group_hyp_id+1]
                cur_nov_normal_order[4:5] = cur_nov_rand_order[cur_group_hyp_id:cur_group_hyp_id+1]
                cur_hep_normal_order[4:5] = cur_hep_rand_order[cur_group_hyp_id:cur_group_hyp_id+1]
                cur_group_hyp_id += 1
            elif cur_order == 3:
                cur_hyp_normal_order[5:8] = cur_hyp_rand_order[cur_group_hyp_id:cur_group_hyp_id+3]
                cur_val_normal_order[5:8] = cur_val_rand_order[cur_group_hyp_id:cur_group_hyp_id+3]
                cur_nov_normal_order[5:8] = cur_nov_rand_order[cur_group_hyp_id:cur_group_hyp_id+3]
                cur_hep_normal_order[5:8] = cur_hep_rand_order[cur_group_hyp_id:cur_group_hyp_id+3]
                cur_group_hyp_id += 3
            else:
                print("cur_order: ", cur_order)
                raise Exception
        assert cur_group_hyp_id == 8


        full_list_of_hyp_normal_order += cur_hyp_normal_order
        full_list_of_validness_normal_order += cur_val_normal_order
        full_list_of_novelty_normal_order += cur_nov_normal_order
        full_list_of_helpfulness_normal_order += cur_hep_normal_order

    # save
    columns = ["Hypothesis", "Validness", "Novelty", "Helpfulness"]
    df = pd.DataFrame(list(zip(full_list_of_hyp_normal_order, full_list_of_validness_normal_order, full_list_of_novelty_normal_order, full_list_of_helpfulness_normal_order)), columns=columns)
    if expert_file_rand_order == "picked_hypotheses_for_expert_evaluation_rand_order_for_each_group_Junxian_labeled.xlsx":
        output_file_name = "expert_evaluation_normal_order.xlsx"
    elif expert_file_rand_order == "expert_1_2.xlsx":
        output_file_name = "expert_evaluation_1_2_normal_order.xlsx"
    else:
        raise Exception("not supported rand order expert file")
    output_full_addr = os.path.join(root_data_dir, output_file_name)
    assert not os.path.exists(output_full_addr)
    df.to_excel(output_full_addr)
    print(output_file_name)












if __name__ == "__main__":
    # "picked_hypotheses_for_expert_evaluation_rand_order_for_each_group_Junxian_labeled.xlsx"
    # "expert_1_2.xlsx"
    expert_file_rand_order = "expert_1_2.xlsx"
    main(expert_file_rand_order, if_save_output_file=True)
    print("finished")
