import os
import torch
import numpy as np
import pandas as pd
from consistency_between_expert_gpt4 import read_expert_scores, consistency
np.set_printoptions(precision=3)


def main():
    ## Hyper-parameter
    # if_hard_consistency: 0/1
    if_hard_consistency = 1

    # expert evaluation file
    expert_file_0 = 'expert_evaluation_normal_order.xlsx'
    expert_file_1 = 'expert_evaluation_1_2_normal_order.xlsx'

    full_list_of_validness_expert_0, full_list_of_novelty_expert_0, full_list_of_helpfulness_expert_0, len_evaluated_effective_data_0 = read_expert_scores(expert_file_0)
    full_list_of_validness_expert_1, full_list_of_novelty_expert_1, full_list_of_helpfulness_expert_1, len_evaluated_effective_data_1 = read_expert_scores(expert_file_1)
    print("len_evaluated_effective_data_0: {}; len_evaluated_effective_data_1: {}".format(len_evaluated_effective_data_0, len_evaluated_effective_data_1))
    consist_valid = consistency(full_list_of_validness_expert_0, full_list_of_validness_expert_1, if_hard_consistency)
    consist_novel = consistency(full_list_of_novelty_expert_0, full_list_of_novelty_expert_1, if_hard_consistency)
    consist_helpf = consistency(full_list_of_helpfulness_expert_0, full_list_of_helpfulness_expert_1, if_hard_consistency)
    print("\nconsist_valid: {:.3f}; consist_novel: {:.3f}; consist_helpf: {:.3f}".format(consist_valid, consist_novel, consist_helpf))






if __name__ == "__main__":
    main()
    print("finished")
