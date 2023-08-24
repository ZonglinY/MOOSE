import os
import torch
import numpy as np
import pandas as pd


# ckpt_file: a list of ckpt_addr, e.g., [ckpt_addr0, ckpt_addr1, ...]; the order of ckpt_addr should follow the order of background used to generate the ckpt_addr
# direct_or_indirect: 0 or 1
# itrs: a list indicating which iterations of hyp to select, e.g., [0,2,4] or [0]
def random_pick(ckpt_file, ckpt_root_dir, hyp_file, direct_or_indirect, itrs, if_save_file=False):
    assert direct_or_indirect == 0 or direct_or_indirect == 1
    # full_background and full_hypotheses
    full_background = []
    full_hypotheses = {}
    for cur_id in range(len(ckpt_file)):
        cur_file = os.path.join(ckpt_root_dir, ckpt_file[cur_id], hyp_file)
        cur_data = torch.load(cur_file)
        cur_bkg = cur_data[2]
        cur_hyp = cur_data[8]
        full_background += cur_bkg
        full_hypotheses.update(cur_hyp)
    assert len(full_background) == len(full_hypotheses)
    assert len(full_background) == 50
    # picked_hyp
    picked_hyp = []
    picked_hyp_id = []
    for cur_bkg_id, cur_bkg in enumerate(full_background):
        hyp_bkg = full_hypotheses[cur_bkg]
        if not ("indirect1" in ckpt_file[0] and "ban0" in ckpt_file[0]):
            assert len(hyp_bkg) == 1
            cur_hyp_direct_or_indirect = hyp_bkg[direct_or_indirect]
        else:
            # for tomato_pf ckpt, we only pick the hyp that utilize both future feedback and past feedback
            assert len(hyp_bkg) == 2
            cur_hyp_direct_or_indirect = hyp_bkg[direct_or_indirect]
        lucky_id = np.random.randint(0, len(cur_hyp_direct_or_indirect))
        cur_picked_hyp_all_itrs = cur_hyp_direct_or_indirect[lucky_id]
        cur_picked_hyp = [cur_picked_hyp_all_itrs[i] for i in itrs]
        assert len(cur_picked_hyp) == len(itrs)
        picked_hyp_id.append(lucky_id)
        picked_hyp.append(cur_picked_hyp)
        # if cur_bkg_id == 0:
        #     print("\ncur_picked_hyp: ", cur_picked_hyp)
        #     print("len(cur_picked_hyp): ", len(cur_picked_hyp))
    assert len(picked_hyp_id) == len(picked_hyp)
    assert len(picked_hyp_id) == len(full_background)
    # save picked_hyp_id
    if if_save_file:
        picked_hyp_id_file_addr = os.path.join(ckpt_root_dir, ckpt_file[0], "picked_hyp_id_{}.pt".format(direct_or_indirect))
        torch.save(picked_hyp_id, picked_hyp_id_file_addr)
        print("Saved picked_hyp_id.pt")
    return picked_hyp


# list_picked_hyp: [picked_hyp0, picked_hyp1, ...]
def unify_picked_hyp_to_xlsx(list_picked_hyp, ckpt_expert_evaluation_dir, ckpt_expert_evaluation_file, if_save_file=False):
    len_background = 50
    rand_order = []
    full_list_of_hyp = []
    # since we are comparing 4 ckpts
    assert len(list_picked_hyp) == 4
    for cur_bkg_id in range(len_background):
        # cur_bkg_ordered_hyp (contain list of list)
        cur_bkg_ordered_hyp = []
        for cur_picked_hyp in list_picked_hyp:
            assert len(cur_picked_hyp) == len_background
            cur_bkg_ordered_hyp.append(cur_picked_hyp[cur_bkg_id])
        # cur_bkg_rand_ordered_hyp (contain list of list)
        cur_order = np.arange(len(cur_bkg_ordered_hyp))
        np.random.shuffle(cur_order)
        cur_bkg_rand_ordered_hyp = [cur_bkg_ordered_hyp[i] for i in cur_order]
        rand_order.append(cur_order)
        # cur_bkg_list_of_hyp (not contain list of list)
        cur_bkg_list_of_hyp = []
        for cur_list_of_hyp in cur_bkg_rand_ordered_hyp:
            assert isinstance(cur_list_of_hyp, list)
            for cur_hyp in cur_list_of_hyp:
                cur_bkg_list_of_hyp.append(cur_hyp)
        # since we know we should have 8 hypotheses per group
        assert len(cur_bkg_list_of_hyp) == 8
        full_list_of_hyp += cur_bkg_list_of_hyp
    assert len(rand_order) == len_background
    # since we know we should have 8 hypotheses per group
    assert len(full_list_of_hyp) == len_background * 8
    if if_save_file:
        # save rand_order
        torch.save(rand_order, os.path.join(ckpt_expert_evaluation_dir, "rand_order_for_each_group.pt"))
        # save full_list_of_hyp
        full_list_of_validness = ["" for i in range(len(full_list_of_hyp))]
        full_list_of_novelty = ["" for i in range(len(full_list_of_hyp))]
        full_list_of_helpfulness = ["" for i in range(len(full_list_of_hyp))]
        columns = ["Hypothesis", "Validness", "Novelty", "Helpfulness"]
        df = pd.DataFrame(list(zip(full_list_of_hyp, full_list_of_validness, full_list_of_novelty, full_list_of_helpfulness)), columns=columns)
        df.to_excel(os.path.join(ckpt_expert_evaluation_dir, 'picked_hypotheses_for_expert_evaluation_rand_order_for_each_group.xlsx'))
        print("Saved rand_order_for_each_group.pt and picked_hypotheses_for_expert_evaluation_rand_order_for_each_group.xlsx")


def main():
    ## Warning: don't set if_save_file to True after picking hyp
    if_save_file = False
    assert if_save_file == False
    ## baseline ckpt
    ckpt_baseline2_0_50 = "chatgpt_50bkg_0itr_bkgnoter0_indirect0_onlyindirect0_close0_ban1_baseline2_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ## Tomato-base ckpts
    ckpt_tomato_base_0_25 = "chatgpt_25bkg_4itr_bkgnoter0_indirect0_onlyindirect0_close0_ban1_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ckpt_tomato_base_25_50 = "chatgpt_25bkg_4itr_bkgnoter25_indirect0_onlyindirect0_close0_ban1_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ## Tomato-past-future ckpts
    ckpt_tomato_pf_0_25 = "chatgpt_25bkg_4itr_bkgnoter0_indirect1_onlyindirect0_close0_ban0_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ckpt_tomato_pf_25_50 = "chatgpt_25bkg_4itr_bkgnoter25_indirect1_onlyindirect0_close0_ban0_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor"
    ## ckpt_dir and hyp_file_name
    # "/export/home/zonglin001/Checkpoints/Tomato/"
    ckpt_root_dir = "./Checkpoints/"
    hyp_file = "background_inspiration_hypotheses.pt"
    ckpt_expert_evaluation_dir = "./Checkpoints/expert_evaluation/"
    ckpt_expert_evaluation_file = "expert_evaluation.xlsx"
    ## random_pick
    picked_hyp_baseline2 = random_pick([ckpt_baseline2_0_50], ckpt_root_dir, hyp_file, direct_or_indirect=0, itrs=[0], if_save_file=if_save_file)
    picked_hyp_tomato_base = random_pick([ckpt_tomato_base_0_25, ckpt_tomato_base_25_50], ckpt_root_dir, hyp_file, direct_or_indirect=0, itrs=[0,2,4], if_save_file=if_save_file)
    picked_hyp_tomato_pf_onlyf = random_pick([ckpt_tomato_pf_0_25, ckpt_tomato_pf_25_50], ckpt_root_dir, hyp_file, direct_or_indirect=0, itrs=[4], if_save_file=if_save_file)
    picked_hyp_tomato_pf_bothpf = random_pick([ckpt_tomato_pf_0_25, ckpt_tomato_pf_25_50], ckpt_root_dir, hyp_file, direct_or_indirect=1, itrs=[0,2,4], if_save_file=if_save_file)
    ## unify_picked_hyp_to_xlsx
    unify_picked_hyp_to_xlsx([picked_hyp_baseline2, picked_hyp_tomato_base, picked_hyp_tomato_pf_onlyf, picked_hyp_tomato_pf_bothpf], ckpt_expert_evaluation_dir, ckpt_expert_evaluation_file, if_save_file=if_save_file)








if __name__ == "__main__":
    main()
    print("finished")
