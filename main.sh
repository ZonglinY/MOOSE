#!/bin/bash
#SBATCH -J Tomato
#SBATCH --partition=DGXq
#SBATCH -w node18
#SBATCH --gres=gpu:1
#SBATCH --output /export/home/zonglin001/Outs/Tomato/chatgpt_50bkg_4itr_bkgnoter0_indirect0_onlyindirect0_close0_ban1_baseline0_survey1_bkgInspPasgSwap0_hypSuggestor0_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor.out


# vicuna / gpt2 / chatgpt / vicuna13 / falcon
python -u main.py --model_name chatgpt \
        --output_dir ~/Checkpoints/Tomato/chatgpt_50bkg_4itr_bkgnoter0_indirect0_onlyindirect0_close0_ban1_baseline0_survey1_bkgInspPasgSwap0_hypSuggestor0_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor \
        --num_background_for_hypotheses 50 --num_CoLM_feedback_times 4 --bkg_corpus_chunk_noter 0 \
        --if_indirect_feedback 0 --if_only_indirect_feedback 0 \
        --if_close_domain 0 --if_ban_selfeval 1 \
        --if_baseline 0 \
        --if_novelty_module_have_access_to_surveys 1 --if_insp_pasg_for_bkg_and_bkg_pasg_included_in_insp 0 \
        --if_hypothesis_suggstor 0
