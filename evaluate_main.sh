#!/bin/bash
#SBATCH -J gpt4
#SBATCH --partition=DGXq
#SBATCH -w node19
#SBATCH --gres=gpu:1
#SBATCH --output /export/home/zonglin001/Outs/Tomato/gpt4_eval_chatgpt_50bkg_4itr_bkgnoter0_indirect1_onlyindirect2_close0_ban0_baseline0_survey1_bkgInspPasgSwap0_hypSuggestor0_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor.out

# chatgpt / gpt4
python -u evaluate_main.py --if_groundtruth_hypotheses 0 \
        --model_name gpt4 --num_CoLM_feedback_times 4 \
        --start_id 0 --end_id 50 \
        --if_azure_api 1  \
        --output_dir ~/Checkpoints/Tomato/chatgpt_50bkg_4itr_bkgnoter0_indirect1_onlyindirect2_close0_ban0_baseline0_survey1_bkgInspPasgSwap0_hypSuggestor0_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor \
