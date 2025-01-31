#!/bin/bash
#SBATCH -J gpt4
#SBATCH --partition=DGXq
#SBATCH -w node19
#SBATCH --gres=gpu:1
#SBATCH --output /export/home/zonglin001/Outs/Tomato/gpt4_eval_claude_45bkg_4itr_bkgnoter5_indirect1_onlyindirect0_close0_ban0_baseline0_survey1_bkgInspPasgSwap0_hypSuggestor1.out

# chatgpt / gpt4
python -u evaluate_main.py --if_groundtruth_hypotheses 0 \
        --model_name gpt4 --num_CoLM_feedback_times 4 \
        --if_indirect_feedback 1 --if_only_indirect_feedback 0 \
        --start_id 5 --end_id 50 \
        --if_azure_api 1 \
        --output_dir ~/Checkpoints/Tomato/claude_45bkg_4itr_bkgnoter5_indirect1_onlyindirect0_close0_ban0_baseline0_survey1_bkgInspPasgSwap0_hypSuggestor1 \
        --api_key sk-
