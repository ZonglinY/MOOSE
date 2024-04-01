import os, argparse, time
from evaluator import Evaluator
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="chatgpt",
                        help="model name: gpt4/chatgpt")
    parser.add_argument("--root_data_dir", type=str, default="./Data/")
    parser.add_argument("--output_dir", type=str, default="~/Checkpoints/Tomato/try")
    parser.add_argument("--num_CoLM_feedback_times", type=int, default=1, help="number of re-generation times given new feedbacks for CoLM")
    parser.add_argument("--start_id", type=int, default=0, help="To evaluate [start_id : end_id] of the Checkpoint file; -1 when not using it")
    parser.add_argument("--end_id", type=int, default=10, help="To evaluate [start_id : end_id] of the Checkpoint file; -1 when not using it")
    parser.add_argument("--if_indirect_feedback", type=int, default=1, help="whether conduct indirect feedback modules such as inspiration_changer and background_changer; also can be called --if_past_feedback")
    parser.add_argument("--if_only_indirect_feedback", type=int, default=0, help="0: tomato-base will perform; 1: Do NOT perform tomato-base because tomato-base has been performed in this checkpoint (prev data will be load up); 2: Do NOT perform tomato-base, but at least tomato-base + past feedback")
    # used for prev_eval_output_dir: ~/Outs/Tomato/gpt4_eval_chatgpt_25bkg_4itr_bkgnoter0_indirect0_onlyindirect0_close0_ban1_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor_5_25.out
    parser.add_argument("--prev_eval_output_dir", type=str, default="", help="In case previous evaluation code has exception, but we don't want to waste money on openai API to re-evaluate the already evaluated hypotheses -- we pick up the previous score from the 'x.out' file")
    parser.add_argument("--if_azure_api", type=int, default=0, help="0: Use openai api from openai website; 1: use openai api from azure")
    parser.add_argument("--if_groundtruth_hypotheses", type=int, default=0, help="0: use ckpt's hypotheses to eval; 1: use groudtruth hypotheses to eval")
    parser.add_argument("--api_key", type=str, default="")
    args = parser.parse_args()

    assert args.model_name == 'gpt4' or args.model_name == 'chatgpt'
    assert args.start_id >= -1 and args.end_id >= -1
    assert args.if_indirect_feedback == 1 or args.if_indirect_feedback == 0
    assert args.if_only_indirect_feedback == 0 or args.if_only_indirect_feedback == 1 or args.if_only_indirect_feedback == 2
    assert args.if_azure_api == 0 or args.if_azure_api == 1
    assert args.if_groundtruth_hypotheses == 0 or args.if_groundtruth_hypotheses == 1
    if args.start_id == -1 or args.end_id == -1:
        assert args.start_id == -1 and args.end_id == -1
    if args.prev_eval_output_dir != "":
        assert args.model_name == 'gpt4'
        assert args.model_name in args.prev_eval_output_dir
        if args.start_id == -1 or args.end_id == -1:
            assert "{}_{}".format(start_id, end_id) in args.prev_eval_output_dir
        if args.if_groundtruth_hypotheses == 0:
            assert args.output_dir.split("Tomato/")[1].strip("/") in args.prev_eval_output_dir


    eval = Evaluator(args)
    eval.read_from_checkpoint()
    eval.llm_init()
    eval.evaluate()





if __name__ == "__main__":
    begin_time = time.time()
    main()
    end_time = time.time()
    duration_minutes = (end_time - begin_time) / 60
    print("finished in {:.2f} minutes".format(duration_minutes))
