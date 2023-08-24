import os, argparse, time
import torch
from utils import load_data, print_nvidia_smi
from tomato import Tomato
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vicuna",
                        help="model name: gpt2/llama/vicuna/vicuna13/chatgpt/falcon")
    parser.add_argument("--root_data_dir", type=str, default="./Data/")
    parser.add_argument("--survey_data_dir", type=str, default="./Data/Surveys/")
    parser.add_argument("--output_dir", type=str, default="~/Checkpoints/Tomato/try")
    parser.add_argument("--num_CoLM_feedback_times", type=int, default=1, help="number of re-generation times given new feedbacks for CoLM")
    parser.add_argument("--num_background_for_hypotheses", type=int, default=10, help="number of background to find until stop, where background is used to induce hypotheses")
    parser.add_argument("--bkg_corpus_chunk_noter", type=int, default=0, help="start from which corpus_chunk to find background, mainly designed to resume traverse corpus_chunk to find background and therefore to find new hypotheses given new background")
    parser.add_argument("--max_chunks_each_passage", type=int, default=1, help="for each passage in corpus, control the max number of chunks to be counted in self.corpus_chunk; in practice currently it can only influence background chunk and inspiration chunk, but not title chunk and literature chunk")
    parser.add_argument("--if_indirect_feedback", type=int, default=1, help="whether conduct indirect feedback modules such as inspiration_changer and background_changer; also can be called --if_past_feedback")
    parser.add_argument("--if_only_indirect_feedback", type=int, default=0, help="0: tomato-base will perform; 1: Do NOT perform tomato-base because tomato-base has been performed in this checkpoint (prev data will be load up); 2: Do NOT perform tomato-base, but at least tomato-base + past feedback")
    parser.add_argument("--if_close_domain", type=int, default=0, help="if 1, use annotated background and inspirations; else 0, need to find background and inspiration on its own.")
    parser.add_argument("--if_ban_selfeval", type=int, default=0, help="if 0, self.if_self_eval_module will be all false, so no more future feedbacks; also can be called --if_ban_future_feedback")
    parser.add_argument("--if_baseline", type=int, default=0, help="if 0: use gpt-3.5-turbo modules to select background and inspirations; if 1: use randomly pick background and use BM25 to select 6 inspiration sentences from 6 different passage (hypothesis_generator uses the same prompt; no past / present / future feedbacks); if 2: use only randomly picked background to generate hypotheses; if 3: use randomly picked background and randomly picked 6 inspirations.")
    parser.add_argument("--if_novelty_module_have_access_to_surveys", type=int, default=1, help="0: novelty_detector() doesn't have access to surveys; 1: novelty_detector() has access to surveys")
    parser.add_argument("--if_insp_pasg_for_bkg_and_bkg_pasg_included_in_insp", type=int, default=0, help="0: use background passages to select background, and inspiration passages to select inspirations; 1: use inspiration passage to select background, and background passage can also be used as inspiration passage")
    parser.add_argument("--if_hypothesis_suggstor", type=int, default=0, help="0: not use hypothesis_suggstor() in CoLM_controller(); 1: use hypothesis_suggstor() in CoLM_controller()")
    args = parser.parse_args()

    # check hyper-parameters
    assert args.model_name == 'llama' or args.model_name == 'vicuna' or args.model_name == 'vicuna13' or args.model_name == 'gpt2' or args.model_name == 'chatgpt' or args.model_name == 'falcon'
    assert args.if_indirect_feedback == 1 or args.if_indirect_feedback == 0
    assert args.if_only_indirect_feedback == 0 or args.if_only_indirect_feedback == 1 or args.if_only_indirect_feedback == 2
    assert args.if_close_domain == 1 or args.if_close_domain == 0
    assert args.if_ban_selfeval == 0 or args.if_ban_selfeval == 1
    assert args.if_baseline == 0 or args.if_baseline == 1 or args.if_baseline == 2 or args.if_baseline == 3
    assert args.if_novelty_module_have_access_to_surveys == 0 or args.if_novelty_module_have_access_to_surveys == 1
    assert args.if_insp_pasg_for_bkg_and_bkg_pasg_included_in_insp == 0 or args.if_insp_pasg_for_bkg_and_bkg_pasg_included_in_insp == 1
    assert args.if_hypothesis_suggstor == 0 or args.if_hypothesis_suggstor == 1
    # No need to change background and inspirations when given golden background and inspirations
    if args.if_close_domain == 1:
        assert args.if_indirect_feedback == 0
        assert args.if_only_indirect_feedback == 0
        assert args.if_baseline == 0
    if args.if_only_indirect_feedback == 1:
        assert "onlyindirect1" in args.output_dir
    if args.if_baseline == 1 or args.if_baseline == 2 or args.if_baseline == 3:
        # ban past feedback
        assert args.if_indirect_feedback == 0 and args.if_only_indirect_feedback == 0
        # ban current feedback
        assert args.num_CoLM_feedback_times == 0
        # ban future feedback
        assert args.if_ban_selfeval == 1
        # also ban gpt-3.5-turbo selection of background and inspirations in tomato.py, but random background and using BM25 to find inspirations. But prompt for hypothesis_generator() is the same with Tomato framework
        print("Running baseline method: ", args.if_baseline)
    # create output_dir if does not exist
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # check gpu
    n_gpu = torch.cuda.device_count()
    print("n_gpu: ", n_gpu)
    if not args.model_name == 'chatgpt':
        print_nvidia_smi()
        assert n_gpu >= 1

    # load data
    corpus, background_corpus, inspiration_corpus, background_golden, inspiration_golden, existing_literature = load_data(args)
    # load framework
    tmt = Tomato(args, corpus, background_corpus, inspiration_corpus, background_golden, inspiration_golden, existing_literature)
    # begin framework
    tmt.llm_init()
    tmt.corpus_chunking_init()
    tmt.main_controller()







if __name__ == "__main__":
    with torch.no_grad():
        begin_time = time.time()
        main()
        end_time = time.time()
        duration_minutes = (end_time - begin_time) / 60
    print("finished in {:.2f} minutes".format(duration_minutes))
