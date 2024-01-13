import os, argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./Checkpoints/chatgpt_50bkg_4itr_bkgnoter0_indirect1_onlyindirect2_close0_ban1_baseline0_survey1_bkgInspPasgSwap0_hypSuggestor1_hypEqlInsp_manualTitleSuggester_clearSplit_pastfdbkmodified_hypSuggestor", help="output directory")
    parser.add_argument("--research_background_id", type=int, default=5, help="id of the research background being used to generate research hypotheses")
    parser.add_argument("--hypothesis_id", type=int, default=0, help="id of those hypotheses generated from the research background")
    parser.add_argument("--hypothesis_refinement_round", type=int, default=0, help="refinement round of the hypothesis (present-feedback)")
    args = parser.parse_args()

    assert args.hypothesis_refinement_round >= 0 and args.hypothesis_refinement_round <= 3

    print("####### Parameters #######")
    print("checkpoint_dir:", args.checkpoint_dir)
    print("research_background_id:", args.research_background_id)
    print("hypothesis_id:", args.hypothesis_id)
    print("hypothesis_refinement_round:", args.hypothesis_refinement_round)

    data = torch.load(os.path.join(args.checkpoint_dir, "background_inspiration_hypotheses.pt"))
    research_background = data[2][args.research_background_id]
    hypothesis = data[8][research_background][0][args.hypothesis_id][args.hypothesis_refinement_round]
    present_feedback = data[10][research_background][0][hypothesis]
    
    print("\n####### Hypothesis #######\n", hypothesis)
    print("\n####### Present-feedback #######")
    print("\n#### Reality Feedback ####\n{}\n\n#### Novelty Feedback ####\n{}\n\n#### Clarity Feedback ####\n{}\n\n".format(present_feedback[0], present_feedback[1], present_feedback[2]))






if  __name__ == '__main__':
    main()