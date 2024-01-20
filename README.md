# Large Language Models for Automated Open-domain Scientific Hypotheses Discovery

<p align="center" width="100%">
  <img src="MOOSE and TOMATO.png" alt="MOOSE" style="width: 35%; display: block; margin: auto;"></a>
</p>

This repository is the official implementation of the paper \<Large Language Models for Automated Open-domain Scientific Hypotheses Discovery\>.   
[\[Arxiv version\]](https://arxiv.org/pdf/2309.02726.pdf), 
[\[Most updated paper version\]](https://openreview.net/pdf?id=qPV0U98Mn0j)

In general, with this repository, you can     
(1) generate hypotheses with MOOSE framework,   
(2) evaluate the generated hypotheses by GPT4,    
(3) display results listed in the paper (Table 3-10) from existing checkpoints (where we store the generated hypotheses, and evaluation scores by GPT4 & human expert), and  
(4) display hypotheses and corresponding intermediate generations from existing checkpoints (e.g., research background, research inspirations, future-feedback, hypothesis, and present-feedback).

### Hypotheses Generation with MOOSE  
MOOSE can be run with the python command in ```main.sh```. Option parameters for the python command can be adjusted. Specifically, the function for the options are described below:  
**--num_background_for_hypotheses*: how many number of background to find for hypotheses generation. Each background will be used to generate a set of hypotheses.  
**--num_CoLM_feedback_times*: number of present-feedback iterations.  
**--bkg_corpus_chunk_noter*: start from which background corpus to find background.  
**--if_indirect_feedback*: if run past-feedback (0: no; 1: yes).  
**--if_only_indirect_feedback*: advanced options for past-feedback, by default is 0.
**--if_close_domain*: if adopts groundtruth background and inspirations for hypotheses generation (0: not adopt; 1: adopt).  
**--if_ban_selfeval*: if ban future-feedback 1 (0: run future-feedback 1; 1: not run future-feedback 1).  
**--if_baseline*: baseline options, by default is 0.  
**--if_novelty_module_have_access_to_surveys*: 0: no access; 1: have access.  
**--if_insp_pasg_for_bkg_and_bkg_pasg_included_in_insp*: if randomized corpus (0: no; 1: yes), by default is 0.  
**--if_hypothesis_suggstor*: if run future-feedback 2 (0: not run future-feedback 2; 1: run future-feedback 2).    
**--api_key*: your openai api key to run gpt-3.5-turbo.  

### Hypotheses Evaluation with GPT4  
Hypotheses can be evaluated by GPT4 with the python command in ```evaluation_main.sh```. Specifically, the function for the options are described below:   
**--if_groundtruth_hypotheses*: if evaluate groundtruth hypotheses, by default is 0.  
**--model_name*: model used for evaluation, by default is gpt4.  
**--num_CoLM_feedback_times*: number of present-feedback iterations used for generating the hypotheses.  
**--start_id*: the background corpus id as start to generate hypotheses.  
**--end_id*: the background corpus id as end to generate hypotheses.  
**--if_azure_api*: whether the api is from azure; set to 0 if the api is from openai.   
**--api_key*: your openai api key to run gpt-4 for evaluation.  

### Display results in Table 3 and Table 4  
```python compare_score.py```  

### Display results in Table 5 and Table 6  
```python read_expert_eval.py```  

### Display results in Table 7 and Table 8  
It can be done by adjusting the *method_name1* varibale in ```compare_score.py```.  
Specifically, *method_name1* can be set to *"rand_background_baseline"*, *"rand_background_rand_inspiration_baseline"*, *"rand_background_BM25_inspiration_baseline"*, *"gpt35_background_gpt35_inspiration"*, *"MOOSE_wo_ff1"*, *"MOOSE_wo_ff2"*, *"MOOSE_wo_survey"*, and *"MOOSE_w_random_corpus"*.

### Display results in Table 9  
```python consistency_between_expert_gpt4.py```  
*if_hard_consistency* variable in main() can be adjusted (0 or 1) to check soft or hard consistency score.

### Display results in Table 10 
```python consistency_between_experts.py```  
*if_hard_consistency* variable in main() can be adjusted (0 or 1) to check soft or hard consistency score.  

### Display hypotheses and corresponding intermediate generations (e.g., research background, research inspirations, future-feedback, hypothesis, and present-feedback)  
```python check_intermediate_hypothesis_and_feedback.py --research_background_id 5 --hypothesis_id 0 --hypothesis_refinement_round 0```  
**--research_background_id*: id of research background. The range is [0, 49]  
**--hypothesis_id*: id of the hypotheses generated from a research background (and retrieved inspirations). The typical range is [0, 3 or 4]  
**--hypothesis_refinement_round*: id of hypothesis refinement round (referred as present-feedback). The range is [0, 3]  

### Functions of Other Python files
```data_format_converter.py```: used to transform expert annotated data file to usable data file (mostly format transformation).  
```expert_eval_random_order_to_normal_order.py```: transform expert-evaluated file from random order (to minimize bias from expert) to normal order (to calculate consistency).  
```pick_hyp_for_expert_eval.py```: used to randomly pick hypotheses for expert evaluation.  
```read_from_pdf.py```: extract text contents from social science survey paper.  
