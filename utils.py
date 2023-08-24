import pandas as pd
import numpy as np
import os, math, re
import requests
import subprocess
import torch
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from read_from_pdf import PdfConverter


# OUTPUT
# corpus: [['title0', 'passage0'], ...]; used as input to develop hypothesis
# background_corpus / inspiration_corpus: [['title0', 'passage0'], ...]; background_corpus + inspiration_corpus = corpus
# existing_literature: [['title0', 'existing_literature0'], ...]; used to check the novelty of developed hypothesis
# background_golden / inspiration_golden: [[bkg0, bkg1](for line 0), ...]
def load_data(args):
    ## corpus, background_corpus, inspiration_corpus
    corpus, background_corpus, inspiration_corpus = [], [], []
    # background_golden / inspiration_golden: [[bkg0, bkg1](for line 0), ...]
    background_golden, inspiration_golden = [], []
    raw_corpus = pd.read_excel(os.path.join(args.root_data_dir, 'business_research.xlsx'))
    for cur_data_id in range(len(raw_corpus)):
        cur_data_bkg_golden, cur_data_insp_golden = [], []
        for cur_ctnt in ['background_1', 'background_2', 'inspiration_1', 'inspiration_2', 'inspiration_3']:
            cur_data = {}
            # cur_data_title and cur_data_passage
            # some titles are all capitalized
            cur_data_title = raw_corpus['{}_title'.format(cur_ctnt)][cur_data_id]
            cur_data_passage = raw_corpus['{}_passage'.format(cur_ctnt)][cur_data_id]
            cur_data_golden = raw_corpus['{}_golden'.format(cur_ctnt)][cur_data_id]
            if isinstance(cur_data_title, str):
                if cur_data_title.strip() != "":
                    assert isinstance(cur_data_passage, str) and cur_data_passage.strip() != ""
                    assert isinstance(cur_data_golden, str) and cur_data_golden.strip() != ""
                    cur_data_title = cur_data_title.strip()
                    cur_data_passage = cur_data_passage.strip()
                    cur_data_golden = cur_data_golden.strip()
                    cur_data = [cur_data_title.capitalize(), cur_data_passage]
                    corpus.append(cur_data)
                    if 'background' in cur_ctnt:
                        background_corpus.append(cur_data)
                        cur_data_bkg_golden.append(cur_data_golden)
                    elif 'inspiration' in cur_ctnt:
                        inspiration_corpus.append(cur_data)
                        cur_data_insp_golden.append(cur_data_golden)
                    else:
                        raise Exception("cur_ctnt is neither background nor inspiration: ", cur_ctnt)
        background_golden.append(cur_data_bkg_golden)
        inspiration_golden.append(cur_data_insp_golden)
    print("len(corpus): ", len(corpus))
    ## existing_literature: [['title0', 'existing_literature0'], ...]
    existing_literature = []
    survey_names = os.listdir(args.survey_data_dir)
    assert len(survey_names) >= 5
    for cur_survey_id in range(len(survey_names)):
        cur_literature, cur_literature_title = "", ""
        cur_survey_full_address = os.path.join(args.survey_data_dir, survey_names[cur_survey_id])
        if not "pdf" in cur_survey_full_address:
            print("Warning: file in survey_data_dir not ends in pdf:", cur_survey_full_address)
            continue
        pdfConverter = PdfConverter(file_path=cur_survey_full_address)
        cur_literature_title, cur_literature = pdfConverter.convert_pdf_to_txt()
        existing_literature.append([cur_literature_title, cur_literature])
    print("len(corpus): {}; len(background_corpus): {}; len(inspiration_corpus): {}".format(len(corpus), len(background_corpus), len(inspiration_corpus)))
    return corpus, background_corpus, inspiration_corpus, background_golden, inspiration_golden, existing_literature


# Function: chunking passage
# Input:
#   corpus:
#       [['title0', 'passage0'], ...] if if_title_chunk == False
#       ['title0', 'title1', ...] if if_title_chunk == True
#   model_input_len: an integer
#   if_title_chunk: True or False, when true, the input corpus will be in ['title0', 'title1', ...] format
#   if_with_title: True or False, determine whether all chunks are with their corresponding title;
#                   if if_title_chunk == True, if_with_title should be False
# Output:
#   passage_chunk: ['title0, passage0_chunk0', 'title0, passage0_chunk1', ...] (if_with_title: True)
#   passage_chunk: ['passage0_chunk0', 'passage0_chunk1', ...] (if_with_title: False)
def chunk_passage(corpus, model_input_len, max_chunks_each_passage=30, if_title_chunk=False, if_with_title=False, word_limit_weight=1/4):
    assert if_title_chunk == True or if_title_chunk == False
    assert if_with_title == True or if_with_title == False
    if if_title_chunk == True:
        assert if_with_title == False
    # get all_title and all_passage
    if not if_title_chunk:
        # all_title: ['title0', 'title1', ...]
        all_title = [corpus[i][0] for i in range(len(corpus))]
        # all_passage: ['passage0', 'passage1', ...]
        all_passage = [corpus[i][1] for i in range(len(corpus))]
        assert len(all_title) == len(corpus)
        assert len(all_title) == len(all_passage)
    else:
        # all_title: ['']
        all_title = ['']
        # all_passage: ['title0. title1. title2. ...']
        all_passage = ["'"+corpus[i]+"'." for i in range(len(corpus))]
        all_passage = [' '.join(all_passage)]
        assert len(all_title) == len(all_passage)
    ## passage_chunk
    passage_chunk = []
    # use number of words to mimic satisfying the input limit on number of sub-words
    # leave 5/8 for generation; it seems llm tend to perform worse given longer input text
    word_limit = int(model_input_len * word_limit_weight)
    # for each passage to process
    for cur_id_psg in range(len(all_passage)):
        cur_psg = all_passage[cur_id_psg]
        cur_psg_word_cnt = word_count_approx(cur_psg)
        # the list of sent in cur_psg
        cur_psg_sent_list = re.split('\.|\•|\\n', cur_psg)
        # get rid of empty sentence caused by split
        cur_psg_sent_list = [cur_psg_sent_list[i].strip() for i in range(len(cur_psg_sent_list)) if cur_psg_sent_list[i].strip() != ""]
        len_cur_psg_sent_list = len(cur_psg_sent_list)
        # the list of the word count for each sent in cur_psg
        cur_psg_sent_wordcnt_list = [word_count_approx(cur_psg_sent_list[i]) for i in range(len_cur_psg_sent_list)]
        # note which sent in cur_psg has been counted to chunks, this id points to the next sent that has not been counted into any chunks
        cur_psg_cur_sent_id = 0
        # cnt_chunks_cur_psg should be <= max_chunks_each_passage for each passage
        cnt_chunks_cur_psg = 0
        while cur_psg_cur_sent_id < len_cur_psg_sent_list:
            cur_chunk = ''
            cur_chunk_word_cnt = 0
            while (cur_psg_cur_sent_id + 1 <= len_cur_psg_sent_list) and \
                    (cur_chunk_word_cnt + cur_psg_sent_wordcnt_list[cur_psg_cur_sent_id] < word_limit):
                cur_chunk += cur_psg_sent_list[cur_psg_cur_sent_id] + '.'
                cur_chunk_word_cnt += cur_psg_sent_wordcnt_list[cur_psg_cur_sent_id]
                cur_psg_cur_sent_id += 1
            if cur_chunk == '':
                # print("passage_chunk: ", passage_chunk)
                print("cur_psg_sent_list[cur_psg_cur_sent_id]: ", cur_psg_sent_list[cur_psg_cur_sent_id])
                print("cur_psg_cur_sent_id: ", cur_psg_cur_sent_id)
                print("len_cur_psg_sent_list: ", len_cur_psg_sent_list)
                print("cur_chunk_word_cnt: ", cur_chunk_word_cnt)
                print("cur_psg_sent_wordcnt_list[cur_psg_cur_sent_id]: ", cur_psg_sent_wordcnt_list[cur_psg_cur_sent_id])
                print("word_limit: ", word_limit)
                raise Exception
            if if_with_title:
                # passage_chunk.append('Title of this passage is ' + all_title[cur_id_psg] + '.\n Here is the passage chunk: ' + cur_chunk)
                # it seems less prompt is more concise and less ambiguous when combining with other prompts
                # here '.\n' is used in inspiration_passage_retriever() to seperate title
                passage_chunk.append(all_title[cur_id_psg] + '.\n' + cur_chunk)
            else:
                passage_chunk.append(cur_chunk)
            cnt_chunks_cur_psg += 1
            if cnt_chunks_cur_psg == max_chunks_each_passage:
                break
        # just an ancillary check
        assert cur_psg_cur_sent_id == len_cur_psg_sent_list or cnt_chunks_cur_psg == max_chunks_each_passage
    return passage_chunk


# passage: 'text'
def word_count_approx(passage):
    word_count = len(passage.split(' '))
    return word_count


# cur_sent_matched: ['sent0', ... (best_k)], sents are from sent_list that match best with cur_gene
def find_simi_score_using_BM25(cur_gene, sent_list, best_k=1):
    assert len(sent_list) >= best_k
    tokenized_all_sent = [sent.split(" ") for sent in sent_list]
    bm25 = BM25Okapi(tokenized_all_sent)
    tokenized_cur_gene = cur_gene.split(" ")
    simi_scores = bm25.get_scores(tokenized_cur_gene)
    assert len(simi_scores) == len(sent_list)
    cur_sent_index = np.argsort(simi_scores)
    # cur_sent_index: [low similarity, ->, high similarity]
    cur_sent_matched = []
    for i in range(len(sent_list)):
        cur_selected_sent = sent_list[cur_sent_index[-i]]
        if len(cur_selected_sent) > 90:
            cur_sent_matched.append(cur_selected_sent)
        if len(cur_sent_matched) == best_k:
            break
    return cur_sent_matched


# Function: find existing titles that are in all_title from title_generation
# Input:
#   title_generation: 'text' -- might contain multiple title, by default (according to our trials with demo) we assume multiple titles are split by "\n"
#   title_eval_generation: 'text' -- might contain multiple eval for title; by default we assume multiple titles are split by "\n"
#   all_title: ['title0', 'title1', ...]
#   keyword_key_generation: 'text'; Examples: 'Background:', 'Title:'
# Output:
#   title_collection: ['existing title0', 'existing title1', ...]
#   title_collection_eval: ['existing title0 eval', 'existing title0 eval', ...] if if_confident_enough_to_include_eval == True and if_with_eval == True else ['', '', ...]
def match_existing_title_given_title_generation(title_generation_split, title_eval_generation_split, if_confident_enough_to_include_eval, if_with_eval, all_title):
    # title_collection, title_collection_eval
    title_collection, title_collection_eval = [], []
    for cur_gene_id, cur_gene in enumerate(title_generation_split):
        tokenized_all_title = [doc.split(" ") for doc in all_title]
        bm25 = BM25Okapi(tokenized_all_title)
        tokenized_cur_gene = cur_gene.split(" ")
        simi_scores = bm25.get_scores(tokenized_cur_gene)
        assert len(simi_scores) == len(all_title)
        # 1.5 is based on heuristics
        if max(simi_scores) > 2.5:
            cur_ttl_index = np.argmax(simi_scores)
            title_collection.append(all_title[cur_ttl_index])
            if if_confident_enough_to_include_eval and if_with_eval:
                title_collection_eval.append(title_eval_generation_split[cur_gene_id])
            else:
                title_collection_eval.append('')
    assert len(title_collection) == len(title_collection_eval)
    return title_collection, title_collection_eval


# Function: given titles,
#                        (1): avoid repetiton in titles, process title_eval the same way to ensure title and title_eval are matched
#                        (2): retrieve corresponding chunks in corpus_chunk
# INPUT:
#   title_collection: [title0, title1, ...]
#   title_collection_eval: ['existing title0 eval', 'existing title1 eval', ...], some 'existing title eval' could be '' if no enough confidence
#   corpus_chunk: ['title0, passage0_chunk0', 'title0, passage0_chunk1', ...]
# OUTPUT:
#   passage_collections_chunks: ['title0, passage0_chunk0', 'title0, passage0_chunk1', ...], where the length of chunks follows self.model_input_len
#   passage_collections_chunks_title_eval: ['corresponding title eval', ...], some 'eval' could be '' if no enough confidence
def find_passages_with_titles(title_collection, title_collection_eval, corpus_chunk):
    # (1): avoid repetiton in titles
    concised_title_collection, concised_title_collection_eval = [], []
    for cur_id, cur_ttl in enumerate(title_collection):
        if cur_ttl not in concised_title_collection:
            concised_title_collection.append(cur_ttl)
            concised_title_collection_eval.append(title_collection_eval[cur_id])
    # (2): retrieve corresponding chunks in corpus_chunk
    passage_collections_chunks = []
    passage_collections_chunks_title_eval = []
    for cur_ttl_id, cur_ttl in enumerate(concised_title_collection):
        if_found = False
        for cur_chk in corpus_chunk:
            if cur_ttl in cur_chk:
                passage_collections_chunks.append(cur_chk)
                passage_collections_chunks_title_eval.append(concised_title_collection_eval[cur_ttl_id])
                if_found = True
                break
        assert if_found == True
    print("len(concised_title_collection): ", len(concised_title_collection))
    assert len(passage_collections_chunks) == len(concised_title_collection)
    assert len(passage_collections_chunks) == len(passage_collections_chunks_title_eval)
    assert len(passage_collections_chunks) > 0
    return passage_collections_chunks, passage_collections_chunks_title_eval


# Function: transfer the format of feedbacks
# Input:
#   list_of_CoLM_feedbacks: [cur_consistency_feedback, cur_reality_feedback, cur_novelty_feedback, cur_specification_feedback]
#       cur_*_feedback: {'hypothesis0': *_feedback0, 'hypothesis1': *_feedback1, ...}
# Output:
#   cur_feedbacks_hypotheses: {'hypothesis0': ['consistency_feedback0', 'reality_feedback', 'novelty_feedback', 'specification_feedback'], ...}
def unify_feedbacks_to_format(list_of_CoLM_feedbacks):
    # cur_consistency_feedback, cur_reality_feedback, cur_novelty_feedback, cur_specification_feedback = list_of_CoLM_feedbacks
    cur_reality_feedback, cur_novelty_feedback, cur_specification_feedback = list_of_CoLM_feedbacks
    # assert cur_consistency_feedback.keys() == cur_reality_feedback.keys()
    assert cur_reality_feedback.keys() == cur_novelty_feedback.keys()
    assert cur_reality_feedback.keys() == cur_specification_feedback.keys()
    cur_feedbacks_hypotheses = {}
    for key in cur_reality_feedback.keys():
        # cur_feedbacks_hypotheses[key] = [cur_consistency_feedback[key], cur_reality_feedback[key], cur_novelty_feedback[key], cur_specification_feedback[key]]
        cur_feedbacks_hypotheses[key] = [cur_reality_feedback[key], cur_novelty_feedback[key], cur_specification_feedback[key]]
    return cur_feedbacks_hypotheses


# Function: pick the most interested key_generation from the output of self.llm_generation (when model_name == 'chatgpt', directly return cur_generation)
# Input:
#   cur_generation|post_prompt: 'text'
#   keyword_key_generation|keyword_key_generation_eval: 'text:', used for split key_generation and its eval from output
#   if_with_eval: True or False, whether self_eval or not; if False, cur_keygeneration_self_eval is ''
# Output:
#   cur_keygeneration|cur_keygeneration_self_eval: 'text'
# Special treatment for 'hypothesis_generator_refine' and 'indiscriminate_confirmation_handler' and 'hypothesis_suggstor'
def pick_from_generation(model_name, cur_generation, post_prompt, if_with_eval=False, keyword_key_generation=None, keyword_key_generation_eval=None, module_name=None):
    ## split output
    if model_name != 'chatgpt':
        cur_generation = cur_generation.split(post_prompt)
        if len(cur_generation) != 2:
            print("Warning: len(cur_generation.split(post_prompt)) > 2")
            print("len(cur_generation): {}; post_prompt: {}".format(len(cur_generation), post_prompt))
            print("cur_generation: {}".format(cur_generation))
        key_generation_raw = cur_generation[1]
    else:
        key_generation_raw = cur_generation
    # we don't want indiscriminate_confirmation_handler() to provide suggestions since our main focus is novelty aspect
    if module_name == 'indiscriminate_confirmation_handler' or module_name == 'hypothesis_suggstor':
        return [key_generation_raw], [""], False
    ## key_generation, key_generation_eval
    key_generation_split = key_generation_raw.split('\n')
    key_generation, key_generation_eval = [], []
    # mode_generation_to_split: 0: noise sentences; 1: key_generation; 2: key_generation_eval
    mode_generation_to_split = 0
    # if_append: 0: if match with keyword_key_generation or keyword_key_generation_eval
    if_append = 0
    # if_unfinished_matched_keywords: 0: possible to change mode_generation_to_split to 0; 1: can't set mode_generation_to_split to 0 (just past only matched keyword text block, e.g. "Reasoning process: \n\n", no content for the keyword found yet)
    if_unfinished_matched_keywords = 0
    for cur_gene in key_generation_split:
        if cur_gene == "":
            # newly added, since empirically when there's '\n\n' and it not just past only keyword text block, the next paragraph is not relevant to the previous paragraph
            if if_unfinished_matched_keywords == 0:
                mode_generation_to_split = 0
            continue
        # mode_generation_to_split, if_append
        if len(re.findall(r'{}[\s]*[0-9]*:'.format(keyword_key_generation.strip(":").strip()), cur_gene)) >= 1:
            if_append = 1
            mode_generation_to_split = 1
            cur_gene = re.sub(r'{}[\s]*[0-9]*:'.format(keyword_key_generation.strip(":").strip()), "", cur_gene)
        elif len(re.findall(r'{}[\s]*[0-9]*:'.format(keyword_key_generation_eval.strip(":").strip()), cur_gene)) >= 1:
            if_append = 1
            mode_generation_to_split = 2
            cur_gene = re.sub(r'{}[\s]*[0-9]*:'.format(keyword_key_generation_eval.strip(":").strip()), "", cur_gene)
        # check cur_gene
        cur_gene = cur_gene.strip()
        if cur_gene == "":
            if_unfinished_matched_keywords += 1
            continue
        if cur_gene.strip()[-1] == ":":
            if_unfinished_matched_keywords += 2
        # key_generation, key_generation_eval
        if mode_generation_to_split == 1:
            if if_append == 1:
                key_generation.append(cur_gene)
            else:
                key_generation[-1] += cur_gene
        elif mode_generation_to_split == 2:
            if if_append == 1:
                key_generation_eval.append(cur_gene)
            else:
                key_generation_eval[-1] += cur_gene
        else:
            print("Warning: noise sentence exist before key_generation: ", cur_gene)
        if_append = 0
        # not blocked by "continue", some text should have been matched, so not unfinished
        if if_unfinished_matched_keywords > 0:
            if_unfinished_matched_keywords -= 1
    ## if_matched
    if len(key_generation) == len(key_generation_eval) and len(key_generation) >= 1:
        if_matched = True
    # sometimes in hypothesis_generator_refine when not using future feedbacks, the generation does not fit the format (especially not matched number of key_generation and key_generation_eval). However we only need to use the first generation in key_generation and key_generation_eval
    elif module_name == "hypothesis_generator_refine" and len(key_generation) >= 1 and len(key_generation_eval) >= 1:
        key_generation = key_generation[:1]
        key_generation_eval = key_generation_eval[:1]
        if_matched = True
        print("Warning: unmatched key_generation and key_generation_eval in hypothesis_generator_refine module")
    else:
        if_matched = False
    ## return
    if if_matched == True:
        assert len(key_generation) == len(key_generation_eval)
        if if_with_eval == True:
            return key_generation, key_generation_eval, if_matched
        else:
            return key_generation, ["" for i in range(len(key_generation_eval))], if_matched
    else:
        print("Warning: if_matched is False in {} module".format(module_name))
        return [key_generation_raw], [""], if_matched





# OUTPUT:
#   cur_background: 'background'
#   cur_background_eval: 'background_eval'
#   cur_title_matched: ['existing title0', 'existing title1', ...]
#   cur_title_matched_self_eval: ['existing title0 eval', 'existing title1 eval', ...]
#   cur_inspirations: ['inspiration0', 'inspiration1', ...]
#   cur_inspirations_eval: ['inspiration0_eval', 'inspiration1_eval', ...]
#   cur_hypotheses: 'hypothesis-latest'
#   cur_hypotheses_reasoning_process: 'hypothesis_reasoning-lastest'
#   cur_feedbacks_hypotheses: {'hypothesis-latest': ['consistent feedback', 'reality feedback', 'novelty feedback']}
def load_variables_for_debug(self, output_dir, cur_id_background):
    # it can only work independently when initial bkg_corpus_chunk_noter == 0; otherwise it can't find where to start
    if cur_id_background == 0:
        # bkg_corpus_chunk_noter equals 0 means we start from background[self.bkg_corpus_chunk_noter + cur_id_background]
        assert self.bkg_corpus_chunk_noter == 0
    ## Load data
    # data = torch.load(os.path.join(output_dir, "background_inspiration_hypotheses.pt"))
    # must change name to 'prev_' to save new variables
    data = torch.load(os.path.join(output_dir, "prev_background_inspiration_hypotheses.pt"))
    model_name, bkg_corpus_chunk_noter, background, background_self_eval, selected_titles, selected_titles_self_eval, inspiration, inspiration_self_eval, hypothesis, hypothesis_reasoning_process, hypothesis_CoLM_internal_feedback, hypothesis_CoLM_external_feedback, max_chunks_each_passage, corpus_chunk, prev_args, suggestion, if_baseline = data
    # len(background) in prev_background_inspiration_hypotheses.pt must equal to current length of experiments
    assert self.num_background_for_hypotheses == len(background)

    ## Save variables in current step
    # background_finder_wrapper
    self.bkg_corpus_chunk_noter = cur_id_background+1
    self.background.append(background[cur_id_background])
    self.background_self_eval.append(background_self_eval[cur_id_background])
    # inspiration_title_retriever
    cur_background = background[cur_id_background]
    if cur_background not in self.selected_titles:
        self.selected_titles[cur_background] = [selected_titles[cur_background][0]]
        assert cur_background not in self.selected_titles_self_eval
        self.selected_titles_self_eval[cur_background] = [selected_titles_self_eval[cur_background][0]]
    else:
        self.selected_titles[cur_background].append(selected_titles[cur_background][0])
        assert cur_background in self.selected_titles_self_eval
        self.selected_titles_self_eval[cur_background].append(selected_titles_self_eval[cur_background][0])
    # inspiration_passage_retriever
    if cur_background not in self.inspiration:
        self.inspiration[cur_background] = [inspiration[cur_background][0]]
        assert cur_background not in self.inspiration_self_eval
        self.inspiration_self_eval[cur_background] = [inspiration_self_eval[cur_background][0]]
    else:
        self.inspiration[cur_background].append(inspiration[cur_background][0])
        assert cur_background in self.inspiration_self_eval
        self.inspiration_self_eval[cur_background].append(inspiration_self_eval[cur_background][0])
    # CoLM_controller
    if cur_background in self.hypothesis:
        self.hypothesis[cur_background].append(hypothesis[cur_background][0])
        assert cur_background in self.hypothesis_reasoning_process
        self.hypothesis_reasoning_process[cur_background].append(hypothesis_reasoning_process[cur_background][0])
        assert cur_background in self.hypothesis_CoLM_internal_feedback
        self.hypothesis_CoLM_internal_feedback[cur_background].append(hypothesis_CoLM_internal_feedback[cur_background][0])
    else:
        self.hypothesis[cur_background] = [hypothesis[cur_background][0]]
        assert cur_background not in self.hypothesis_reasoning_process
        self.hypothesis_reasoning_process[cur_background] = [hypothesis_reasoning_process[cur_background][0]]
        assert cur_background not in self.hypothesis_CoLM_internal_feedback
        self.hypothesis_CoLM_internal_feedback[cur_background] = [hypothesis_CoLM_internal_feedback[cur_background][0]]

    ## return running variables for further usage
    return background[cur_id_background], background_self_eval[cur_id_background], selected_titles[background[cur_id_background]][0], selected_titles_self_eval[background[cur_id_background]][0], inspiration[background[cur_id_background]][0], inspiration_self_eval[background[cur_id_background]][0], hypothesis[background[cur_id_background]][0], hypothesis_reasoning_process[background[cur_id_background]][0], hypothesis_CoLM_internal_feedback[background[cur_id_background]][0]



# prompt_mode: some modules need more than one set of prompts (e.g., inspiration_passage_retriever)
def prompts_for_tomato_modules(model_name, module_name, if_with_eval=False, prompt_mode=0, if_baseline=0):
    assert module_name == 'background_finder' or module_name == 'inspiration_title_retriever' or module_name == 'inspiration_passage_retriever' or \
    module_name == 'background_evaluator' or module_name == 'hypothesis_suggstor' or \
    'hypothesis_generator' in module_name or module_name == 'deductive_consistency_evaluator' or \
    module_name == 'indiscriminate_confirmation_handler' or module_name == 'generalization_checker' or \
    module_name == 'novelty_detector' or module_name == 'specification_detector' or module_name == 'background_changer' or module_name == 'inspiration_title_changer' or module_name == 'inspiration_title_suggestor'
    assert if_with_eval == True or if_with_eval == False
    assert if_baseline == 0 or if_baseline == 1 or if_baseline == 2 or if_baseline == 3

    if module_name == 'background_finder':
        # We use this if_with_eval==True post_prompt_format for whichever if_with_eval to split cur_gene and cur_gene_feedback. When if_with_eval == False, tomato.py code should take charge of not using cur_gene_feedback
        post_prompt_format = ", and also provide an evaluation of the selected background in terms of what are possible business research directions given the background (response format: 'Background: \nEvaluation: \n...')."
        pre_prompt = "In the provided passage, likely from a business-related report, try to collect the best paragraph (or sentence) in the reports that could serve as suitable academic background for business research. The chosen academic background in business should encompass research topics that can be further developed into hypotheses for business research. The passage is: \n"
        mid_prompt = ""
        post_prompt = "\nPlease give a response to the initial question of exactly extracting the best business academic background paragraph (or sentence) from the given passage" + post_prompt_format
    elif module_name == 'inspiration_title_retriever':
        # We use this if_with_eval==True post_prompt_format for whichever if_with_eval to split cur_gene and cur_gene_feedback. When if_with_eval == False, tomato.py code should take charge of not using cur_gene_feedback
        post_prompt_format = ", and also evaluate the selected titles in terms of how it could potentially help business research hypothesis developing (response format: 'Title: \nEvaluation: \nTitle: \nEvaluation: \n...')."
        # "usually a hypothesis is more novel if its inspirations are less directly related to the given background" not used, since it should be reflected as a past feedback
        pre_prompt = "Given an academic background in business research and titles of business-related reports, which titles (and their corresponding business reports) could contain research inspirations which combined with the background could lead to non-trivial hypotheses in business research?\n The academic background is "
        mid_prompt = "\nThe title collections are:\n"
        post_prompt = "\nPlease give a response to the initial question of extracting three titles that most probably contain suitable research inspirations given the business research background" + post_prompt_format
    elif module_name == 'inspiration_title_suggestor':
        # We use this if_with_eval==True post_prompt_format for whichever if_with_eval to split cur_gene and cur_gene_feedback. When if_with_eval == False, tomato.py code should take charge of not using cur_gene_feedback
        post_prompt_format = ", and give some suggestions on future report selection to help generate better business research hypotheses (response format: 'Problem: \nSuggestion: \nProblem: \nSuggestion: \n...')."
        if if_with_eval:
            pre_prompt = "Given an academic background in business research, previously selected titles of business-related reports, previously generated business research hypothesis using the academic backgroud and some inspirations from the selected reports (according to selected titles for reports), and evaluation of previously generated hypothesis, try to understand potential problems of previously generated business research hypothesis that might be caused by improper selection of business reports, identify potential problems of report selection, and give some suggestions on future report selection to generate better hypotheses.\n The academic background is "
        else:
            # not mentioning provide suggestions, but only problems
            pre_prompt = "Given an academic background in business research, previously selected titles of business-related reports, previously generated business research hypothesis using the academic backgroud and some inspirations from the selected reports (according to selected titles for reports), and evaluation of previously generated hypothesis, try to understand potential problems of previously generated business research hypothesis that might be caused by improper selection of business reports, and identify potential problems of report selection. \nThe academic background is "
        mid_prompt = ["\nThe previously selected titles are: \n", "\nThe prevously generated hypotheses and their evaluation are: \n"]
        post_prompt = "\nPlease give a response to the initial question of identifying and elaborating problems of the previously selected report titles that might cause negative effect on generating the given specific hypothesis" + post_prompt_format
    # hand-coded suggestions into the prompt of 'inspiration_title_changer' ('also remember the advice that...')
    elif module_name == 'inspiration_title_changer':
        # We use this if_with_eval==True post_prompt_format for whichever if_with_eval to split cur_gene and cur_gene_feedback. When if_with_eval == False, tomato.py code should take charge of not using cur_gene_feedback
        post_prompt_format = ", and also evaluate the selected titles in terms of how it could potentially help business research hypothesis developing (response format: 'Title: \nEvaluation: \nTitle: \nEvaluation: \n...')."
        pre_prompt = "Given an academic background in business research and titles of business-related reports, which titles (and their corresponding business reports) could contain research inspirations which combined with the background could lead to non-trivial hypotheses in business research (usually a hypothesis is more novel if its inspirations are less directly related to the given background)? \nSome feedbacks of the previous selected titles for hypotheses generation are also given, maybe also leverage the feedbacks when selecting titles. \nThe academic background is "
        mid_prompt = ["\nFeedbacks of previous selected titles:\n", "\nThe title collections are:\n"]
        post_prompt = "\nPlease give a response to the initial question of extracting three titles that most probably contain suitable research inspirations given the business research background" + post_prompt_format
    # Q: would the if_with_eval prompt here evaluating from too many aspects that it could be overwhelming to provide evaluation?
    elif module_name == 'inspiration_passage_retriever':
        # We use this if_with_eval==True post_prompt_format for whichever if_with_eval to split cur_gene and cur_gene_feedback. When if_with_eval == False, tomato.py code should take charge of not using cur_gene_feedback
        post_prompt_format = ", and also evaluate the extracted inspiration in terms of its own quality, how it can potentially help business research hypothesis developing, and how is it related to given background (response format: 'Inspiration: \nEvaluation: \n')."
        post_prompt = "\nPlease give a response to the initial question of exactly extracting the best one sentence or one paragraph from the business-related report (but not from background or evaluation of titles) as a possible inspiration" + post_prompt_format
        if prompt_mode == 0:
            # "usually a hypothesis is more novel if its inspirations are less directly related to the given background" not used, since it should be reflected as a past feedback
            pre_prompt = "Given an academic background in business research and a business-related report, try to collect the best one sentence or one paragraph in the report that possibly contain an inspiration, which could be used together with the given background to further develope a hypothesis in business research. \nThe academic background is "
            mid_prompt = ["\nThe business report is: \n", "\nPrevious feedbacks on how this passage could possibly contribute to a hypothesis by only seeing the title of this inspiration passage: \n"]
        elif prompt_mode == 1:
            pre_prompt = "Given an academic background in business research and a business-related report, try to collect the best one sentence or one paragraph in the report that possibly contain an inspiration, which could be used together with the given background to further develope a hypothesis in business research (usually a hypothesis is more novel if its inspiration is less directly related to the given background). \nThe academic background is "
            mid_prompt = ["\nThe business report is: \n", "\nPrevious feedbacks on how this passage could possibly contribute to a hypothesis by only seeing the title of this inspiration passage: \n"]
        else:
            raise NotImplementedError
    elif module_name == 'hypothesis_suggstor':
        assert if_with_eval == False
        pre_prompt = "Given an academic background in business research and some possible inspirations which combined with the background could lead to meaningful business research hypothesis, please try to give some suggestions on how these inspirations could be combined to be potentially helpful to propose novel business research hypotheses. Multiple inspirations are encouraged to be used together to generate new hypotheses. Inspirations which seem to be less connected to the background could probably contribute more to a novel hypothesis. A good business hypothesis should be novel and not intuitive, should has never been formally proposed in the business research fields ever before. \nThe background is:\n"
        mid_prompt = "\nThe possible inspirations are:\n"
        post_prompt = "Please give a response to the initial question of generating suggestions on how the background and inspirations could be combined to generate novel business research hypotheses. Each suggestion should leverage more than two inspirations (response format: 'Suggestion 1: \nSuggestion 2: \n...')"
    # should be abandoned now
    elif module_name == 'hypothesis_generator_first':
        raise Exception("Using abandoned module: ", module_name)
        # We use this if_with_eval==True post_prompt_format for whichever if_with_eval to split cur_gene and cur_gene_feedback. When if_with_eval == False, tomato.py code should take charge of not using cur_gene_feedback
        # Here if_with_eval does not mean self_eval, but self_present_reasoning_process
        # post_prompt_format = ", and also evaluate the generated hypothesis (response format: 'Hypothesis: \nEvaluation: \n')."
        # post_prompt_format = ", and also give the reasoning process from background and inspirations to hypothesis (response format: 'Hypothesis: \nReasoning process: \nHypothesis: \nReasoning process: \nHypothesis: \nReasoning process: \n...')."
        post_prompt_format = ", and also give the reasoning process from background and inspirations to hypothesis (response format: 'Hypothesis: \nReasoning process: \nHypothesis: \nReasoning process: \n...')."
        pre_prompt = "Given an academic background in business research and some possible inspirations which combined with the background could lead to meaningful business research hypothesis, try to give unique hypotheses based on the background and inspirations. Multiple inspirations are encouraged to be used together to generate new hypotheses. Inspirations which seem to be less connected to the background could probably contribute more to a novel hypothesis. A good business hypothesis should (1) contain an independent variable and a dependent variable, and describe how the independent variable can influence the dependent variable, and (2) be novel and not intuitive, should has never been formally proposed in the business research fields ever before. The background is: \n"
        mid_prompt = "\nThe possible inspirations are: \n"
        post_prompt = "\nPlease give a response to the initial question of generating unique meaningful business research hypotheses given the background and inspirations" + post_prompt_format
    elif module_name == 'hypothesis_generator_first_without_future_fdbk':
        if if_baseline == 0 or if_baseline == 1 or if_baseline == 3:
            post_prompt_format = "For each hypothesis, please give the reasoning processing first, and then give the hypothesis. (response format: 'Reasoning process: \nHypothesis: \nReasoning process: \nHypothesis: \n...')."
            pre_prompt = "Given an academic background in business research and some possible inspirations which combined with the background could lead to meaningful business research hypothesis, try to give unique hypotheses based on the background and inspirations. Multiple inspirations are encouraged to be used together to generate new hypotheses. Inspirations which seem to be less connected to the background could probably contribute more to a novel hypothesis. A good business hypothesis should (1) contain an independent variable and a dependent variable, and describe how the independent variable can influence the dependent variable, and (2) be novel and not intuitive, should has never been formally proposed in the business research fields ever before. The background is: \n"
            mid_prompt = "\nThe possible inspirations are: \n"
            post_prompt = "\nPlease give a response to the initial question of generating unique meaningful business research hypotheses given the background and inspirations. Each hypothesis should leverage more than two inspirations." + post_prompt_format
        elif if_baseline == 2:
            post_prompt_format = "For each hypothesis, please give the reasoning processing first, and then give the hypothesis. (response format: 'Reasoning process: \nHypothesis: \nReasoning process: \nHypothesis: \n...')."
            pre_prompt = "Given an corpus related to business research, try to give unique hypotheses based on the corpus. A good business hypothesis should (1) contain an independent variable and a dependent variable, and describe how the independent variable can influence the dependent variable, and (2) be novel and not intuitive, should has never been formally proposed in the business research fields ever before. The corpus is: \n"
            mid_prompt = ""
            post_prompt = "\nPlease give a response to the initial question of generating unique meaningful business research hypotheses given the corpus. " + post_prompt_format
        else:
            raise NotImplementedError
    elif module_name == 'hypothesis_generator_first_with_future_fdbk':
        assert if_baseline == 0
        post_prompt_format = "For each hypothesis, please give the reasoning processing first, and then give the hypothesis. (response format: 'Reasoning process: \nHypothesis: \nReasoning process: \nHypothesis: \n...')."
        pre_prompt = "Given an academic background in business research, some possible inspirations which combined with the background could lead to meaningful business research hypothesis, and some initial suggestions on how to leverage these inspirations to build hypotheses, try to give unique hypotheses based on the background, inspirations, and the initial suggestions. Multiple inspirations and suggestions are encouraged to be used together to generate new hypotheses. Inspirations which seem to be less connected to the background could probably contribute more to a novel hypothesis. A good business hypothesis should (1) contain an independent variable and a dependent variable, and describe how the independent variable can influence the dependent variable, and (2) be novel and not intuitive, should has never been formally proposed in the business research fields ever before. The background is: \n"
        mid_prompt = ["\nThe possible inspirations are: \n", "\nThe suggestions are:\n"]
        post_prompt = "\nPlease give a response to the initial question of generating unique meaningful business research hypotheses given the background, inspirations, and suggestions. Each hypothesis should leverage more than two suggestions or inspirations." + post_prompt_format
    elif module_name == 'hypothesis_generator_refine' or module_name == 'hypothesis_generator_refine_without_future_fdbk':
        assert module_name != 'hypothesis_generator_refine'
        assert if_baseline == 0
        # We use this if_with_eval==True post_prompt_format for whichever if_with_eval to split cur_gene and cur_gene_feedback. When if_with_eval == False, tomato.py code should take charge of not using cur_gene_feedback
        # post_prompt_format = ", and also give the reasoning process from background, inspirations, previous hypothesis and previous feedbacks to the refined hypothesis (response format: 'Hypothesis: \nReasoning process: \n')."
        post_prompt_format = ", and also concisely answer how the refined hypothesis improves from the feedbacks (response format: 'Reasoning process: \nRefined hypothesis: \n')."
        pre_prompt = "Given an academic background in business research, some possible inspirations which combined with the background could lead to meaningful business research hypothesis, a previous generated hypothesis based on the background and the inspirations, and some feedbacks for the generated hypothesis, try to refine the previous hypothesis by addressing the concerns of the hypothesis in the feedbacks (especially the novelty feedbacks). If the previous hypothesis seriously violates some standards in any feedbacks, the previous hypothesis should be correspondingly largely revised or even be discarded and propose a new one. Multiple inspirations are encouraged to be used together to generate new hypotheses. Inspirations which seem to be less connected to the background could probably contribute more to a novel hypothesis.\n The background is: \n"
        mid_prompt = ["\nThe possible inspirations are: \n", "\nThe previous hypothesis is: \n", "\nThe feedbacks for the previous hypothesis are: \n"]
        # (a good business hypothesis should contain an independent variable and a dependent variable, and describe how the independent variable can influence the dependent variable)
        post_prompt = "\nPlease give a response to the initial question of refining the previous hypothesis to a better business research hypothesis which can address the concerns in the feedbacks" + post_prompt_format
    elif module_name == 'hypothesis_generator_refine_with_future_fdbk':
        assert if_baseline == 0
        post_prompt_format = ", and also concisely answer how the refined hypothesis improves from the feedbacks (response format: 'Reasoning process: \nRefined hypothesis: \n')."
        pre_prompt = "Given an academic background in business research, some possible inspirations which combined with the background could lead to meaningful business research hypothesis, some initial suggestions on how to leverage these inspirations to build hypotheses, a previous generated hypothesis based on the background and the inspirations, and some feedbacks for the generated hypothesis, try to refine the previous hypothesis by addressing the concerns of the hypothesis in the feedbacks (especially the novelty feedbacks). If the previous hypothesis seriously violates some standards in any feedbacks, the previous hypothesis should be correspondingly largely revised or even be discarded and propose a new one. Multiple inspirations and suggestions are encouraged to be used together to generate new hypotheses. Inspirations which seem to be less connected to the background could probably contribute more to a novel hypothesis.\n The background is: \n"
        mid_prompt = ["\nThe possible inspirations are: \n", "\nThe suggestions are:\n", "\nThe previous hypothesis is: \n", "\nThe feedbacks for the previous hypothesis are: \n"]
        # (a good business hypothesis should contain an independent variable and a dependent variable, and describe how the independent variable can influence the dependent variable)
        post_prompt = "\nPlease give a response to the initial question of refining the previous hypothesis to a better business research hypothesis which can address the concerns in the feedbacks" + post_prompt_format
    # TD: self_eval prompt could change to ", and also give some suggestions on how xxx"
    elif module_name == 'deductive_consistency_evaluator':
        if if_with_eval:
            post_prompt_format = ", and also evaluate the proposed feedbacks (response format: 'Feedback: \nEvaluation: \n')."
        else:
            post_prompt_format = "."
        pre_prompt = "Given an academic background in business research, some possible evidences in business field, and a business related hypothesis, try to give some feedbacks on whether the hypothesis violates any background or evidence. The background is: \n"
        mid_prompt = ["\nThe evidences are: \n", "\nThe hypothesis is: \n"]
        post_prompt = "\nPlease give a response to the initial question of providing feedbacks on whether the hypothesis violates any sentence in background or evidences" + post_prompt_format
    # TD: self_eval prompt could change to ", and also give some suggestions on how xxx"
    elif module_name == 'indiscriminate_confirmation_handler':
        if if_with_eval:
            post_prompt_format = ", and also evaluate the proposed feedbacks (response format: 'Feedback: \nEvaluation: \n')."
        else:
            post_prompt_format = "."
        pre_prompt = "Given a research hypothesis in business research, try to give some feedbacks on whether the hypothesis by any chance does not reflects the reality. Please directly answer this question. \nThe hypothesis is: \n"
        mid_prompt = ""
        post_prompt = "\nPlease give a response to the initial question of providing feedbacks on whether the research hypothesis reflects the reality" + post_prompt_format
    elif module_name == 'novelty_detector':
        post_prompt_format = ", and also give some suggestions on how the hypothesis can be more novel (response format: 'Feedback: \nSuggestion: \n')."
        post_prompt = "\nPlease give a responses to the initial question of providing detailed feedbacks on whether the research hypothesis is by any chance not novel (not a semantically direct copy of any inspiration or any argument in existing business literature)" + post_prompt_format
        # prompt_mode == 1 means args.if_novelty_module_have_access_to_surveys == 1
        if prompt_mode == 1:
            # We use this if_with_eval==True post_prompt_format for whichever if_with_eval to split cur_gene and cur_gene_feedback. When if_with_eval == False, tomato.py code should take charge of not using cur_gene_feedback
            pre_prompt = "Given a research hypothesis in business research, some inspirations used for developing the hypothesis, and a possibly related paragraph from a relevant business research survey, try to give some feedbacks on whether the hypothesis is by any chance not novel (the reason is that the hypothesis is used for business research, where novel and not ever proposed hypotheses are preferred). To be novel, the hypothesis should at least not be semantically a direct copy of any inspiration or any arguments in existing business literature (including literatures that are not provided as input), but could be a conclusion from multiple reasoning steps using the inspirations, and probably then with (slightly / some) deviations from the conclusion. \nThe hypothesis is: \n"
            mid_prompt = ["\nThe inspirations used for developing the hypothesis are: \n", "\nOne of the most similar existing business literature paragraph is: \n"]
        elif prompt_mode == 0:
            pre_prompt = "Given a research hypothesis in business research and some inspirations used for developing the hypothesis, try to give some feedbacks on whether the hypothesis is by any chance not novel (the reason is that the hypothesis is used for business research, where novel and not ever proposed hypotheses are preferred). To be novel, the hypothesis should at least not be semantically a direct copy of any inspiration or any arguments in existing business literature, but could be a conclusion from multiple reasoning steps using the inspirations, and probably then with (slightly / some) deviations from the conclusion. \nThe hypothesis is: \n"
            mid_prompt = ["\nThe inspirations used for developing the hypothesis are: \n"]
        else:
            raise NotImplementedError
    elif module_name == 'specification_detector':
        # We use this if_with_eval==True post_prompt_format for whichever if_with_eval to split cur_gene and cur_gene_feedback. When if_with_eval == False, tomato.py code should take charge of not using cur_gene_feedback
        post_prompt_format = ", and also give some suggestions on how the hypothesis can be more specific (response format: 'Feedback: \nSuggestion: \n')."
        pre_prompt = "Given a research hypothesis in business research, try to give some feedbacks on whether the hypothesis is clear and specific enough. By specific, it means a hypothesis should not only indicate two elements are related, but also how they are related, to what extent they are related, why they are related, and which specific sub-elements of the two elements are related. The hypothesis is: \n"
        mid_prompt = ""
        post_prompt = "\nPlease give a response to the initial question on whether the hypothesis is clear and specific enough" + post_prompt_format
    else:
        raise NotImplementedError

    # # To fit to LLMs's pretraining format
    # if "vicuna" in model_name:
    #     pre_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n" + pre_prompt
    # elif "alphaca" in model_name:
    #     raise NotImplementedError
    return pre_prompt, mid_prompt, post_prompt


def print_nvidia_smi():
    print(subprocess.check_output("nvidia-smi", shell=True).decode('utf-8'))


# Function: given a url, return clean text (here clean means primary processing)
# url: "http:..."
# title / clean_text: "text"
def crawler(url):
    # response = requests.get(url)
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    html = response.content
    # Step 2: Parse the HTML with Beautiful Soup
    soup = BeautifulSoup(html, 'html.parser')
    # Extract the title
    # title = soup.title.string if soup.title else "No Title"
    if soup.title:
        title = soup.title.string
    else:
        title = "No Title"
        print("Warning: no title, url: ", url)
    # Step 3: Extract the main text
    # This will vary depending on the structure of the webpage.
    # Here, we're naively assuming that the main text is all in paragraph tags.
    text = ' '.join([p.text for p in soup.find_all('p')])
    # Step 4: Clean up the text (if necessary)
    # This will depend on what "artifacts" are in the text, such as unwanted whitespace,
    # HTML entities, etc. For example, to remove leading/trailing whitespace:
    clean_text = text.strip()
    # some text use too many '\n' within their sentences for display,, while we don't need them
    if clean_text.count('\n') > 80:
        print("clean_text.count('\\n'): ", clean_text.count('\n'))
        clean_text = clean_text.replace('\n', '')
    if len(clean_text) < 45:
        text = ' '.join([p.text for p in soup.find_all(['p', 'div', 'article', 'main'])])
        clean_text = text.strip()
        # some text use too many '\n' within their sentences for display,, while we don't need them
        if clean_text.count('\n') > 80:
            print("clean_text.count('\\n'): ", clean_text.count('\n'))
            clean_text = clean_text.replace('\n', '')
        if len(clean_text) < 45:
            print("Warning: too short passage\nPassage: {}\nurl: {}".format(clean_text, url))
        else:
            print("Warning: better check the passage of url mached with the extracted text: {}".format(url))
    return title, clean_text
