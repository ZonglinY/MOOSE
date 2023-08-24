import pandas as pd
import numpy as np
import os, argparse, validators, math
from utils import crawler

# change none to ""
#   data: 'text' or none or '\'
def simple_processing(data):
    # if_collect_real_data
    if_collect_real_data = True
    if isinstance(data, str):
        data = data.strip()
        if data == '\\' or data == '':
            if_collect_real_data = False
    elif math.isnan(data):
        if_collect_real_data = False
    # return
    if if_collect_real_data:
        if len(data) < 10:
            print("Warning: data: {}".format(data))
        return data
    else:
        return ""



def converter():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir", type=str, default="./Data/")
    args = parser.parse_args()

    # main_corpus: raw data
    main_corpus = pd.read_excel(os.path.join(args.root_data_dir, 'summary_to_read_with_pandas.xlsx'))
    # format_corpus: correct format, only to collect its format
    format_corpus = pd.read_excel(os.path.join(args.root_data_dir, 'raw_corpus_trial.xlsx'))
    assert list(main_corpus.keys()) == list(format_corpus.keys())

    # final data
    # columns
    columns = list(main_corpus.keys())
    format_columns = list(format_corpus.keys())
    assert columns == ['No', 'Title', 'Link', 'Date', 'background_1_golden', 'background_1_title', 'background_1_passage', 'background_1_link', 'background_2_golden', 'background_2_title', 'background_2_passage', 'background_2_link', 'inspiration_1_golden', 'inspiration_1_title', 'inspiration_1_passage', 'inspiration_1_link', 'inspiration_2_golden', 'inspiration_2_title', 'inspiration_2_passage', 'inspiration_2_link', 'inspiration_3_golden', 'inspiration_3_title', 'inspiration_3_passage', 'inspiration_3_link', 'Main hypotheis', 'Complexity (logic)', 'Complexity (generating)', 'Steps']
    No = list(main_corpus['No'])
    Title = list(main_corpus['Title'])
    Link = list(main_corpus['Link'])
    Date = list(main_corpus['Date'])
    Main_hypotheis = list(main_corpus['Main hypotheis'])
    Complexity_logic = list(main_corpus['Complexity (logic)'])
    Complexity_generating = list(main_corpus['Complexity (generating)'])
    Steps = list(main_corpus['Steps'])

    ## Prepare remaining final data
    background_1_golden, background_1_title, background_1_passage, background_1_link, background_2_golden, background_2_title, background_2_passage, background_2_link, inspiration_1_golden, inspiration_1_title, inspiration_1_passage, inspiration_1_link, inspiration_2_golden, inspiration_2_title, inspiration_2_passage, inspiration_2_link, inspiration_3_golden, inspiration_3_title, inspiration_3_passage, inspiration_3_link = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for cur_id in range(len(main_corpus)):
        # background_1
        cur_background_1_golden = simple_processing(main_corpus['background_1_golden'][cur_id])
        cur_background_1_link = simple_processing(main_corpus['background_1_link'][cur_id])
        if cur_background_1_link != "":
            assert cur_background_1_golden != ""
            cur_background_1_title, cur_background_1_passage = crawler(cur_background_1_link)
            cur_background_1_title = simple_processing(cur_background_1_title)
            cur_background_1_passage = simple_processing(cur_background_1_passage)
        else:
            assert cur_background_1_golden == ""
            cur_background_1_title, cur_background_1_passage = "", ""
        background_1_golden.append(cur_background_1_golden)
        background_1_title.append(cur_background_1_title)
        background_1_passage.append(cur_background_1_passage)
        background_1_link.append(cur_background_1_link)
        # background_2
        cur_background_2_golden = simple_processing(main_corpus['background_2_golden'][cur_id])
        cur_background_2_link = simple_processing(main_corpus['background_2_link'][cur_id])
        if cur_background_2_link != "":
            assert cur_background_2_golden != ""
            cur_background_2_title, cur_background_2_passage = crawler(cur_background_2_link)
            cur_background_2_title = simple_processing(cur_background_2_title)
            cur_background_2_passage = simple_processing(cur_background_2_passage)
        else:
            assert cur_background_2_golden == ""
            cur_background_2_title, cur_background_2_passage = "", ""
        background_2_golden.append(cur_background_2_golden)
        background_2_title.append(cur_background_2_title)
        background_2_passage.append(cur_background_2_passage)
        background_2_link.append(cur_background_2_link)
        # inspiration_1
        cur_inspiration_1_golden = simple_processing(main_corpus['inspiration_1_golden'][cur_id])
        cur_inspiration_1_link = simple_processing(main_corpus['inspiration_1_link'][cur_id])
        if cur_inspiration_1_link != "":
            assert cur_inspiration_1_golden != ""
            cur_inspiration_1_title, cur_inspiration_1_passage = crawler(cur_inspiration_1_link)
            cur_inspiration_1_title = simple_processing(cur_inspiration_1_title)
            cur_inspiration_1_passage = simple_processing(cur_inspiration_1_passage)
        else:
            assert cur_inspiration_1_golden == ""
            cur_inspiration_1_title, cur_inspiration_1_passage = "", ""
        inspiration_1_golden.append(cur_inspiration_1_golden)
        inspiration_1_title.append(cur_inspiration_1_title)
        inspiration_1_passage.append(cur_inspiration_1_passage)
        inspiration_1_link.append(cur_inspiration_1_link)
        # inspiration_2
        cur_inspiration_2_golden = simple_processing(main_corpus['inspiration_2_golden'][cur_id])
        cur_inspiration_2_link = simple_processing(main_corpus['inspiration_2_link'][cur_id])
        if cur_inspiration_2_link != "":
            assert cur_inspiration_2_golden != ""
            cur_inspiration_2_title, cur_inspiration_2_passage = crawler(cur_inspiration_2_link)
            cur_inspiration_2_title = simple_processing(cur_inspiration_2_title)
            cur_inspiration_2_passage = simple_processing(cur_inspiration_2_passage)
        else:
            assert cur_inspiration_2_golden == ""
            cur_inspiration_2_title, cur_inspiration_2_passage = "", ""
        inspiration_2_golden.append(cur_inspiration_2_golden)
        inspiration_2_title.append(cur_inspiration_2_title)
        inspiration_2_passage.append(cur_inspiration_2_passage)
        inspiration_2_link.append(cur_inspiration_2_link)
        # inspiration_3
        cur_inspiration_3_golden = simple_processing(main_corpus['inspiration_3_golden'][cur_id])
        cur_inspiration_3_link = simple_processing(main_corpus['inspiration_3_link'][cur_id])
        if cur_inspiration_3_link != "":
            assert cur_inspiration_3_golden != ""
            cur_inspiration_3_title, cur_inspiration_3_passage = crawler(cur_inspiration_3_link)
            cur_inspiration_3_title = simple_processing(cur_inspiration_3_title)
            cur_inspiration_3_passage = simple_processing(cur_inspiration_3_passage)
        else:
            assert cur_inspiration_3_golden == ""
            cur_inspiration_3_title, cur_inspiration_3_passage = "", ""
        inspiration_3_golden.append(cur_inspiration_3_golden)
        inspiration_3_title.append(cur_inspiration_3_title)
        inspiration_3_passage.append(cur_inspiration_3_passage)
        inspiration_3_link.append(cur_inspiration_3_link)

    # final_data = [No, Title, Link, Date, background_1_golden, background_1_title, background_1_passage, background_1_link, background_2_golden, background_2_title, background_2_passage, background_2_link, inspiration_1_golden, inspiration_1_title, inspiration_1_passage, inspiration_1_link, inspiration_2_golden, inspiration_2_title, inspiration_2_passage, inspiration_2_link, inspiration_3_golden, inspiration_3_title, inspiration_3_passage, inspiration_3_link, Main_hypotheis, Complexity_logic, Complexity_generating, Steps]
    df = pd.DataFrame(list(zip(No, Title, Link, Date, background_1_golden, background_1_title, background_1_passage, background_1_link, background_2_golden, background_2_title, background_2_passage, background_2_link, inspiration_1_golden, inspiration_1_title, inspiration_1_passage, inspiration_1_link, inspiration_2_golden, inspiration_2_title, inspiration_2_passage, inspiration_2_link, inspiration_3_golden, inspiration_3_title, inspiration_3_passage, inspiration_3_link, Main_hypotheis, Complexity_logic, Complexity_generating, Steps)), columns=columns)
    df.to_excel(os.path.join(args.root_data_dir, 'business_research.xlsx'))





if __name__ == "__main__":
    converter()
    print("finished")
