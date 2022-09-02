"""
This file is to organize samples
selected by informative vs. inquisitive classifier
"""
import os
import glob
import torch
import numpy as np
import argparse
import pandas as pd
from data_utils import clean_unnecessary_spaces

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "sample generations"
    # output path
    parser.add_argument('-log_path', default='/home-nfs/users/output/inquisitive/qtype_allgen/',
                        help='path of the log file')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    df_scores = pd.read_excel(os.path.join(args.log_path, 'cssq_inqusitive_scores.xlsx'))
    df_top3 = pd.DataFrame(df_scores.apply(lambda s: s.nlargest(3).index.tolist(), axis=1).tolist(),
                           columns=['1', '2', '3'])

    # Get questions
    f_path_list = glob.glob('/home-nfs/users/output/inquisitive/qtype_allgen/context_source_span_qtype/dev_*.txt')
    for f_path in f_path_list:
        qtype = os.path.basename(f_path)[4:-4]
        if 'df_question' in locals():
            df_question[qtype] = pd.read_csv(f_path, sep='\t', encoding='utf-8', header=None).squeeze()
        else:
            df_question = pd.read_csv(f_path, sep='\t', encoding='utf-8', header=None).squeeze().rename(qtype).to_frame()

    # get context
    cxt_path = '/home-nfs/users/data/inquisitive/fairseq/context_source_span/val.csv'
    df_context = pd.read_csv(cxt_path, sep='\t', encoding='utf-8')
    df_context = pd.concat([df_context[['source']], df_top3], axis=1, join="inner")
    for row in range(0, df_context.shape[0]):
        df_context.at[row, '1'] = df_question.at[row, df_context.at[row, '1']]
        df_context.at[row, '2'] = df_question.at[row, df_context.at[row, '2']]
        df_context.at[row, '3'] = df_question.at[row, df_context.at[row, '3']]
    df_context.to_excel(os.path.join(args.log_path, 'cssq_rank_dev_all_samples.xlsx'), index=False)