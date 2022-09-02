"""
This is to inference the MTurk samples
"""


"""
This is to generate all the questions (using four different models) for MTurk
"""

import os
import glob
import numpy as np
import argparse
import pandas as pd
from data_utils import clean_unnecessary_spaces

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "MTurk sample generations"
    parser.add_argument('-data_dir', type=str, default='/home-nfs/users/output/inquisitive/')
    parser.add_argument('-data_path', default='/home-nfs/users/data/inquisitive/fairseq'
                                              '/context_source_span_qtype/test.csv',
                        help='path of the 50 expert questions')
    parser.add_argument('-log_path', default='/home-nfs/users/output/inquisitive/MTurk/',
                        help='path of the log file')

    return parser.parse_args()


def get_all_questions(file_path):
    "random select question"
    f_path_list = glob.glob(os.path.join(file_path, 'test_*.txt'))
    for f_path in f_path_list:
        lname = os.path.basename(f_path)[5:-4]
        if 'df_question' in locals():
            df_question[lname] = pd.read_csv(f_path, sep='\t', encoding='utf-8', header=None).squeeze()
        else:
            df_question = pd.read_csv(f_path, sep='\t', encoding='utf-8', header=None).squeeze().rename(lname).to_frame()
    return df_question.drop(columns=['Other'])


if __name__ == "__main__":
    args = get_args()
    np.random.seed(1)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path, exist_ok=True)

    model_dir = {'cssnqt': 'qtype_allgen/context_source_span_qtype'}

    # first sample 500 sentences from test set, make sure that the 300 expert questions are excluded
    data_df = pd.read_csv(args.data_path, sep='\t', encoding='utf-8')

    # get 1st batch index
    new_df_1 = data_df.sample(n=50, random_state=1)

    # get unique source and exclude 1st batch
    for i in new_df_1.index:
        if i in data_df.index:
            Article_Id = data_df.at[i, 'Article_Id']
            Sentence_Id = data_df.at[i, 'Sentence_Id']
            data_df = data_df.loc[(data_df['Article_Id'] != Article_Id) | (data_df['Sentence_Id'] != Sentence_Id)]

    data_df_unique = data_df.groupby(['Article_Id', 'Sentence_Id'], group_keys=False).apply(
        lambda data_df: data_df.sample(1, random_state=1))
    data_df_2 = data_df_unique.sample(n=250, random_state=1)

    # get the unique index part (417)
    data_df_unique = data_df_unique.drop(data_df_2.index)
    final_idx = data_df_unique.index

    # get the non-unique index part (83)
    for i in data_df_2.index:
        if i in data_df.index:
            new_df = data_df.loc[(data_df['Article_Id'] == data_df_2.at[i, 'Article_Id']) &
                                 (data_df['Sentence_Id'] == data_df_2.at[i, 'Sentence_Id']) &
                                 (data_df['Span'] != data_df_2.at[i, 'Span'])]
            if new_df.shape[0] > 0:
                final_idx = final_idx.union(new_df.sample(1, random_state=1).index)
                if final_idx.shape[0] >= 500:
                    break

    # get sentences and context, find span position
    final_df = pd.read_csv(args.data_path.replace('context_source_span_qtype', 'context_source_span'),
                          sep='\t', encoding='utf-8').loc[final_idx]
    final_df['context_id'] = 'context_' + final_df['Article_Id'].astype('str') + '_' \
                             + final_df['Sentence_Id'].astype('str')
    final_df['sentence_id'] = 'source_' + final_df['Article_Id'].astype('str') + '_' \
                              + final_df['Sentence_Id'].astype('str')
    final_df['question_id'] = 'question_' + final_df.index.to_series().astype('str')
    final_df.rename(columns={"prev_sent": "context"}, inplace=True)
    final_df['source'] = final_df['source'].str.replace(' [SEP] ', '\t',
                                                       regex=False).str.split('\t').apply(lambda x: x[1])
    final_df['sentence'] = final_df['source'].copy()
    final_df = final_df.reset_index(drop=True)
    for row in range(0, final_df.shape[0]):
        final_df.at[row, 'sentence'] = clean_unnecessary_spaces(final_df.at[row, 'sentence'])
    final_df = final_df[['sentence']]

    # get all questions
    for i, key in enumerate(model_dir):
        df_question = get_all_questions(args.data_dir + model_dir[key]).loc[final_idx].reset_index(drop=True)
        final_df = pd.concat([final_df, df_question], axis=1)
        # shuffle order and output
        output_df = final_df.copy()
        output_df = output_df.sample(frac=1, random_state=3)
        output_df.to_csv(os.path.join(args.log_path, 'rc_inquisitive_mturk_input_' + key + '.csv'), sep='\t',
                         encoding='utf-8', index=False)

