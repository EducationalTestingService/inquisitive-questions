"""
This file is to generate all questions with ranked question type
"""

import os
import glob
import numpy as np
import argparse
import pandas as pd
from data_utils import clean_unnecessary_spaces
# from fairseq.models.roberta import RobertaModel
# from run_ranking import predict

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "MTurk sample generations"
    parser.add_argument('-data_dir', type=str, default='/home-nfs/users/output/inquisitive/qtype_allgen/')
    parser.add_argument('-data_path', default='/home-nfs/users/data/inquisitive/fairseq'
                                              '/source/test.csv',
                        help='path of the 50 expert questions')


    parser.add_argument('-model_dir', type=str, default='/home-nfs/users/output/inquisitive'
                                                        '/ranking/HM/source_q1_q2_modify')
    parser.add_argument('-checkpoint_name', type=str, default='checkpoint_best.pt')


    parser.add_argument('-model_name', default='source_span_qtype',
                        help='model name')
    parser.add_argument('-log_path', default='/home-nfs/users/output/inquisitive/fairseq/',
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
    args.log_path = os.path.join(args.log_path, args.model_name)
    np.random.seed(1)

    # # load model
    # roberta = RobertaModel.from_pretrained(args.model_dir, checkpoint_file=args.checkpoint_name,
    #                                        data_name_or_path=args.model_dir.replace('output', 'data'))
    # roberta.eval()  # disable dropout
    # roberta.cuda()

    # get all questions
    df_question = get_all_questions(args.data_dir + args.model_name)
    df_question = df_question.rename(columns={"Definition": 1, "Background": 2, "Instantiation": 3,
                                              "Explanation": 4, "Forward": 5, "Elaboration": 6})
    df_question['sentence'] = (pd.read_csv(args.data_path, sep='\t', encoding='utf-8')['source'])\
        .astype('str').apply(clean_unnecessary_spaces)

    # # get ranked questions
    # output_df = df_question.copy().drop(['sentence'], axis=1)
    # for col in [1, 2, 3, 4, 5, 6]:
    #     output_df[col].values[:] = 0.
    # score_df = output_df.copy()
    # label_fn = lambda label: roberta.task.label_dictionary.string(
    #     [label + roberta.task.label_dictionary.nspecial]
    # )
    # for row in range(0, df_question.shape[0]):
    #     for i in range(1, 7, 1):
    #         for j in range(1, 7, 1):
    #             if i != j:
    #                 sentence = df_question.at[row, 'sentence'] + ' [SEP] ' + \
    #                            df_question.at[row, i] + ' [SEP] ' + df_question.at[row, j]
    #                 pred, scorei, scorej = predict(sentence, roberta, label_fn)
    #                 output_df.at[row, i] += pred
    #                 output_df.at[row, j] += 1 - pred
    #                 score_df.at[row, i] += scorei
    #                 score_df.at[row, j] += scorej
    # output_df = output_df.rename(columns={1: "Definition", 2: "Background", 3: "Instantiation",
    #                                       4: "Explanation", 5: "Forward", 6: "Elaboration"})
    # score_df = score_df.rename(columns={1: "Definition", 2: "Background", 3: "Instantiation",
    #                                     4: "Explanation", 5: "Forward", 6: "Elaboration"})
    # output_df.to_csv(os.path.join(args.log_path, 'test_ranked_r.csv'), sep='\t',
    #                                encoding='utf-8', index=False)
    # score_df.to_csv(os.path.join(args.log_path, 'test_ranked_s.csv'), sep='\t',
    #                  encoding='utf-8', index=False)
    output_df = pd.read_csv(os.path.join(args.log_path, 'test_ranked_r.csv'), sep='\t', encoding='utf-8')
    score_df = pd.read_csv(os.path.join(args.log_path, 'test_ranked_s.csv'), sep='\t', encoding='utf-8')

    output_df['max_all'] = output_df.eq(output_df.max(1), axis=0).dot(output_df.columns + ',').str.rstrip(',')
    output_df['max'] = output_df['max_all'].copy()
    df_question = df_question.assign(**dict.fromkeys(['selected'], ' '))
    df_question = df_question.rename(columns={1: "Definition", 2: "Background", 3: "Instantiation",
                                              4: "Explanation", 5: "Forward", 6: "Elaboration"})
    for row in range(0, output_df.shape[0]):
        if ',' in output_df.at[row, 'max_all']:
            col_values = output_df.at[row, 'max_all'].split(',')
            output_df.at[row, 'max'] = score_df[col_values].loc[row].idxmax()
        df_question.at[row, 'selected'] = df_question.at[row, output_df['max'].loc[row]]

    # write to file
    df_question['selected'].to_csv(os.path.join(args.log_path, 'test_ranked_qtype.txt'), sep='\t',
                                   encoding='utf-8', index=False, header=False)

