"""
This file is to generate all questions with non reference question type
"""

import os
import glob
import numpy as np
import argparse
import pandas as pd
# from ..data_utils import clean_unnecessary_spaces
from fairseq.models.roberta import RobertaModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "NQT all test question orgnization"
    parser.add_argument('-data_dir', type=str, default='/home-nfs/users/output/inquisitive/qtype_allgen/')

    parser.add_argument('-model_dir', type=str, default='/home-nfs/users/output/inquisitive/roberta/orig/')
    parser.add_argument('-checkpoint_name', type=str, default='checkpoint_best.pt')
    parser.add_argument('-data_path', type=str, default='/home-nfs/users/data/inquisitive/discriminator/orig')


    parser.add_argument('-model_name', default='source_qtype',
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


def get_score(tokens, roberta):
    tokens_encoded = roberta.encode(tokens)
    return float(roberta.predict('rescorer_head', tokens_encoded).data.to('cpu')[0][1])


if __name__ == "__main__":
    args = get_args()
    args.log_path = os.path.join(args.log_path, args.model_name)
    np.random.seed(1)

    # load model
    roberta = RobertaModel.from_pretrained(args.model_dir, checkpoint_file=args.checkpoint_name,
                                           data_name_or_path=args.data_path)
    roberta.eval()  # disable dropout
    roberta.cuda()

    # get all questions
    df_question = get_all_questions(args.data_dir + args.model_name)
    output_df = df_question.copy()

    # get ranked questions
    for key in output_df.columns.values:
        output_df[key] = output_df[key].apply(lambda x: get_score(x, roberta))

    df_top1 = output_df.idxmax(axis=1)
    df_question = df_question.assign(**dict.fromkeys(['selected'], ' '))
    for row in range(0, df_question.shape[0]):
        df_question.at[row, 'selected'] = df_question.at[row, df_top1.loc[row]]

    # write to file
    df_question['selected'].to_csv(os.path.join(args.log_path, 'test_nqtype.txt'), sep='\t',
                                   encoding='utf-8', index=False, header=False)

