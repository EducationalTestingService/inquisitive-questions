"""
This file is to select and rank question types by scores
"""
import os
import glob
import torch
import numpy as np
import argparse
import pandas as pd
from data_utils import clean_unnecessary_spaces
from fairseq.models.roberta import RobertaModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "MTurk sample generations"

    # load roberta model
    parser.add_argument('-model_dir', type=str, default='/home-nfs/users/output/inquisitive/roberta/orig_enhanced/')
    parser.add_argument('-checkpoint_name', type=str, default='checkpoint_best.pt')
    parser.add_argument('-data_path', type=str, default='/home-nfs/users/data/inquisitive/discriminator/orig_enhanced')

    # output path
    parser.add_argument('-log_path', default='/home-nfs/users/output/inquisitive/qtype_allgen/',
                        help='path of the log file')

    return parser.parse_args()


def get_score(tokens, roberta):
    tokens_encoded = roberta.encode(tokens)
    return float(roberta.predict('rescorer_head', tokens_encoded).data.to('cpu')[0][1])


if __name__ == "__main__":
    args = get_args()
    f_path_list = glob.glob('/home-nfs/users/output/inquisitive/qtype_allgen/context_source_span_qtype/test_*.txt')

    # load model
    roberta = RobertaModel.from_pretrained(args.model_dir, checkpoint_file=args.checkpoint_name,
                                           data_name_or_path=args.data_path)
    roberta.eval()  # disable dropout
    roberta.cuda()

    # get all questions
    for f_path in f_path_list:
        qtype = os.path.basename(f_path)[5:-4]
        if 'df_question' in locals():
            df_question[qtype] = pd.read_csv(f_path, sep='\t', encoding='utf-8', header=None).squeeze()
        else:
            df_question = pd.read_csv(f_path, sep='\t', encoding='utf-8', header=None).squeeze().rename(qtype).to_frame()

    # calculate all scores
    for key in df_question.columns.values:
        df_question[key] = df_question[key].apply(lambda x: get_score(x, roberta))

    df_question.to_excel(os.path.join(args.log_path, 'cssq_new_inqusitive_test_scores.xlsx'), index=False)


    # read df_question
    df_question = pd.read_excel(os.path.join(args.log_path, 'cssq_new_inqusitive_test_scores.xlsx'))

    # # top 3 types
    # df_top3 = pd.DataFrame(df_question.apply(lambda s: s.nlargest(3).index.tolist(), axis=1).tolist(),
    #                        columns=['1', '2', '3'])
    # print(df_top3['1'].value_counts())
    # print((df_top3['1'].value_counts() + df_top3['2'].value_counts() + df_top3['3'].value_counts()).sort_values(ascending=False))
    # df_top3.to_excel(os.path.join(args.log_path, 'cssq_rank_test.xlsx'), index=False)

    # # remove other
    # df_question.drop(['Other'], axis=1, inplace=True)
    # df_top3 = pd.DataFrame(df_question.apply(lambda s: s.nlargest(3).index.tolist(), axis=1).tolist(),
    #                        columns=['1', '2', '3'])
    # print(df_top3['1'].value_counts())
    # print((df_top3['1'].value_counts() + df_top3['2'].value_counts() + df_top3['3'].value_counts()).sort_values(ascending=False))
    # df_top3.to_excel(os.path.join(args.log_path, 'cssq_rank_test_no_other.xlsx'), index=False)

    # compare with reference question type
    df_top1 = df_question.idxmax(axis=1)
    refer_qtype = pd.read_csv('/home-nfs/users/data/inquisitive/fairseq/context_source_span_qtype/test.source',
                              sep='\t', encoding='utf-8', header=None).squeeze()
    refer_qtype = refer_qtype.str.replace(' [SEP] ', '\t', regex=False).str.split('\t').str[-1]
    df_comp = df_top1.rename('pred').to_frame()
    df_comp['refer'] = refer_qtype
    df_comp['eq'] = refer_qtype.eq(df_top1)
    df_top3 = pd.DataFrame(df_question.apply(lambda s: s.nlargest(3).index.tolist(), axis=1).tolist(),
                           columns=['1', '2', '3'])
    (refer_qtype.eq(df_top3['1']) | refer_qtype.eq(df_top3['2'])).value_counts()
    print(df_comp.loc[df_comp['eq']]['refer'].value_counts())
    df_comp = df_comp.loc[df_comp['eq'] == False]
    print(df_comp['refer'].value_counts())
    print(df_comp['pred'].value_counts())

    # generate top 3 questions selected by the classifier




