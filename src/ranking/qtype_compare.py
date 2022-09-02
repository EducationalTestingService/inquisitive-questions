"""
For fifty test sentences generate all the questions using the different question types for ETS experts.
And then prepare a excel/csv file with the following format,
context - source text - span- generated question - question type (e.g., causal, elaboration)
Note, for each row here, we are generating multiple outputs (e.g., questions) here that need to evaluated.
"""

import os
import glob
import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "compare different generations with"

    # data set directory
    parser.add_argument('-data_path', default='/home-nfs/users/data/inquisitive/fairseq'
                                              '/context_source_span_qtype/test.csv',
                        help='path of the log file')
    parser.add_argument('-file_path', default='/home-nfs/users/output/inquisitive/qtype_allgen/',
                        help='path of the log file')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # model_name_list = ['context_source_qtype', 'source_qtype', 'context_source_span_qtype', 'source_span_qtype']
    model_name_list = ['context_source_span_qtype']

    # get data
    data_df = pd.read_csv(args.data_path, sep='\t', encoding='utf-8')
    data_df['source'] = data_df['source'].str.replace(' [SEP] ', '\t', regex=False).str.split('\t')
    new_df = pd.DataFrame(data_df['source'].tolist(), columns=['context', 'source', 'span', 'qtype'])
    new_df.drop(['qtype'], axis=1, inplace=True)

    # get 1st batch index
    new_df_1 = new_df.sample(n=50, random_state=1)
    new_df = new_df.drop(new_df_1.index)

    # get unique source
    for i in new_df_1.index:
        if i in data_df.index:
            Article_Id = data_df.at[i, 'Article_Id']
            Sentence_Id = data_df.at[i, 'Sentence_Id']
            data_df = data_df.loc[(data_df['Article_Id'] != Article_Id) | (data_df['Sentence_Id'] != Sentence_Id)]

    data_df = data_df.groupby(['Article_Id', 'Sentence_Id'], group_keys=False).apply(
        lambda data_df: data_df.sample(1, random_state=1))
    data_df = data_df.sample(n=250, random_state=1)
    new_df = new_df.loc[data_df.index]

    # get test file
    for model in model_name_list:
        file_list = glob.glob(os.path.join(args.file_path, model) + '/test_*.txt')
        output_df = new_df.copy()
        for fname in file_list:
            df = pd.read_csv(fname, sep='\t', encoding='utf-8', header=None).squeeze()
            cname = os.path.basename(fname)[5:-4]
            output_df[cname] = df[df.index.isin(output_df.index)]
        output_df = output_df.drop(['Other'], axis=1)
        output_df.to_excel(os.path.join(args.file_path, model + '_2.xlsx'), index=False)