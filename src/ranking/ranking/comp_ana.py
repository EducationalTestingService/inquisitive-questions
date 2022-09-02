"""
This file is to get best questions and generate MTurk files
"""
import os
import glob
import json
import numpy as np
import argparse
import pandas as pd
from data_utils import clean_unnecessary_spaces
import matplotlib.pyplot as plt


def get_order(data_df):
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
    return final_idx


if __name__ == '__main__':
    # read results
    df = pd.read_csv('/home-nfs/users/output/inquisitive/MTurk/rc_inquisitive_mturk_input_cssnqt_f.csv',
                     sep='\t', encoding='utf-8')

    # get statistics
    print(df['max'].value_counts())
    df['index'] = pd.read_csv('/home-nfs/users/output/inquisitive/MTurk/rank_index.csv', sep='\t', encoding='utf-8')
    df = df.set_index(['index'])

    # compare with rqt
    # rqt_index = pd.read_csv('/home-nfs/users/output/inquisitive/MTurk/rqt_index.csv', sep='\t', encoding='utf-8')
    # comp_df = df.loc[rqt_index.set_index(['0']).index]
    rqt_df = pd.read_csv('/home-nfs/users/data/inquisitive/fairseq/source_qtype/test.csv', sep='\t', encoding='utf-8')
    final_idx = get_order(rqt_df)
    rqt_df = rqt_df.loc[final_idx].reset_index(drop=True).sample(frac=1, random_state=3)
    rqt_df['rank'] = df['max']
    rqt_df['Diff'] = np.where(rqt_df['rank'] == rqt_df['qtype'], 1, 0)
    print(rqt_df['Diff'].sum())
    rqt_df = rqt_df[['rank', 'qtype', 'Diff']]
    print(rqt_df.groupby(['rank', 'Diff']).count())
    print(rqt_df.groupby(['qtype', 'Diff']).count())

    # compare with nqt
    nqt_index = pd.read_csv('/home-nfs/users/output/inquisitive/MTurk/nqt_index.csv', sep='\t', encoding='utf-8')
    df.loc[rqt_index.set_index(['0']).index]
