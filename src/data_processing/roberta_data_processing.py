# encoding: utf-8
"""
This file is to generate inquisitive and informative data for the discriminator (newest version)

* sample data and make it fully balanced between inquisitive and squad.
* try the same sentences for fair comparison for different settings

possible problem for the whole model: using the same training data for the rescorer and for the BART (i.e. inquisitive)
"""

import os
import argparse
import pandas as pd

from utils import clean_unnecessary_spaces

def get_args():
    parser = argparse.ArgumentParser(description='generate data for rescorer with squad and inquisitive')
    parser.add_argument('-out', default='/home-nfs/users/data/inquisitive/discriminator',
                        help='path of the output file')
    return parser.parse_args()

# read csv and remove space before punctuations
def proc_csv(fname):
    df = pd.read_csv(fname, sep='\t', encoding='utf-8')
    df['target'] = df['target'].apply(clean_unnecessary_spaces)
    return df

def sample_train(pos_1, pos_2, neg_1, neg_2, num, fname):
    pos_df1 = proc_csv(pos_1).rename(columns = {'target':'ner'})
    pos_df2 = proc_csv(pos_2).rename(columns = {'target':'orig'})
    neg_df1 = proc_csv(neg_1).rename(columns = {'target':'ner'})
    neg_df2 = proc_csv(neg_2).rename(columns = {'target':'orig'})
    print(pos_df1.shape, pos_df2.shape, neg_df1.shape, neg_df2.shape)

    # join to the same dataframe
    pos_df = pd.concat([pos_df1, pos_df2.drop(columns='label')], axis=1, join="inner")
    neg_df = pd.concat([neg_df1, neg_df2.drop(columns='label')], axis=1, join="inner")
    print(pos_df.shape, neg_df.shape)

    # sample respectively
    pos_df = pos_df.sample(n=num, random_state=1)
    neg_df = neg_df.sample(n=num, random_state=1)
    print(pos_df.shape, neg_df.shape)

    # append and shuffle
    new_df = pos_df.append(neg_df)
    new_df = new_df.sample(frac=1, random_state=1).reset_index(drop=True)
    print(new_df.shape)

    # output files
    new_df['orig'].to_csv(os.path.join(args.out, 'orig/' + fname + '.input'), header=False,
                          index=False, sep='\t', encoding='utf-8')
    new_df['ner'].to_csv(os.path.join(args.out, 'ner/' + fname + '.input'), header=False,
                         index=False, sep='\t', encoding='utf-8')
    new_df['label'].to_csv(os.path.join(args.out, 'orig/' + fname + '.label'), header=False,
                           index=False, sep='\t', encoding='utf-8')
    new_df['label'].to_csv(os.path.join(args.out, 'ner/' + fname + '.label'), header=False,
                           index=False, sep='\t', encoding='utf-8')


if __name__ == "__main__":
    args = get_args()

    # sample file
    sample_num = [8000, 1500, 1500]
    fname_list = ['train', 'dev', 'test']
    for i, fname in enumerate(fname_list):
        pos_1 = os.path.join(args.out, '_'.join(['inquisitive', 'ner', fname]) + '.txt')
        pos_2 = os.path.join(args.out, '_'.join(['inquisitive', 'orig', fname]) + '.txt')
        neg_1 = os.path.join(args.out, '_'.join(['squad', 'ner', fname]) + '.txt')
        neg_2 = os.path.join(args.out, '_'.join(['squad', 'orig', fname]) + '.txt')
        sample_train(pos_1, pos_2, neg_1, neg_2, sample_num[i], fname)


