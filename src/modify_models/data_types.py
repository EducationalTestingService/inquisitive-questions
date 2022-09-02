"""
This file is to generate data for inquisitive v.s. informative classifier
format: source + question
modified from data/roberta_data_processing.py
"""

import os
import csv
import argparse
import pandas as pd

from utils import clean_unnecessary_spaces

def get_args():
    parser = argparse.ArgumentParser(description='generate data for rescorer with squad and inquisitive')
    parser.add_argument('-inq', default='/home-nfs/users/data/inquisitive/fairseq/source',
                        help='path of the output file')
    parser.add_argument('-squad', default='/home-nfs/users/data/squad',
                        help='path of the output file')
    parser.add_argument('-out', default='/home-nfs/users/data/inquisitive/discriminator',
                        help='path of the output file')
    return parser.parse_args()

# read csv and remove space before punctuations
def proc_csv(fpath):
    df = pd.read_csv(fpath + '.source', sep='\t\n', encoding='utf-8', names=['source'])
    df['target'] = pd.read_csv(fpath + '.target', header=None, sep='\n', encoding='utf-8').squeeze()
    df['source'] = df['source'] + ' [SEP] ' + df['target']
    df['source'] = df['source'].apply(clean_unnecessary_spaces)
    print(df.shape)
    return df

def proc_neg_csv(fpath):
    df = pd.read_csv(fpath, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')
    df['source'] = df['source'] + ' [SEP] ' + df['target']
    df['source'] = df['source'].apply(clean_unnecessary_spaces)
    print(df.shape)
    return df

def sample_train(pos_df, neg_df, num, fname):
    # sample respectively
    print(pos_df.shape, neg_df.shape)
    pos_df = pos_df.sample(n=num, random_state=1)
    neg_df = neg_df.sample(n=num, random_state=1)
    pos_df = pos_df.assign(**dict.fromkeys(['label'], 1))
    neg_df = neg_df.assign(**dict.fromkeys(['label'], 0))
    print(pos_df.shape, neg_df.shape)

    # append and shuffle
    new_df = pos_df.append(neg_df)
    new_df = new_df.sample(frac=1, random_state=1).reset_index(drop=True)
    print(new_df.shape)

    # output files
    new_df['source'].to_csv(os.path.join(args.out, 'orig_enhanced/' + fname + '.input'), header=False,
                          index=False, sep='\t', encoding='utf-8')
    new_df['label'].to_csv(os.path.join(args.out, 'orig_enhanced/' + fname + '.label'), header=False,
                           index=False, sep='\t', encoding='utf-8')


if __name__ == "__main__":
    args = get_args()

    # sample file
    sample_num = [8000, 1500, 1500]
    fname_list = ['train', 'val', 'test']
    for i, fname in enumerate(fname_list):
        pos_df = proc_csv(os.path.join(args.inq, fname))
        neg_df = proc_neg_csv(os.path.join(args.squad, fname + '.txt'))
        if fname == 'val':
            fname = 'dev'
        sample_train(pos_df, neg_df, sample_num[i], fname)
