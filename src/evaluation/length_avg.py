"""
Get average length for all test questions
"""

import os
import glob
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "length computation"
    parser.add_argument('-data_dir', type=str, default='/home-nfs/users/output/inquisitive/fairseq/')
    parser.add_argument('-log_path', default='/home-nfs/users/output/inquisitive/fairseq/',
                        help='path of the log file')
    parser.add_argument('-log_name', default='test_ranked_qtype.txt',
                        help='name of the log file')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    f_path_list = glob.glob(os.path.join(args.log_path, '*/' + args.log_name))
    avg_dict = dict()
    median_dict = dict()
    for fpath in f_path_list:
        df = pd.read_csv(fpath, sep='\t', encoding='utf-8', names=['questions'])

        # filter out 300 questions
        df_index = pd.read_csv('/home-nfs/users/output/inquisitive/fairseq/test_index.csv', names=['index'],
                               sep='\t', encoding='utf-8')
        df_index = df_index.set_index(['index'])
        df = df.iloc[df_index.index[1:]]

        df['questions'] = df['questions'].apply(word_tokenize)
        df['Length'] = df['questions'].str.len()
        avg_dict[os.path.basename(os.path.dirname(fpath))] = df['Length'].mean()
        median_dict[os.path.basename(os.path.dirname(fpath))] = df['Length'].median()
    avg_df = pd.DataFrame.from_dict(avg_dict, orient='index', columns=['avg_score']).reset_index()
    avg_df = avg_df.sort_values(by=['avg_score']).reset_index(drop=True)
    print(avg_df)
    median_df = pd.DataFrame.from_dict(median_dict, orient='index', columns=['median_score']).reset_index()
    median_df = median_df.sort_values(by=['median_score']).reset_index(drop=True)
    print(median_df)