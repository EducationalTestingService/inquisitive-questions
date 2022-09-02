"""
This file is to remove all punctuations of original data for BART
"""

import os
import glob
import argparse
import pandas as pd

from shutil import copyfile
from utils import clean_unnecessary_spaces

def get_args():
    parser = argparse.ArgumentParser(description='remove all punctuations of original data for BART')
    parser.add_argument('-old_path', default='/home-nfs/users/data/inquisitive/fairseq_past',
                        help='path of the output file')
    parser.add_argument('-new_path', default='/home-nfs/users/data/inquisitive/fairseq',
                        help='path of the output file')
    return parser.parse_args()

def glob_path(args):
    dir_list = glob.glob(os.path.join(args.old_path, '*source*'))
    return dir_list

if __name__ == "__main__":
    args = get_args()

    dir_list = glob_path(args)
    for dir_path in dir_list:
        output_path = dir_path.replace('fairseq_past', 'fairseq')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # read csv
        csv_list = glob.glob(os.path.join(dir_path, '*.csv'))
        for file_path in csv_list:
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')

            # remove that line of dev
            if 'val' in file_path:
                no_span_idx = df.index[df['Span'] == 'NO_SPAN'].tolist()
                if not no_span_idx:
                    no_span_idx = df.index[df['Span'].isnull()].tolist()
                df = df.drop(no_span_idx).reset_index(drop=True)

            df.to_csv(file_path.replace('fairseq_past', 'fairseq'), index=False, sep='\t', encoding='utf-8')

            # remove extra spaces
            df['source'] = df['source'].apply(clean_unnecessary_spaces)
            df['target'] = df['target'].apply(clean_unnecessary_spaces)
            df['Span'] = df['Span'].apply(clean_unnecessary_spaces)
            df['prev_sent'] = df['prev_sent'].apply(clean_unnecessary_spaces)

            file_name = os.path.basename(file_path).split('.')[0]

            src_file = os.path.join(output_path, file_name + '.source')
            tgt_file = os.path.join(output_path, file_name + '.target')
            with open(src_file, 'w+', encoding='utf-8') as output1, open(tgt_file, 'w+', encoding='utf-8') as output2:
                output1.write('\n'.join(df['source'].tolist()))
                output2.write('\n'.join(df['target'].tolist()))