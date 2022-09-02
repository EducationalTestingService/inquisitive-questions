# encoding: utf-8

import os
import re
import glob
import argparse
import pandas as pd


"""
This is to add span to all current models
"""

def get_path(input_path):
    file_path = glob.glob(os.path.join(input_path, '*_*'))
    for input_fpath in file_path:
        if os.path.basename(input_fpath) in ['context_source', 'source_rel']:
            continue
        if 'span' not in input_fpath:
            output_fpath = input_fpath.replace('source', 'source_span')
            yield input_fpath, output_fpath

def get_args():
    parser = argparse.ArgumentParser(description='Reformat the data to CoNLL format')
    parser.add_argument('-input_path', type=str,
                        default='/home-nfs/users/data/inquisitive/fairseq/',
                        help='location of the datasets')
    return parser.parse_args()

def add_span(input_fpath, output_fpath):
    """
    not all the first 5 sentences in each article have a question
    """
    df = pd.read_csv(input_fpath, sep='\t', encoding='utf-8')
    df['Span'].fillna(value='NO_SPAN', inplace=True)

    # need to insert after the 2nd [SEP] if there's context
    if 'context_' in input_fpath:
        for index, row in df.iterrows():
            str_context, str_source, str_others = row['source'].split(" [SEP] ", 2)
            df.at[index, 'source'] = ' [SEP] '.join([str_context, str_source, row['Span'], str_others])
    else:
        for index, row in df.iterrows():
            str_source, str_others = row['source'].split(" [SEP] ", 1)
            df.at[index, 'source'] = ' [SEP] '.join([str_source, row['Span'], str_others])
    df.to_csv(output_fpath + '.csv', index=False, sep='\t', encoding='utf-8')

    # output source and target file
    src_file = output_fpath + '.source'
    tgt_file = output_fpath + '.target'
    with open(src_file, 'w+', encoding='utf-8') as output1, open(tgt_file, 'w+', encoding='utf-8') as output2:
        output1.write('\n'.join(df['source'].tolist()))
        output2.write('\n'.join(df['target'].tolist()))

if __name__ == "__main__":
    args = get_args()

    for input_fpath, output_fpath in get_path(args.input_path):
        if not os.path.exists(output_fpath):
            os.makedirs(output_fpath)

        file_name = ['train', 'val', 'test']
        for fname in file_name:
            add_span(os.path.join(input_fpath, fname + '.csv'), os.path.join(output_fpath, fname))
