"""
This file is to generate all input with all qtypes
"""

"""
This file is to collect the question type predictions, and generate output for model
"""

import os
import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "Generate input with all question types"
    parser.add_argument('-input_path', type=str,
                        default='/home-nfs/users/data/inquisitive/fairseq',
                        help='path for input with control codes')
    parser.add_argument('-output_path', type=str,
                        default='/home-nfs/users/data/inquisitive/qtype_allgen',
                        help='path for output with control codes')
    return parser.parse_args()


def change_qtype(input_path, output_path_base):
    for file_name in ['val', 'test']:
        df = pd.read_csv(os.path.join(input_path, file_name + '.csv'), sep='\t', encoding='utf-8')

        # separate qtype
        df['source_ele'] = df['source'].copy()
        for index, row in df.iterrows():
            sep_ele = row['source'].split(" [SEP] ")[:-1]
            df.at[index, 'source_ele'] = ' [SEP] '.join(sep_ele)

        for qtype in ['Explanation', 'Elaboration', 'Definition', 'Background', 'Instantiation', 'Forward', 'Other']:
            output_path = os.path.join(output_path_base, qtype)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            df['source'] = df['source_ele'] + ' [SEP] ' + qtype
            df.to_csv(os.path.join(output_path, file_name + '.csv'), index=False, sep='\t', encoding='utf-8')
            src_file = os.path.join(output_path, file_name + '.source')
            tgt_file = os.path.join(output_path, file_name + '.target')
            with open(src_file, 'w+', encoding='utf-8') as output1, open(tgt_file, 'w+', encoding='utf-8') as output2:
                output1.write('\n'.join(df['source'].tolist()))
                output2.write('\n'.join(df['target'].tolist()))


if __name__ == '__main__':
    args = get_args()

    # go through all path names
    path_name = ['context_source_span_qtype', 'context_source_qtype', 'source_span_qtype', 'source_qtype']

    # create generations
    for fpath in path_name:
        change_qtype(os.path.join(args.input_path, fpath), os.path.join(args.output_path, fpath))