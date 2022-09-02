"""
This file is to separate train and dev for SQuAD and INQUISITIVE,
also those used for classifier training and those are not for scoring
"""


import os
import argparse
import re
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description='generate files')
    parser.add_argument('-data', default='squad', choices=['squad', 'inquisitive'],
                        help='squad/inquisitive')
    parser.add_argument('-used_path', default='/home/nlp-text/dynamic/users/data/INQUISITIVE/discriminator',
                        help='path of the output file')
    parser.add_argument('-output_path', default='/home/nlp-text/dynamic/users/data/INQUISITIVE/manual',
                        help='path of the output file')
    return parser.parse_args()


def proc_file(input_file, output_file, label, type):
    df = pd.read_csv(input_file, sep='\t', encoding='utf-8')
    df = df.assign(**dict.fromkeys(['label'], label))
    if type == 'ner':
        models_dir = '/home/nlp-text/dynamic/users/stanza_resources/'
        nlp = stanza.Pipeline(dir=models_dir, processors='tokenize,ner')
        for index, row in df.iterrows():
            doc = nlp(row['target'])
            for parsed_sentence in doc.sentences:
                for ent in parsed_sentence.entities[::-1]:
                    df.at[index, 'target'] = df.at[index, 'target'][:ent.start_char] + ent.type \
                                             + df.at[index, 'target'][ent.end_char:]
    df[['label', 'target']].to_csv(output_file, index=False, sep='\t', encoding='utf-8')

def sample_train(input_file1, input_file2, num, output_file):
    df1 = pd.read_csv(input_file1, sep='\t', encoding='utf-8')
    df2 = pd.read_csv(input_file2, sep='\t', encoding='utf-8')
    new_df = df1.sample(n=num, random_state=1)
    new_df = new_df.append(df2.sample(n=num, random_state=1))
    new_df = new_df.sample(frac=1, random_state=1).reset_index(drop=True)
    new_df.to_csv(output_file, header=False, index=False, sep='\t', encoding='utf-8')


if __name__ == "__main__":
    args = get_args()

    # path dictionary and file name
    data_path_dict = {'squad': ['/home/nlp-text/dynamic/user4/AQG/bart/data', '.txt'],
                      'inquisitive': ['/home/nlp-text/dynamic/users/data/INQUISITIVE/context', '.csv']}
    fname_list = ['train', 'dev', 'test']
    label = 1

    # process file
    # data_path = data_path_dict[args.data]
    # for fname in fname_list:
    #     if args.data == 'squad':
    #         input_file = os.path.join(data_path[0], 'combined' + fname + data_path[1])
    #         label = 0
    #     else:
    #         input_file = os.path.join(data_path[0], fname + data_path[1])
    #     output_file = os.path.join(args.out, '_'.join([args.data, args.type, fname]) + '.txt')
    #     proc_file(input_file, output_file, label, args.type)

    # sample file
    sample_num = [15000, 1500, 1500]
    for i, fname in enumerate(fname_list):
        input_file1 = os.path.join(args.out, '_'.join(['inquisitive', 'ner', fname]) + '.txt')
        input_file2 = os.path.join(args.out, '_'.join(['squad', 'ner', fname]) + '.txt')
        output_file = os.path.join(args.out, fname + '_combined_ner.txt')
        sample_train(input_file1, input_file2, sample_num[i], output_file)