# encoding: utf-8

import torch
import random
import argparse
import pandas as pd



"""
This is to split the dataset into train, dev and test set in csv format
"""
def get_args():
    parser = argparse.ArgumentParser(description='randomly split dataset to train, dev and test')
    parser.add_argument('-input_file', type=str,
                        default='/home/nlp-text/static/corpora/nonets/ACG/INQUISITIVE/questions.txt',
                        help='location of the datasets')
    parser.add_argument('-output_path', type=str, default='/home/nlp-text/dynamic/users/data/INQUISITIVE/',
                        help='location of the datasets')
    parser.add_argument('-fix_seed', action='store_true', help='fix random seed to get reproducible results')
    parser.add_argument('-seed', type=int, default=0, help='random seed choices for reproducibility')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # set the random seed manually for reproducibility
    if args.fix_seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # df = pd.read_csv(args.input_file, sep='\t', lineterminator='\n',
    #                  encoding='utf-8', error_bad_lines=False)
    results = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        column_name = f.readline().strip().split('\t')
        for line in f:
            read_line = line.strip().split('\t')
            if len(read_line) > 7:
                print(read_line)
            results.append(read_line)
    df = pd.DataFrame(results, columns=column_name).drop_duplicates()
    df[['Article_Id', 'Sentence_Id', 'Span_Start_Position', 'Span_End_Position']] = \
        df[['Article_Id', 'Sentence_Id', 'Span_Start_Position', 'Span_End_Position']].apply(pd.to_numeric)
    df = df.rename(columns={"Sentence": "source"})
    df = df.rename(columns={"Question": "target"})

    # get dev data
    dev_data = df.loc[(df.Article_Id <= 100) | ((df.Article_Id >= 1051) & (df.Article_Id <= 1100))]
    dev_data.fillna(value='', inplace=True)
    dev_data.to_csv(args.output_path + 'dev.csv', index=False, sep='\t', encoding='utf-8')

    # get test data
    test_data = df.loc[((df.Article_Id >= 101) & (df.Article_Id <= 150)) |
                       ((df.Article_Id >= 501) & (df.Article_Id <= 550)) |
                       ((df.Article_Id >= 1101) & (df.Article_Id <= 1150))]
    test_data.fillna(value='', inplace=True)
    test_data.to_csv(args.output_path + 'test.csv', index=False, sep='\t', encoding='utf-8')

    # get train data
    train_data = df.loc[((df.Article_Id >= 151) & (df.Article_Id <= 500)) |
                       ((df.Article_Id >= 551) & (df.Article_Id <= 1050)) |
                       ((df.Article_Id >= 1151) & (df.Article_Id <= 1500))]
    train_data.fillna(value='', inplace=True)
    train_data.to_csv(args.output_path + 'train.csv', index=False, sep='\t', encoding='utf-8')

    print(train_data.shape, dev_data.shape, test_data.shape, df.shape)




