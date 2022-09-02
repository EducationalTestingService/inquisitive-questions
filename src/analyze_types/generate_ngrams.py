"""
This file is to run inquisitive v.s., informative classifier on prefixes (unigram, bigram, trigram)
"""

import os
import glob
import argparse
import pandas as pd
from fairseq.models.roberta import RobertaModel

from utils import output_csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "classfication with roberta"

    # load roberta model
    parser.add_argument('-model_type', type=str, default='orig', choices=['ner_orig', 'ner', 'orig'])
    parser.add_argument('-model_dir', type=str, default='/home-nfs/users/output/inquisitive/roberta/')
    parser.add_argument('-checkpoint_name', type=str, default='checkpoint_best.pt')
    parser.add_argument('-data_path', type=str, default='/home-nfs/users/data/inquisitive/discriminator/orig')

    # data set directory
    parser.add_argument('-gold', type=str, default='/home-nfs/users/data/inquisitive/context/')
    parser.add_argument('-squad', type=str, default='/home-nfs/users/data/squad/')
    parser.add_argument('-pred', type=str, default='/home-nfs/users/data/inquisitive/ngrams/')

    # model generation directory
    parser.add_argument('-out', type=str, default='/home-nfs/users/data/inquisitive/ngrams/scores')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # load model
    args.model_dir = os.path.join(args.model_dir, args.model_type)
    args.out = os.path.join(args.out, args.model_type)
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    roberta = RobertaModel.from_pretrained(args.model_dir, checkpoint_file=args.checkpoint_name,
                                           data_name_or_path=args.data_path)
    roberta.eval()  # disable dropout

    # # generate unigram, bigram, trigram for gold data (train and dev)
    ngram_list = ['unigram', 'bigrams']
    fname_list = ['train', 'dev']
    cname = 'ngram'
    # inquisitive gold
    for fname in fname_list:
        df_base = pd.read_csv(os.path.join(args.gold, fname + '.csv'), sep='\t', encoding='utf-8')
        for idx, ngram in enumerate(ngram_list, 1):
            key_values = df_base['target'].str.split(' ').apply(lambda x: ' '.join(x[:idx])).value_counts()
            key_values = key_values.to_frame().reset_index().rename(columns={'target': 'numbers', 'index': 'ngram'})
            key_values['percentage'] = (key_values['numbers'] / key_values['numbers'].sum()) * 100
            output_csv(key_values, cname, roberta, os.path.join(args.out, '_'.join([fname, ngram, 'inq_std.csv'])))

    # squad gold
    fname_list = ['dev']
    for fname in fname_list:
        df_base = pd.read_csv(os.path.join(args.squad, fname + '.txt'), sep='\t', encoding='utf-8')
        for idx, ngram in enumerate(ngram_list, 1):
            # key_values = df_base['target'].str.split(' ').apply(lambda x: ' '.join(x[:idx]).lower()).value_counts()
            key_values = df_base['target'].str.split(' ').apply(lambda x: ' '.join(x[:idx])).value_counts()
            key_values = key_values.to_frame().reset_index().rename(columns={'target': 'numbers', 'index': 'ngram'})
            key_values['percentage'] = (key_values['numbers'] / key_values['numbers'].sum()) * 100
            output_csv(key_values, cname, roberta, os.path.join(args.out, '_'.join([fname, ngram, 'squad_std.csv'])))

    # # read predictions for dev data and cal scores
    # for ngram in ngram_list:
    #     l = [pd.read_csv(filename, names=['ngram', 'numbers', 'percentage'], sep='\t', encoding='utf-8')
    #          for filename in glob.glob(os.path.join(args.pred, ngram + '/*.txt'))]
    #     combined = pd.concat(l, axis=0).groupby(['ngram']).sum().reset_index()
    #     combined['percentage'] = (combined['numbers'] / combined['numbers'].sum()) * 100
    #     output_csv(combined, cname, roberta, os.path.join(args.out, '_'.join(['dev', ngram, 'pred.csv'])))


