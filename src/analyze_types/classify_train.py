"""
This file is to classify the train/dev set with Roberta model (inquisitive v.s., informative classifier),
sort them by the predicted class probability of inquisitive

need to separate:
1. seen and unseen
2. train and dev
3. INQUISITIVE and SQuAD

part of codes are copied from data/roberta_data_processing.py in order to recover the process of sampling
"""
import os
import warnings
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

    # INQUISITIVE
    parser.add_argument('-out', type=str, default='/home-nfs/users/data/inquisitive/discriminator',
                        help='training path')

    parser.add_argument('-output_path', type=str,
                        default='/home-nfs/users/data/inquisitive/manual/trainset_score/',
                        help='training set path')
    return parser.parse_args()

def clean_unnecessary_spaces(out_string):
    """
    Some of the data have spaces before punctuation marks that we need to remove.
    This is used for BART/Roberta
    modified from:
    https://towardsdatascience.com/bart-for-paraphrasing-with-simple-transformers-7c9ea3dfdd8c
    """
    if not isinstance(out_string, str):
        warnings.warn(f">>> {out_string} <<< is not a string.")
        out_string = str(out_string)
    out_string = (
        out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" — ", "—")
            .replace(" - ", "-")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
            .replace(" ’ ", "’")
            .replace("“ ", "“")
            .replace(" ”", "”")
            .replace(" ( ", " (")
            .replace(" ) ", ") ")
    )
    return out_string

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
    pos_df_seen = pos_df.sample(n=num, random_state=1)
    neg_df_seen = neg_df.sample(n=num, random_state=1)
    print(pos_df_seen.shape, neg_df_seen.shape)

    # append and shuffle
    new_df = pos_df_seen.append(neg_df_seen)
    new_df = new_df.sample(frac=1, random_state=1).reset_index(drop=True)
    print(new_df.shape)

    # output files
    compare_orig = pd.read_csv(os.path.join(args.out, 'orig/' + fname + '.input'),
                               names=['orig'], sep='\t', encoding='utf-8')
    diff = new_df['orig'].compare(compare_orig.squeeze())
    assert(diff.shape[0] == 0)
    cname = 'orig'
    pos_df_unseen = pos_df.drop(pos_df_seen.index)
    neg_df_unseen = neg_df.drop(neg_df_seen.index).sample(n=1500, random_state=1)
    output_csv(pos_df_seen, cname, roberta, os.path.join(args.output_path, 'inquisitive_seen_' + fname + '.csv'))
    output_csv(pos_df_unseen, cname, roberta, os.path.join(args.output_path, 'inquisitive_unseen_' + fname + '.csv'))
    output_csv(neg_df_seen, cname, roberta, os.path.join(args.output_path, 'squad_seen_' + fname + '.csv'))
    output_csv(neg_df_unseen, cname, roberta, os.path.join(args.output_path, 'squad_unseen_' + fname + '.csv'))

if __name__ == "__main__":
    args = get_args()
    args.output_path = os.path.join(args.output_path, args.model_type)

    # load model
    args.model_dir = os.path.join(args.model_dir, args.model_type)
    roberta = RobertaModel.from_pretrained(args.model_dir, checkpoint_file=args.checkpoint_name,
                                           data_name_or_path=args.data_path)
    roberta.eval()  # disable dropout
    roberta.cuda()

    # sample file
    sample_num = [8000, 1500]
    fname_list = ['train', 'dev']
    for i, fname in enumerate(fname_list):
        pos_1 = os.path.join(args.out, '_'.join(['inquisitive', 'ner', fname]) + '.txt')
        pos_2 = os.path.join(args.out, '_'.join(['inquisitive', 'orig', fname]) + '.txt')
        neg_1 = os.path.join(args.out, '_'.join(['squad', 'ner', fname]) + '.txt')
        neg_2 = os.path.join(args.out, '_'.join(['squad', 'orig', fname]) + '.txt')
        sample_train(pos_1, pos_2, neg_1, neg_2, sample_num[i], fname)


