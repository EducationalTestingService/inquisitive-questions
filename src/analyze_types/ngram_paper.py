"""
This file is to calculate scores for ngrams from inquisitive paper using inquisitive v.s., informative classifier
"""
import os
import argparse
import pandas as pd
from fairseq.models.roberta import RobertaModel

from utils import output_csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "classfication with roberta"

    # load roberta model
    parser.add_argument('-model_type', type=str, default='ner', choices=['ner_orig', 'ner', 'orig'])
    parser.add_argument('-model_dir', type=str, default='/home-nfs/users/output/inquisitive/roberta/')
    parser.add_argument('-checkpoint_name', type=str, default='checkpoint_best.pt')
    parser.add_argument('-data_path', type=str, default='/home-nfs/users/data/inquisitive/discriminator/orig')

    parser.add_argument('-output_path', type=str,
                        default='/home-nfs/users/data/inquisitive/ngrams',
                        help='training set path')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    d = ['what is why is why did what are why was why are how did what was what does why were who is why would why does who are how does',
         'what is what was how many when did in what what did when was who was what does what are what type how much what year where did what do',
         'did he what was what did are there how did did they when did what is what happened what else did she where did what other what was what were',
         'what is what did how many who is what was what does who was where did what are where was when did where did what do who did what has'
         ]
    final = set()
    for x in d:
        token = x.split(" ")
        for i in range(0, len(token), 2):
            final.add(' '.join(token[i:i+2]))

    df = pd.DataFrame(list(final), columns=['ngrams'])

    # load model
    args.model_dir = os.path.join(args.model_dir, args.model_type)
    roberta = RobertaModel.from_pretrained(args.model_dir, checkpoint_file=args.checkpoint_name,
                                           data_name_or_path=args.data_path)
    roberta.eval()  # disable dropout
    roberta.cuda()

    output_csv(df, 'ngrams', roberta, os.path.join(args.output_path, args.model_type + '_paper_ngram.csv'))