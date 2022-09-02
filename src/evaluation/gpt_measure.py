"""
find the probability of a sentence using GPT-2?
Note this file only predict sentence[1:] without first token as gpt2 default not to add bos.
"""

import os
import glob
import torch
import argparse
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "question type distributions"

    # data set directory
    parser.add_argument('-log_path', default='/home-nfs/users/output/inquisitive/fairseq/',
                        help='path of the log file')
    parser.add_argument('-log_name', default='test_30maxlen.txt',
                        help='name of the log file')

    return parser.parse_args()

def get_score(tokens, model):
    tokens_tensor = tokenizer.encode(tokens, add_special_tokens=False, return_tensors="pt").to(args.device)
    loss = model(tokens_tensor, labels=tokens_tensor)[0]
    return np.exp(loss.cpu().detach().numpy())

def output_csv(df, cname, model, output_path):
    # filter out 300 questions
    df_index = pd.read_csv('/home-nfs/users/output/inquisitive/fairseq/test_index.csv', names=['index'],
                           sep='\t', encoding='utf-8')
    df_index = df_index.set_index(['index'])
    df = df.iloc[df_index.index[1:]]

    df['score'] = df[cname].apply(lambda x: get_score(x, model))
    df.to_csv(output_path, index=False, sep='\t', encoding='utf-8')
    print('finished')

if __name__ == "__main__":
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get path
    file_path = glob.glob(os.path.join(args.log_path, '*/' + args.log_name))


    # set parameters and model
    cname = 'generation'
    model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    model.to(args.device)
    if 'dev' in args.log_name:
        output_name = 'dev_gen_gpt2.csv'
    else:
        output_name = 'test_gen_gpt2.csv'
        if args.log_name == 'test_nqtype.txt':
            output_name = 'nqtype/' + output_name
        elif args.log_name == 'test_ranked_qtype.txt':
            output_name = 'ranked/' + output_name

    # read data and cal scores
    for fpath in file_path:
        # for key in ['/context_source_span_qtype/', '/context_source_span/', '/context_source/',
        #             # '/source_span_qtype/', '/source_qtype/', '/source_span/', '/source/'
        #             ]:
        for key in ['/human/']:
            if key in fpath:
                df = pd.read_csv(fpath, names=[cname], sep='\t', encoding='utf-8')
                if not os.path.exists(fpath.replace(args.log_name, 'nqtype')):
                    os.makedirs(fpath.replace(args.log_name, 'nqtype'), exist_ok=True)
                output_csv(df, cname, model, fpath.replace(args.log_name, output_name))
