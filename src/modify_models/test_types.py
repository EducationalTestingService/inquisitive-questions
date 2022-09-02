"""
compute test accuracy of classifier
"""
import os
import glob
import math
import argparse
import pandas as pd
from fairseq.models.roberta import RobertaModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "classfication with roberta"

    # load roberta model
    parser.add_argument('-model_type', type=str, default='orig', choices=['orig_enhanced', 'ner', 'orig'])
    parser.add_argument('-model_dir', type=str, default='/home-nfs/users/output/inquisitive/roberta/')
    parser.add_argument('-checkpoint_name', type=str, default='checkpoint_best.pt')
    parser.add_argument('-data_path', type=str, default='/home-nfs/users/data/inquisitive/discriminator/')

    parser.add_argument('-input_path', type=str, default='/home-nfs/users/data/inquisitive/discriminator/orig')
    return parser.parse_args()


def get_score(tokens, label_fn, roberta):
    tokens_encoded = roberta.encode(tokens)
    return label_fn(roberta.predict('rescorer_head', tokens_encoded).argmax().item())


if __name__ == "__main__":
    args = get_args()

    # load model
    args.model_dir = os.path.join(args.model_dir, args.model_type)
    args.data_path = os.path.join(args.data_path, args.model_type)
    roberta = RobertaModel.from_pretrained(args.model_dir, checkpoint_file=args.checkpoint_name,
                                           data_name_or_path=args.data_path)
    roberta.eval()  # disable dropout
    # get func
    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )

    # read predictions for test data and cal scores
    df = pd.read_csv(os.path.join(args.input_path, 'test.input'), sep='\t', encoding='utf-8', names=['source'])
    df['label'] = pd.read_csv(os.path.join(args.input_path, 'test.label'),
                              header=None, sep='\n', encoding='utf-8').squeeze()
    df['pred'] = df['source'].apply(lambda x: get_score(x, label_fn, roberta))
    df.pred = pd.to_numeric(df.pred)
    print(df.loc[df.pred == df.label].shape[0] / 3000)
    print('finished')

    # read predictions for dev data and cal scores
    df = pd.read_csv(os.path.join(args.input_path, 'dev.input'), sep='\t', encoding='utf-8', names=['source'])
    df['label'] = pd.read_csv(os.path.join(args.input_path, 'dev.label'),
                              header=None, sep='\n', encoding='utf-8').squeeze()
    df['pred'] = df['source'].apply(lambda x: get_score(x, label_fn, roberta))
    df.pred = pd.to_numeric(df.pred)
    print(df.loc[df.pred == df.label].shape[0] / 3000)
    print('finished')

