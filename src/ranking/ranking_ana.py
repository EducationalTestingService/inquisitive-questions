"""
Analyze ranking results
"""

"""
This file is to compute error analysis
"""
import os
import argparse
import pandas as pd
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from fairseq.models.roberta import RobertaModel
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "Analyses for ranking regression"

    # load roberta model
    parser.add_argument('-input_name', type=str, default='question')
    parser.add_argument('-model_dir', type=str, default='/home-nfs/users/output/inquisitive'
                                                        '/ranking')
    parser.add_argument('-checkpoint_name', type=str, default='checkpoint_best.pt')
    return parser.parse_args()


def predict(tokens, roberta):
    tokens_encoded = roberta.encode(tokens)
    # get last_layer_features
    features = roberta.extract_features(tokens_encoded)
    pred = roberta.model.classification_heads['reg_head'](features)
    return pred.item()


if __name__ == '__main__':
    args = get_args()
    args.model_dir = os.path.join(args.model_dir, args.input_name)
    args.data_path = args.model_dir.replace('output', 'data')

    # load model
    roberta = RobertaModel.from_pretrained(args.model_dir, checkpoint_file=args.checkpoint_name,
                                           data_name_or_path=args.data_path)
    roberta.eval()  # disable dropout
    # roberta.cuda()

    # get predictions
    df = pd.read_csv(os.path.join(args.data_path, 'dev.input'), names=['source'], sep='\t\n', encoding='utf-8')
    df['label'] = pd.read_csv(os.path.join(args.data_path, 'dev.label'), names=['label'], sep='\t', encoding='utf-8')
    df['pred'] = df['source'].apply(lambda x: predict(x, roberta))

    # start analyze
    # F.mse_loss(torch.Tensor(df['label'].to_numpy()), torch.Tensor(df['pred'].to_numpy()), reduction='sum')
    # np.sqrt(mean_squared_error(df['label'], df['pred']))
    df['label'] = df['label'] / 4
    print(mean_squared_error(df['label'], df['pred']))
    df.to_csv(os.path.join(args.data_path, 'dev_analysis.csv'), index=False, sep='\t', encoding='utf-8')

    df = pd.read_csv(os.path.join(args.data_path, 'dev_analysis.csv'), sep='\t', encoding='utf-8')
    print(df)
    print(df['label'].corr(df['pred'], method='pearson'))
