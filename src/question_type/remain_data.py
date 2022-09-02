"""
This file is to find remain data not annotated
"""
# encoding: utf-8

import argparse
import pandas as pd

"""
This file is to generate 50 examples randomly extracted from training set.

include 50 random source sentences, context, and gold questions for annotation
"""

def get_args():
    parser = argparse.ArgumentParser(description='analyze the eval log with metrics in paper')
    parser.add_argument('-file_path', default='/home-nfs/users/data/inquisitive/context/train.csv',
                        help='path of the log file')
    return parser.parse_args()


class Combination(object):
    def __init__(self, args):
        self.args = args
        self.train = pd.read_csv(args.file_path, sep='\t', encoding='utf-8')
        self.first_idx, self.second_idx, self.third_idx, self.fourth_batch = self.sample_()

    def compare_(self, df, df_path):
        df_comp = pd.read_csv(df_path, sep='\t', encoding='utf-8')
        print(df.reset_index(drop=True).compare(df_comp))

    def sample_(self):
        first_batch = self.train.sample(n=50, random_state=1)
        first_batch = first_batch[['prev_sent', 'source', 'Span', 'target']]
        self.compare_(first_batch, '/home-nfs/users/data/inquisitive/manual/50_train_samples.csv')

        self.train = self.train.drop(first_batch.index)
        second_batch = self.train.sample(n=750, random_state=1)
        second_batch = second_batch[['prev_sent', 'source', 'Span', 'target']]

        self.train = self.train.drop(second_batch.index)
        third_batch = self.train.sample(n=750, random_state=1)
        third_batch = third_batch[['prev_sent', 'source', 'Span', 'target']]

        self.compare_(second_batch[:250], '/home-nfs/users/data/inquisitive/manual/user1_train_samples.csv')
        self.compare_(second_batch[250:500], '/home-nfs/users/data/inquisitive/manual/user3_train_samples.csv')
        self.compare_(second_batch[500:], '/home-nfs/users/data/inquisitive/manual/user2_train_samples.csv')
        self.compare_(third_batch[:250], '/home-nfs/users/data/inquisitive/manual/user1_batch2.csv')
        self.compare_(third_batch[250:500], '/home-nfs/users/data/inquisitive/manual/user3_batch2.csv')
        self.compare_(third_batch[500:], '/home-nfs/users/data/inquisitive/manual/user2_batch2.csv')

        return first_batch.index, second_batch.index, third_batch.index, self.train.drop(third_batch.index)

if __name__ == '__main__':
    args = get_args()
    eval_ = Combination(args)
    eval_.fourth_batch.to_csv('/home-nfs/users/data/inquisitive/annotations/qtype_pred/origin/train.csv',
                             sep='\t', encoding='utf-8', index=False)
    eval_.first_idx.to_series().to_csv('/home-nfs/users/data/inquisitive/annotations/qtype_pred/origin/50_index.csv',
                                       sep='\t', encoding='utf-8', index=False)
    eval_.second_idx.to_series().to_csv('/home-nfs/users/data/inquisitive/annotations/qtype_pred/origin/750_index.csv',
                                        sep='\t', encoding='utf-8', index=False)
    eval_.third_idx.to_series().to_csv('/home-nfs/users/data/inquisitive/annotations/qtype_pred/origin/750_2_index.csv',
                                       sep='\t', encoding='utf-8', index=False)
    eval_.fourth_batch.index.to_series().to_csv('/home-nfs/users/data/inquisitive/annotations/qtype_pred/origin/pred_index.csv',
                                               sep='\t', encoding='utf-8', index=False)
