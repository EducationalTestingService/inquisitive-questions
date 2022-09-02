# encoding: utf-8

import argparse
import pandas as pd

"""
This file is to generate 50 examples randomly extracted from training set (first batch).

include 50 random source sentences, context, and gold questions for annotation

We also add 2nd and 3rd batch code here.
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
        self.sample_()

    def sample_(self):
        first_batch = self.train.sample(n=50, random_state=1)
        first_batch = first_batch[['prev_sent', 'source', 'Span', 'target']]
        # first_batch.to_csv('/home-nfs/users/data/inquisitive/manual/50_train_samples.csv', sep='\t', encoding='utf-8',index=False)

        self.train = self.train.drop(first_batch.index)
        second_batch = self.train.sample(n=750, random_state=1)
        second_batch = second_batch[['prev_sent', 'source', 'Span', 'target']]
        # second_batch[:250].to_csv('/home-nfs/users/data/inquisitive/manual/user1_train_samples.csv',
        #                     sep='\t', encoding='utf-8',index=False)
        # second_batch[250:500].to_csv('/home-nfs/users/data/inquisitive/manual/user3_train_samples.csv',
        #                     sep='\t', encoding='utf-8', index=False)
        # second_batch[500:].to_csv('/home-nfs/users/data/inquisitive/manual/user2_train_samples.csv',
        #                     sep='\t', encoding='utf-8',index=False)

        self.train = self.train.drop(second_batch.index)
        third_batch = self.train.sample(n=750, random_state=1)
        third_batch = third_batch[['prev_sent', 'source', 'Span', 'target']]
        third_batch[:250].to_csv('/home-nfs/users/data/inquisitive/manual/user1_batch2.csv',
                            sep='\t', encoding='utf-8',index=False)
        third_batch[250:500].to_csv('/home-nfs/users/data/inquisitive/manual/user3_batch2.csv',
                            sep='\t', encoding='utf-8', index=False)
        third_batch[500:].to_csv('/home-nfs/users/data/inquisitive/manual/user2_batch2.csv',
                            sep='\t', encoding='utf-8',index=False)


if __name__ == '__main__':
    args = get_args()
    eval_ = Combination(args)
