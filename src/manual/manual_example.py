# encoding: utf-8

import argparse
import os
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description='sample examples for manual annotations')
    parser.add_argument('-log_path', default='/home-nfs/users/output/inquisitive/fairseq/',
                        help='path of the log file')
    parser.add_argument('-log_name', default='dev_30maxlen.txt',
                        help='name of the log file')
    return parser.parse_args()


class Combination(object):
    def __init__(self, args):
        self.args = args
        self.test = pd.read_csv('/home-nfs/users/data/inquisitive/fairseq/context_source_rel_ent/val.csv',
                                sep='\t', encoding='utf-8')
        self.source_test = pd.read_csv('/home-nfs/users/data/inquisitive/fairseq/source/val.csv',
                                       sep='\t', encoding='utf-8')
        self.test = self.process_log(self.test, ['source_rel', 'source_rel_ent', 'source_rel_type',
                                                 'context_source_rel', 'context_source_rel_ent',
                                                 'context_source_rel_type'])
        self.source_test = self.process_log(self.source_test, ['source', 'source_span', 'source_comet',
                                                               'context_source',
                                                               'context_source_span', 'context_source_comet'])
        self.sample_()

    def process_log(self, df, log_list):
        test = df
        # locate prediction results in log
        for log in log_list:
            log_path = os.path.join(self.args.log_path, log)
            result = open(os.path.join(log_path, self.args.log_name), encoding='utf-8').read().split('\n')

            # extract index, sentence and pred
            if result[-1] == '':
                result = result[:-1]
            if log == 'source':
                log = 'source_only'
                log_list[0] = log
            pred = pd.DataFrame({log: result})

            # continue processing
            assert(test.shape[0] == pred.shape[0])
            test = pd.concat([test, pred], axis=1)
        # don't keep prev_sent as there might be difference when prev_sent is null.
        test = test[['Article_Id', 'Sentence_Id', 'source', 'target', 'Span'] + log_list]
        return test.reset_index(drop=True).fillna('')

    def sample_(self):
        self.test = self.test.sample(n=50, random_state=1).drop(columns=['source'])
        # uncomment below when we don't want source_test but only need source text.
        # self.source_test = self.source_test[['Article_Id', 'Sentence_Id', 'target', 'Span', 'source']]
        df = pd.merge(self.test, self.source_test, on=['Article_Id', 'Sentence_Id', 'target', 'Span'])

        # sort column values order and save
        df = df[['Article_Id','Sentence_Id','source','target','Span',
                 'source_only','source_span','source_comet',
                 'source_rel', 'source_rel_ent', 'source_rel_type',
                 'context_source','context_source_span','context_source_comet',
                 'context_source_rel','context_source_rel_ent','context_source_rel_type']]
        df.to_csv('/home-nfs/users/data/inquisitive/manual/new_dev.csv',
                  sep='\t', encoding='utf-8',index=False)


if __name__ == '__main__':
    args = get_args()
    eval_ = Combination(args)
