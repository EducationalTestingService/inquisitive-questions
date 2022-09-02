# encoding: utf-8

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def get_args():
    parser = argparse.ArgumentParser(description='leading bigrams')
    parser.add_argument('-log_path', default='/home-nfs/users/output/inquisitive/fairseq/',
                        help='path of the log file')
    parser.add_argument('-log_name', default='dev_30maxlen.txt',
                        help='name of the log file')
    parser.add_argument('-ngram', default=1, help='unigram or bigram')
    return parser.parse_args()


class Bigrams(object):
    def __init__(self, args):
        self.args = args
        self.test = pd.read_csv('/home-nfs/users/data/inquisitive/fairseq/context_source_rel_ent/val.csv',
                                sep='\t', encoding='utf-8')
        self.source_test = pd.read_csv('/home-nfs/users/data/inquisitive/fairseq/source/val.csv',
                                       sep='\t', encoding='utf-8')
        self.test = self.process_log(self.test, ['source_rel', 'source_rel_ent', 'source_rel_type',
                                                 'context_source_rel', 'context_source_rel_ent',
                                                 'context_source_rel_type',
                                                 'source_span_rel', 'source_span_rel_ent', 'source_span_rel_type',
                                                 'context_source_span_rel', 'context_source_span_rel_ent',
                                                 'context_source_span_rel_type'])
        self.source_test = self.process_log(self.source_test, ['source', 'source_span', 'source_comet',
                                                               'source_span_comet', 'context_source',
                                                               'context_source_span', 'context_source_comet',
                                                               'context_source_span_comet'])
        self.sample_()

    def process_log(self, df, log_list):
        test = df

        # get no_span_idx
        no_span_idx = df.index[df['Span'] == 'NO_SPAN'].tolist()
        if not no_span_idx:
            no_span_idx = df.index[df['Span'].isnull()].tolist()
        test = test.drop(no_span_idx).reset_index(drop=True)

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

            # for doing val
            if 'comet' not in log:
                pred = pred.drop(no_span_idx).reset_index(drop=True)

            # continue processing
            assert(test.shape[0] == pred.shape[0])
            test = pd.concat([test, pred], axis=1)
        # don't keep prev_sent as there might be difference when prev_sent is null.
        test = test[['Article_Id', 'Sentence_Id', 'source', 'target', 'Span'] + log_list]
        return test.reset_index(drop=True).fillna('')

    def plot_hist(self, df, direct_name, key):
        """
        modified from https://mode.com/example-gallery/python_histogram/
        """
        nbins = self.args.ngram * 5
        ax = df[: nbins].plot.bar(y='percentage', color='#86bf91', rot=0)
        plt.title(key)
        plt.ylabel = 'percentage (%)'
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.savefig('/home-nfs/users/data/inquisitive/ngrams/' + direct_name + key + '.png')

    def sample_(self):
        self.test = self.test.drop(columns=['source'])
        df = pd.merge(self.test, self.source_test, on=['Article_Id', 'Sentence_Id', 'target', 'Span'])

        # sort column values order and save
        df = df[['source_only','source_span',
                 'source_comet', 'source_span_comet',
                 'source_rel', 'source_rel_ent', 'source_rel_type',
                 'source_span_rel', 'source_span_rel_ent', 'source_span_rel_type',
                 'context_source','context_source_span',
                 'context_source_comet', 'context_source_span_comet',
                 'context_source_rel','context_source_rel_ent','context_source_rel_type',
                 'context_source_span_rel','context_source_span_rel_ent','context_source_span_rel_type']]

        if self.args.ngram == 2:
            direct_name = 'bigrams/'
        elif self.args.ngram == 1:
            direct_name = 'unigram/'
        else:
            direct_name = 'trigrams/'

        for key in df.columns.values:
            key_values = df[key].str.split(' ').apply(lambda x: ' '.join(x[:self.args.ngram]).lower()).value_counts()
            key_values = key_values.to_frame().rename(columns = {key:'numbers'})
            key_values['percentage'] = (key_values['numbers'] / key_values['numbers'].sum()) * 100

            # output to csv files
            key_values.to_csv('/home-nfs/users/data/inquisitive/ngrams/' + direct_name + key + '.txt',
                              sep='\t', encoding='utf-8', header=False)

            # plot histogram for first 10/5 bigrams
            self.plot_hist(key_values, direct_name, key)


if __name__ == '__main__':
    args = get_args()
    eval_ = Bigrams(args)




