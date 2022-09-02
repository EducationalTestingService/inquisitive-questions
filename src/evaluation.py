# encoding: utf-8

import argparse
import re
import os
import sys
import spacy
import pandas as pd

from nltk.tokenize import word_tokenize
from utils.eval import eval, QGEvalCap

nlp = spacy.load('en')

def get_args():
    parser = argparse.ArgumentParser(description='analyze the eval log with metrics in paper')
    parser.add_argument('-log_path', default= '/home-nfs/users/output/inquisitive/fairseq/human',
                        help='path of the log file')
    parser.add_argument('-log_name', default='test_30maxlen.txt',
                        help='name of the log file')
    parser.add_argument('-train', default='train.csv',
                        help='path of the test file')
    parser.add_argument('-test', default='test.csv',
                        help='path of the test file')
    parser.add_argument('-qsub_script', default='qsub_BERTScore.txt',
                        help='path of the test file')
    parser.add_argument('-first_only', action='store_true',
                        help='only calculate 1st question when 3 questions are generated, '
                             'but it would also affect the output csv file.')
    return parser.parse_args()

class evaluation(object):
    def __init__(self, args):
        self.args = args
        self.test = pd.read_csv(self.args.test, sep='\t', encoding='utf-8')
        self.train = pd.read_csv(self.args.train, sep='\t', encoding='utf-8')
        self.filter_source()
        self.df = self.process_log()

    def filter_source(self):
        source_pos = -1
        if 'qtype' in self.args.log_path:
            source_pos -= 1
        if 'span' in self.args.log_path:
            source_pos -= 1
        for df in [self.train, self.test]:
            df['source'] = df['source'].str.replace(' [SEP] ', '/t', regex=False).str.split('/t').str[source_pos]

    def past_processing(self, result):
        # this is for log processing, abandoned when we move to fairseq
        result = [line.split('\t') for line in result]
        pred = []
        for row, line in enumerate(result):
            if len(line) == 4:
                pred.append({'id': line[0], 'sentence': line[1], 'pred': line[-1]})
            else:
                print(line)
        pred = pd.DataFrame(pred)


    def process_log(self):
        # locate prediction results in log
        result = open(os.path.join(self.args.log_path, self.args.log_name), encoding='utf-8').read().split('\n')

        # # remove [SEP] <NER> tag
        # pt = re.compile(r'\s\[SEP\]\s[A-Z]+\t')
        # result = [re.sub(pt, '\t', line) for line in result]

        # extract index, sentence and pred
        if result[-1] == '':
            result = result[:-1]
        pred = pd.DataFrame({'pred': result})

        # join results with paragraphs
        test = self.test
        # test['Span'].fillna(value='', inplace=True)
        #
        # # when use spans
        # if self.args.aspect == ['SPAN']:
        #     test['source'] = test['source'] + ' [SEP] ' + test['Span']

        # when use the first generated questions
        if self.args.first_only:
            pred = pred[::3]

        # continue processing
        # combined = pd.merge(test, pred, left_on=['source'], right_on=['sentence'])
        assert(test.shape[0] == pred.shape[0])
        combined = pd.concat([test, pred], axis=1)

        combined = combined[['Article_Id', 'Sentence_Id', 'source', 'target', 'Span', 'pred']]

        combined = combined.drop_duplicates()


        # filter out 300 examples
        # get 1st batch index
        new_df_1 = combined.sample(n=50, random_state=1)
        data_df = combined.copy()
        # get unique source and exclude 1st batch
        for i in new_df_1.index:
            if i in data_df.index:
                Article_Id = data_df.at[i, 'Article_Id']
                Sentence_Id = data_df.at[i, 'Sentence_Id']
                data_df = data_df.loc[(data_df['Article_Id'] != Article_Id) | (data_df['Sentence_Id'] != Sentence_Id)]

        data_df_unique = data_df.groupby(['Article_Id', 'Sentence_Id'], group_keys=False).apply(
            lambda data_df: data_df.sample(1, random_state=1))
        data_df_2 = data_df_unique.sample(n=250, random_state=1)
        combined = combined.drop(new_df_1.index)
        combined = combined.drop(data_df_2.index)
        # combined.index.to_series().to_csv('/home-nfs/users/output/inquisitive/fairseq/test_index.csv',
        #                                   sep='\t', encoding='utf-8', index=False)

        # combined[['source', 'target', 'pred']].to_csv(self.args.log_path + '_' + os.path.basename(self.args.test), sep='\t', encoding='utf-8',index=False)
        return combined.reset_index(drop=True)

    def cal_train_n(self, n):
        self.df = self.df.assign(**dict.fromkeys(['ngram_num', 'train_n'], 0))

        ngram_set = set()
        for sent in self.train['target']:
            ngram_set = self.addset(ngram_set, n, sent)

        for row, sent in enumerate(self.df['pred']):
            words_list = [t.text for t in nlp(sent.strip())]
            # sent = re.sub(r'(?=[^a-zA-Z0-9 ])|(?<=[^a-zA-Z0-9 ])', r' ', sent.strip())
            # words_list = re.sub("\s+", " ", sent.strip()).split(' ')
            words_list = word_tokenize(sent.strip())

            for num in range(0, len(words_list) - n + 1):
                ngram = ' '.join(words_list[num:num + n])
                self.df.at[row, 'ngram_num'] += 1
                if ngram in ngram_set:
                    self.df.at[row, 'train_n'] += 1
        return self.df['train_n'].sum() / self.df['ngram_num'].sum()

    @staticmethod
    def addset(ngram_set, n, sent):
        words_list = sent.strip().split(' ')
        for num in range(0, len(words_list) - n + 1):
            ngram = ' '.join(words_list[num:num + n])
            ngram_set.add(ngram)
        return ngram_set

    def cal_article_n(self, n):
        self.df = self.df.assign(**dict.fromkeys(['article_n', 'ngram_num'], 0))

        for row, sent in enumerate(self.df['pred']):
            # process ngram set
            if row == 0:
                # start new ngram_set
                ngram_set = self.addset(set(), n, self.df.at[row, 'source'])
            else:
                if self.df.at[row, 'Article_Id'] != self.df.at[row-1, 'Article_Id']:
                    # start new ngram_set
                    ngram_set = self.addset(set(), n, self.df.at[row, 'source'])
                else:
                    if self.df['Sentence_Id'][row] - self.df['Sentence_Id'][row-1] == 1:
                        ngram_set = self.addset(ngram_set, n, self.df.at[row, 'source'])
                    else:
                        ngram_set = self.addset(ngram_set, n, self.df.at[row, 'source'])

            # begin comparation
            # sent = re.sub(r'(?=[^a-zA-Z0-9 ])|(?<=[^a-zA-Z0-9 ])', r' ', sent.strip())
            # words_list = re.sub("\s+", " ", sent.strip()).split(' ')

            # words_list = [t.text for t in nlp(sent.strip())]
            words_list = word_tokenize(sent.strip())
            for num in range(0, len(words_list) - n + 1):
                ngram = ' '.join(words_list[num:num + n])
                self.df.at[row, 'ngram_num'] += 1
                if ngram in ngram_set:
                    self.df.at[row, 'article_n'] += 1
        return self.df['article_n'].sum() / self.df['ngram_num'].sum()


    def cal_span(self):
        self.df = self.df.assign(**dict.fromkeys(['span_len', 'span_overlap'], 0))

        for row, sent in enumerate(self.df['pred']):
            # words_list = set([t.text for t in nlp(sent.strip())])
            # span_list = list([t.text for t in nlp(self.df.at[row, 'Span'].strip())])

            words_list = set(word_tokenize(sent.strip()))
            span_list = word_tokenize(self.df.at[row, 'Span'].strip())
            # sent = re.sub(r'(?=[^a-zA-Z0-9 ])|(?<=[^a-zA-Z0-9 ])', r' ', sent.strip())
            # words_list = set(re.sub("\s+", " ", sent.strip()).split(' '))
            #
            # span_list = re.sub(r'(?=[^a-zA-Z0-9 ])|(?<=[^a-zA-Z0-9 ])', r' ',
            #                    self.df.at[row, 'Span'].strip())
            # span_list = re.sub("\s+", " ", span_list.strip()).split(' ')

            self.df.at[row, 'span_len'] = len(span_list)
            for word in span_list:
                if word in words_list:
                    self.df.at[row, 'span_overlap'] += 1
        return self.df['span_overlap'].sum() / self.df['span_len'].sum()

    def previous_metrics(self):
        """
        Adapted from Mengxuan's codes
        """
        measurement = ['bleu', 'meteor', 'rouge', 'bert']
        args.model_path = os.path.join(self.args.log_path, 'bert')
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        src_file = os.path.join(self.args.log_path, 'bert/source.txt')
        tgt_file = os.path.join(self.args.log_path, 'bert/target.txt')
        out_file = os.path.join(self.args.log_path, 'bert/pred.txt')

        with open(src_file, 'w+', encoding='utf-8') as output1, \
                open(tgt_file, 'w+', encoding='utf-8') as output2, open(out_file, 'w+', encoding='utf-8') as output3:
            output1.write('\n'.join(self.df['source'].tolist()))
            output2.write('\n'.join(self.df['target'].tolist()))
            output3.write('\n'.join(self.df['pred'].tolist()))

        if 'bert' in measurement:
            cmd = '#!/bin/bash\n\n' \
                  '#SBATCH -p speech-gpu\n' \
                  '#SBATCH -J bert_score\n' \
                  '#SBATCH -C 11g\n' \
                  'source activate nlp\n' \
                  'bert-score -r {1} -c {2} --lang en --rescale_with_baseline'.format(
                os.path.abspath(os.path.dirname(args.qsub_script)), tgt_file, out_file)
            with open(args.qsub_script, 'w') as output:
                output.write(cmd)
                print('*' * 12 + '\ninstructions\n' + '*' * 12)
                print(
                    '{} is generated. You need to qsub it to a gpu node to calculate BERTScore. '
                    'A log file containing BERTScore results will be created by the grid engine in the same directory.\n'.format(
                        os.path.abspath(args.qsub_script)))
                measurement.pop()

        if measurement:
            # gts is gold questions, res is predicted questions
            gts, res = eval(out_file, src_file, tgt_file)
            QGEval = QGEvalCap(gts, res, out_file, tgt_file, measurement)
            QGEval.evaluate()

if __name__ == '__main__':
    args = get_args()
    data_path = args.log_path.replace('output/inquisitive', 'data/inquisitive')
    # data_path = '/home-nfs/users/data/inquisitive/fairseq/source_rel'
    args.train = os.path.join(data_path, 'train.csv')
    args.test = os.path.join(data_path, 'test.csv')
    args.qsub_script = os.path.join(args.log_path,'qsub_BERTScore.txt')
    eval_ = evaluation(args)
    eval_.previous_metrics()
    for n in [2, 3, 4]:
        print('Train-{:d}: {:.3f}'.format(n, eval_.cal_train_n(n)))
    for n in [1, 2, 3]:
        print('Article-{:d}: {:.3f}'.format(n, eval_.cal_article_n(n)))
    print('Span: {:.3f}'.format(eval_.cal_span()))


