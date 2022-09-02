# encoding: utf-8

import os
import re
import argparse
import pandas as pd



"""
This is to add one column of previous context
"""

def get_args():
    parser = argparse.ArgumentParser(description='add one column of previous context')
    parser.add_argument('-input_path', type=str,
                        default='/home/nlp-text/dynamic/users/data/INQUISITIVE/',
                        help='location of the datasets')
    parser.add_argument('-article_path', type=str,
                        default='/home/nlp-text/static/corpora/nonets/ACG/INQUISITIVE/article/',
                        help='location of the datasets')
    parser.add_argument('-output_path', type=str, default='/home/nlp-text/dynamic/users/data/INQUISITIVE/context/',
                        help='location of the new datasets')
    return parser.parse_args()

def get_prev_text(article_path, df, row):
    """
    Note that the article is not preprocessed...
    """
    article_id = str(df.at[row, 'Article_Id'])
    sent_id = df.at[row, 'Sentence_Id']
    article_id = '0' * (4 - len(article_id)) + article_id
    article_path = os.path.join(article_path, article_id + '.txt')
    assert(os.path.exists(article_path))
    prev_sent = ''
    with open(article_path, 'r', encoding='utf-8') as f:
        for sent_num in range(1, sent_id):
            prev_sent += ' ' + next(f).split(" ", 1)[1].strip()
    # preprocess data
    prev_sent = re.sub(r'(?=[^a-zA-Z0-9 ])|(?<=[^a-zA-Z0-9 ])', r' ', prev_sent.strip())
    return re.sub("\s+", " ", prev_sent.strip())

def add_context(input_fpath, article_path, output_fpath):
    """
    not all the first 5 sentences in each article have a question
    """
    df = pd.read_csv(input_fpath, sep='\t', encoding='utf-8')
    df = df.assign(**dict.fromkeys(['prev_sent'], ' '))
    for row, sent in enumerate(df['source']):
        if row != 0:
            if df.at[row, 'Article_Id'] == df.at[row - 1, 'Article_Id']:
                if df['Sentence_Id'][row] != df['Sentence_Id'][row - 1]:
                    # sanity check
                    if df.at[row, 'source'] == df.at[row - 1, 'source']:
                        print('Ill format data!!!')

                    # prev_sent
                    if df['Sentence_Id'][row] - df['Sentence_Id'][row - 1] == 1:
                        if df['Sentence_Id'][row - 1] == 1:
                            df.at[row, 'prev_sent'] = df.at[row - 1, 'source']
                        else:
                            df.at[row, 'prev_sent'] = df.at[row-1, 'prev_sent'] + ' ' + df.at[row-1, 'source']
                    else:
                        # missing at least 1 sentence here.
                        df.at[row, 'prev_sent'] = get_prev_text(article_path, df, row)
                else:
                    df.at[row, 'prev_sent'] = df.at[row - 1, 'prev_sent']
            else:
                if df['Sentence_Id'][row] != 1:
                    # missing at least 1 sentence here.
                    df.at[row, 'prev_sent'] = get_prev_text(article_path, df, row)
    df.to_csv(output_fpath, index=False, sep='\t', encoding='utf-8')


if __name__ == "__main__":
    args = get_args()

    file_name = ['train', 'dev', 'test']
    for fname in file_name:
        add_context(args.input_path + fname + '.csv', args.article_path, args.output_path + fname + '.csv')




