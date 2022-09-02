# encoding: utf-8

import argparse

import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

parser = argparse.ArgumentParser(description='calculate sentence length distribution of INQUISITIVE')
parser.add_argument('-train', default='/home/nlp-text/dynamic/users/data/INQUISITIVE/context/train.csv',
                    help='path of the test file')
args = parser.parse_args()

def cal_length(df_text):
    length = []
    for row in df_text:
        row = row.strip().split(' ')
        if row[0]:
            length.append(len(row))
        else:
            length.append(0)
    return length


if __name__ == '__main__':
    # read training data
    df = pd.read_csv(args.train, sep='\t', encoding='utf-8')
    # df['prev_sent'] = df['prev_sent'].astype('str')
    prev_len = cal_length(df['prev_sent'])
    source_len = cal_length(df['source'])
    text_len = []
    for i in range(len(source_len)):
        text_len.append(prev_len[i] + source_len[i])


    print(max(prev_len), max(source_len), max(text_len))

    # draw distribution plots
    bins = np.arange(0, 280, 20)
    plt.hist(prev_len, density=True, stacked=True, bins=bins)
    plt.xticks(bins)
    plt.gca().set_yticklabels(['{:.0f}'.format(x * 100 * 20) for x in plt.gca().get_yticks()])
    plt.title('Length distribution of INQUISITIVE previous context')
    plt.ylabel('fraction (%)')
    plt.show()

    bins = np.arange(0, 120, 20)
    n, bins, patches = plt.hist(source_len, density=True, stacked=True, bins=bins)
    plt.title('Length distribution of INQUISITIVE source sentence')
    plt.ylabel('fraction (%)')
    plt.xticks(bins)
    plt.gca().set_yticklabels(['{:.0f}'.format(x * 100 * 20) for x in plt.gca().get_yticks()])
    plt.show()

    bins = np.arange(0, 320, 20)
    plt.hist(text_len, density=True, stacked=True, bins=bins)
    plt.title('Length distribution of INQUISITIVE previous context + source sentence')
    plt.ylabel('fraction (%)')
    plt.xticks(bins)
    plt.gca().set_yticklabels(['{:.0f}'.format(x * 100 * 20) for x in plt.gca().get_yticks()])
    plt.show()

