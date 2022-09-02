# encoding: utf-8

import argparse
import re
import pandas as pd

parser = argparse.ArgumentParser(description='calculate distribution of why question in SQuAD')
parser.add_argument('-train', default='/home/nlp-text/dynamic/users/bart/data/combinedtrain.txt',
                    help='path of the test file')
args = parser.parse_args()


# join results with paragraphs
train_data = pd.read_csv(args.train, sep='\t')
print(train_data.shape[0])
train_data = train_data.drop_duplicates()
print(train_data.shape[0])

# filter the why questions
why_data = train_data[train_data['target'].str.contains('Why')]
print(why_data.shape[0], why_data.shape[0]/train_data.shape[0] * 100)
reason_data = train_data[train_data['target'].str.contains('for what reason')]
print(reason_data.shape[0],  reason_data.shape[0]/train_data.shape[0] * 100)

# # get target start strings
# string_set = set()
# for x in train_data['target']:
#     string_set.add(' '.join(x.split(' ')[:2]))
# print(string_set)