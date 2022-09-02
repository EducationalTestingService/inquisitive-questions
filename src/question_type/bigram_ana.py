"""
Are there any relations between leading bigrams of questions and question types?

only analyze annotated data
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.special import softmax
import matplotlib.pyplot as plt
from matplotlib import rcParams
from nltk.tokenize import word_tokenize
rcParams.update({'figure.autolayout': True})
pd.options.display.float_format = '{:,.2f}'.format

ngram = 1

data_path = '/home-nfs/users/data/inquisitive/annotations/question_type/context_source_span_question'
df_train = pd.read_csv(os.path.join(data_path, 'train.input'), names=['question'], sep='\t', encoding='utf-8')
df_train['label'] = pd.read_csv(os.path.join(data_path, 'train.label'), header=None, sep='\t', encoding='utf-8').squeeze()
df_dev = pd.read_csv(os.path.join(data_path, 'dev.input'), names=['question'], sep='\t', encoding='utf-8')
df_dev['label'] = pd.read_csv(os.path.join(data_path, 'dev.label'), sep='\t', header=None, encoding='utf-8').squeeze()

df = pd.concat([df_train, df_dev]).reset_index(drop=True)
df['question'] = df['question'].str.replace(' [SEP] ', '/t', regex=False).str.split('/t').str[-1]

for ngram in [1, 2]:
    df['bigrams'] = df['question'].apply(lambda x: word_tokenize(x)).apply(lambda x: ' '.join(x[:ngram]).lower())
    occur = pd.get_dummies(df.label).groupby(df.bigrams).sum()
    for col_name in ['C', 'E', 'B', 'D', 'I', 'F', 'O']:
        new_df = occur.sort_values(by=[col_name], ascending=False)
        new_df[col_name] = new_df.index.astype('str') + '(' + new_df[col_name].astype('str') + ')'
        print(new_df[col_name][:20].to_string(index=False))
        if ngram == 2 and col_name == 'B':
            new_df = occur.sort_values(by=[col_name], ascending=False)
            new_df[col_name].to_excel(os.path.expanduser('/tmp/Background.xlsx'))
    # occur = occur[occur.sum(axis=1) > 10]
    # s = occur.sum(axis=1)
    # occur = occur.loc[s.sort_values(ascending=False).index].T
    #
    # s = occur.sum(axis=1)
    # occur = occur.loc[s.sort_values(ascending=False).index]
    #
    # print(occur)
    #
    # plt.clf()
    #
    # mycmap = sns.light_palette("seagreen", as_cmap=True)
    # res = sns.heatmap(occur, annot=True, fmt='.0f', cmap=mycmap, cbar=False)
    # res.set(xlabel=None)
    # plt.tick_params(axis='x', labelrotation=30)
    # plt.tick_params(axis='y', labelrotation=0)
    #
    # # plt.savefig("output/crosstab_pandas.png", bbox_inches='tight', dpi=100)
    #
    # plt.show(bbox_inches='tight')
    #
    # # occur = softmax(occur.astype(float), axis=1)
    # occur = occur.div(occur.sum(axis=0), axis=1)
    # print(occur)
    # plt.clf()
    #
    # mycmap = sns.light_palette("seagreen", as_cmap=True)
    # res = sns.heatmap(occur, annot=True, fmt='.1f', cmap=mycmap, cbar=False)
    # res.set(xlabel=None)
    # plt.tick_params(axis='x', labelrotation=30)
    # plt.tick_params(axis='y', labelrotation=0)
    #
    # # plt.savefig("output/crosstab_pandas.png", bbox_inches='tight', dpi=100)
    #
    # plt.show(bbox_inches='tight')