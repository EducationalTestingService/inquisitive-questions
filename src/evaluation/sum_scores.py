"""
This file is to sum scores for all csv file in order to compare between models
"""
import os
import glob
import numpy
import argparse
import pandas as pd

pd.options.display.float_format = '{:,.3f}'.format

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "sum scores"

    # data set directory
    parser.add_argument('-file_path', default='/home-nfs/users/output/inquisitive/fairseq/',
                        help='path of the log file')

    return parser.parse_args()

def get_path(fname):
    file_path = glob.glob(os.path.join(args.file_path, '*/' + fname + '.csv'))
    # file_path = glob.glob(os.path.join(args.file_path, '*/ranked/' + fname + '.csv'))
    # file_path = glob.glob(os.path.join(args.file_path, '*/nqtype/' + fname + '.csv'))
    return file_path

def get_relations():
    """
    get non-repeat index
    get relations
    """
    read_data_path = '/home-nfs/users/data/inquisitive/fairseq/'
    df_data = pd.read_csv(os.path.join(read_data_path, 'context_source_span/' + 'val.csv'),
                          sep='\t', encoding='utf-8')
    df_rel_data = pd.read_csv(os.path.join(read_data_path, 'context_source_span_rel/' + 'val.csv'),
                              sep='\t', encoding='utf-8')
    # get index


    return df_rel_data['source'].str.replace(' [SEP] ', '\t', regex=False).str.split('\t').str[-1]


# def ana_rel(df, mname):
#     if 'rel' in mname:
#         df['rel'] = rel_column
#         pd.crosstab(df['rel'], df['qtype'], rownames=['relation'], colnames=['qtype']).sort_values(by=['Background'])

if __name__ == "__main__":
    args = get_args()
    # first deal with energy based and gpt2
    fname_list = ['test_energy', 'test_gen_gpt2']
    for fname in fname_list:
        print('*' * 5 + fname + '*' * 5)
        f_path = get_path(fname)
        avg_dict = dict()
        for fpath in f_path:
            df = pd.read_csv(fpath, sep='\t', encoding='utf-8')
            if '/ranked/' in fpath or '/nqtype/' in fpath:
                avg_dict[os.path.basename(os.path.dirname(fpath)[:-7])] = df['score'].mean()
            else:
                avg_dict[os.path.basename(os.path.dirname(fpath))] = df['score'].mean()
        avg_df = pd.DataFrame.from_dict(avg_dict, orient='index', columns=['avg_score']).reset_index()
        avg_df = avg_df.sort_values(by=['avg_score']).reset_index(drop=True)
        print(avg_df)
        avg_df.to_excel(os.path.join(args.file_path, fname + '_test.xlsx'))

    # # then deal with qtypes
    # fname = 'dev_qtype_dis'
    # f_path = get_path(fname)
    # avg_dict = {}
    #
    # # get relations
    # rel_column = get_relations()
    #
    # for fpath in f_path:
    #     df = pd.read_csv(fpath, sep='\t', encoding='utf-8')
    #     df['qtype'] = df[['Explanation', 'Background', 'Elaboration','Instantiation',
    #                       'Definition', 'Other', 'Forward']].idxmax(axis=1)
    #     ana_rel(df, os.path.basename(os.path.dirname(fpath)))
    #     avg_dict[os.path.basename(os.path.dirname(fpath))] = df['qtype'].value_counts(normalize=True)
    # avg_df = pd.DataFrame.from_dict(avg_dict, orient='index').reset_index()
    # avg_df = avg_df.sort_values(by=['Explanation']).reset_index(drop=True)
    # print(avg_df)
    # avg_df.to_excel(os.path.join(args.file_path, fname + '.xlsx'))