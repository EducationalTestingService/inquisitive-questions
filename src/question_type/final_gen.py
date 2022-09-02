"""
This file is to collect the question type predictions, and generate output for model
"""

import os
import argparse
import pandas as pd

from data_utils import clean_unnecessary_spaces
from process_data import max_vote, sanity_check


def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "classfication for question types"
    parser.add_argument('-pred_path', type=str,
                        default='/home-nfs/users/data/inquisitive/annotations/qtype_pred/pred',
                        help='question_types_predictions')
    parser.add_argument('-index_path', type=str,
                        default='/home-nfs/users/data/inquisitive/annotations/qtype_pred/origin',
                        help='index for all these files')
    parser.add_argument('-annot_path', type=str,
                        default='/home-nfs/users/data/inquisitive/annotations/original',
                        help='annotations')
    parser.add_argument('-control_path', type=str,
                        default='/home-nfs/users/data/inquisitive/fairseq',
                        help='path for input with control codes')
    parser.add_argument('-control_name', type=str,
                        default='context_source_span_qtype',
                        choices=['context_source_span_qtype', 'context_source_qtype',
                                 'source_span_qtype', 'source_qtype'],
                        help='path for input with control codes')
    return parser.parse_args()

def change_control(df):
    df['qtype'] = df['qtype'].replace('C', 'Explanation')
    df['qtype'] = df['qtype'].replace('E', 'Elaboration')
    df['qtype'] = df['qtype'].replace('D', 'Definition')
    df['qtype'] = df['qtype'].replace('B', 'Background')
    df['qtype'] = df['qtype'].replace('I', 'Instantiation')
    df['qtype'] = df['qtype'].replace('F', 'Forward')
    df['qtype'] = df['qtype'].replace('O', 'Other')
    return df


def prepare_df(df, output_path, file_name):
    # check prev_sent
    df['prev_sent'].fillna('NO_CONTEXT', inplace=True)
    df['prev_sent'].replace(' ', 'NO_CONTEXT', inplace=True)

    # remove extra spaces
    df['prev_sent'] = df['prev_sent'].astype('str')
    df['Span'] = df['Span'].astype('str')
    df['source'] = df['source'].apply(clean_unnecessary_spaces)
    df['Span'] = df['Span'].apply(clean_unnecessary_spaces)
    df['target'] = df['target'].apply(clean_unnecessary_spaces)
    df['prev_sent'] = df['prev_sent'].apply(clean_unnecessary_spaces)

    # change control code
    df = change_control(df)

    # create source sentence
    if 'context_source_span_qtype' in output_path:
        df['source'] = df['prev_sent'] + ' [SEP] ' + df['source'] + ' [SEP] ' + df['Span'] + ' [SEP] ' + df['qtype']
    elif 'context_source_qtype' in output_path:
        df['source'] = df['prev_sent'] + ' [SEP] ' + df['source'] + ' [SEP] ' + df['qtype']
    elif 'source_span_qtype' in output_path:
        df['source'] = df['source'] + ' [SEP] ' + df['Span'] + ' [SEP] ' + df['qtype']
    else:
        df['source'] = df['source'] + ' [SEP] ' + df['qtype']

    df.to_csv(os.path.join(output_path, file_name + '.csv'), index=False, sep='\t', encoding='utf-8')

    src_file = os.path.join(output_path, file_name + '.source')
    tgt_file = os.path.join(output_path, file_name + '.target')
    with open(src_file, 'w+', encoding='utf-8') as output1, open(tgt_file, 'w+', encoding='utf-8') as output2:
        output1.write('\n'.join(df['source'].tolist()))
        output2.write('\n'.join(df['target'].tolist()))


if __name__ == '__main__':
    args = get_args()

    # read annotation
    first_50 = pd.read_excel(os.path.join(args.annot_path, 'Inquisitive_Question_Classification_Pilot_Annotation.xlsx'))
    user2 = pd.read_excel(os.path.join(args.annot_path, 'user3_inquisitive_annotations_250.xlsx'))
    user3 = pd.read_excel(os.path.join(args.annot_path, 'user3_train_samples_250.xlsx'))
    user1 = pd.read_excel(os.path.join(args.annot_path, 'user1_train_samples.xlsx'))
    user22 = pd.read_excel(os.path.join(args.annot_path, 'third_batch/user3_batch2.xlsx'))
    user32 = pd.read_csv(os.path.join(args.annot_path, 'third_batch/user3_batch2.csv'))
    user32 = user32.drop([10, 97, 107, 155]).reset_index(drop=True)
    user32['Annotation Label'].fillna('O', inplace=True)
    user12 = pd.read_excel(os.path.join(args.annot_path, 'third_batch/user1_batch2.xlsx'))

    # process first batch
    user2 = sanity_check(user2.rename(columns={"annotation": "qtype"}))
    user3 = sanity_check(user3.rename(columns={"Annotation Label": 'qtype'}))
    user1 = sanity_check(user1.rename(columns={"Label": "qtype"}))
    user3['qtype'] = user3['qtype'].replace('N', 'O')

    # process second batch
    user22 = sanity_check(user22.rename(columns={"question type": "qtype"}))
    user32 = sanity_check(user32.rename(columns={"Annotation Label": 'qtype'}))
    user12 = sanity_check(user12.rename(columns={"Label": "qtype"}))
    user32['qtype'] = user32['qtype'].replace('N', 'O')

    # process first 50 examples
    first_batch = max_vote(first_50)

    # get index for all
    first_batch['index'] = pd.read_csv(os.path.join(args.index_path, '50_index.csv'), sep='\t', encoding='utf-8')
    second_batch = pd.concat([user1, user3, user2]).reset_index(drop=True)
    second_batch['index'] = pd.read_csv(os.path.join(args.index_path, '750_index.csv'), sep='\t', encoding='utf-8')
    third_batch = pd.concat([user12, user32, user22]).reset_index(drop=True)
    third_batch['index'] = pd.read_csv(os.path.join(args.index_path, '750_2_index.csv'), sep='\t', encoding='utf-8')

    # read prediction and get index
    fourth_batch = pd.read_csv(os.path.join(args.pred_path, 'train.csv'),
                               sep='\t', encoding='utf-8').rename(columns={"label": "qtype"})
    dev = pd.read_csv(os.path.join(args.pred_path, 'dev.csv'),
                      sep='\t', encoding='utf-8').rename(columns={"label": "qtype"})
    test = pd.read_csv(os.path.join(args.pred_path, 'test.csv'),
                       sep='\t', encoding='utf-8').rename(columns={"label": "qtype"})
    fourth_batch['index'] = pd.read_csv(os.path.join(args.index_path, 'pred_index.csv'), sep='\t', encoding='utf-8')

    # process dev and remove no span index
    no_span_idx = dev.index[dev['Span'].isnull()].tolist()
    dev = dev.drop(no_span_idx).reset_index(drop=True)

    # put them together and sort index
    train = pd.concat([first_batch, second_batch, third_batch, fourth_batch])
    train = train.set_index(['index']).sort_index()

    # output to file
    args.control_path = os.path.join(args.control_path, args.control_name)
    if not os.path.exists(args.control_path):
        os.makedirs(args.control_path)
    prepare_df(train, args.control_path, 'train')
    prepare_df(dev, args.control_path, 'val')
    prepare_df(test, args.control_path, 'test')
