"""
This file is to get the gold annotation for the first 50 examples (maximum voting)
And then combine it with other 750 samples.
- 50 samples would be used in training set
- after combination, train, dev, test dataset size would be 600, 100, 100.
"""

import os
import pandas as pd
from data_utils import clean_unnecessary_spaces

def sanity_check(df):
    print(set(df['qtype'].values))
    return df[['prev_sent', 'source', 'Span', 'target', 'qtype']]

def max_vote(df):
    # do maximum voting of first 50 examples
    df = df[['prev_sent', 'source', 'Span', 'target', 'user1\'s new labels', 'user2', 'user3 (Adjudicated) ']]
    # sanity check
    for key in df.columns.values:
        print(set(df[key].values))
    df['qtype'] = df[['user1\'s new labels', 'user2', 'user3 (Adjudicated) ']].mode(axis=1)[0]
    return df[['prev_sent', 'source', 'Span', 'target', 'qtype']]

def proc_file(df):
    df['prev_sent'].fillna('NO_CONTEXT', inplace=True)
    df['prev_sent'].replace(' ', 'NO_CONTEXT', inplace=True)
    df['prev_sent'] = df['prev_sent'].astype('str')
    df['Span'] = df['Span'].astype('str')
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df['all'] = df['prev_sent'] + ' [SEP] ' + df['source'] + ' [SEP] ' + df['Span'] + ' [SEP] ' + df['target']
    # df['all'] = df['source'] + ' [SEP] ' + df['Span']
    df['all'] = df['all'].apply(clean_unnecessary_spaces)
    return df

def cal_stat(first_50, combined):
    """
    sample these for train (1350)
    C: 400 - 16 = 384
    B: 367 - 11 = 356
    E: 329 - 10 = 319
    D: 103 - 6 = 97
    F: 28 - 3 = 25
    I: 144 - 3 = 141
    O: 29 - 1 = 28
    """
    stat = first_50.copy()
    stat = stat['qtype'].value_counts().reset_index()
    print(stat)

    stat = pd.concat([first_50, combined])
    stat['qtype'] = stat['qtype'].replace('N', 'O')
    stat = stat['qtype'].value_counts().reset_index()
    stat['qtype'] = stat['qtype'] * 1400 / 1550
    print(stat)

def sample_train(df):
    # put N into O
    df['qtype'] = df['qtype'].replace('N', 'O')
    # sample based on numbers
    df_c = df[df['qtype'] == 'C'].sample(n=384, random_state=1)
    df_b = df[df['qtype'] == 'B'].sample(n=356, random_state=1)
    df_e = df[df['qtype'] == 'E'].sample(n=319, random_state=1)
    df_d = df[df['qtype'] == 'D'].sample(n=97, random_state=1)
    df_f = df[df['qtype'] == 'F'].sample(n=25, random_state=1)
    df_i = df[df['qtype'] == 'I'].sample(n=141, random_state=1)
    df_o = df[df['qtype'] == 'O'].sample(n=28, random_state=1)
    train = pd.concat([df_c, df_b, df_e, df_d, df_f, df_i, df_o])

    # get dev set for the rest
    dev = df.drop(train.index).sample(frac=1, random_state=1)
    return train, dev


if __name__ == '__main__':
    data_path = '/home-nfs/users/data/inquisitive/annotations/original'
    output_path = '/home-nfs/users/data/inquisitive/annotations/question_type'

    # read data
    first_50 = pd.read_excel(os.path.join(data_path,
                                          'Inquisitive_Question_Classification_Pilot_Annotation.xlsx'))
    user2 = pd.read_excel(os.path.join(data_path,
                                          'user3_inquisitive_annotations_250.xlsx'))
    user3 = pd.read_excel(os.path.join(data_path, 'user3_train_samples_250.xlsx'))
    user1 = pd.read_excel(os.path.join(data_path, 'user1_train_samples.xlsx'))
    user22 = pd.read_excel(os.path.join(data_path,'third_batch/user3_batch2.xlsx'))
    user32 = pd.read_csv(os.path.join(data_path, 'third_batch/user3_batch2.csv'))
    # remove extra lines when break into two questions
    user32 = user32.drop([10, 97, 107, 155]).reset_index(drop=True)
    user32['Annotation Label'].fillna('O', inplace=True)
    user12 = pd.read_excel(os.path.join(data_path, 'third_batch/user1_batch2.xlsx'))

    # process 50 examples
    first_50 = max_vote(first_50)

    # process first batch
    user2 = sanity_check(user2.rename(columns={"annotation": "qtype"}))
    user3 = sanity_check(user3.rename(columns={"Annotation Label": 'qtype'}))
    user1 = sanity_check(user1.rename(columns={"Label": "qtype"}))

    # process second batch
    user22 = sanity_check(user22.rename(columns={"question type": "qtype"}))
    user32 = sanity_check(user32.rename(columns={"Annotation Label": 'qtype'}))
    user12 = sanity_check(user12.rename(columns={"Label": "qtype"}))

    # separate train, dev and test set
    combined = pd.concat([user2, user3, user1, user22, user32, user12]).reset_index(drop=True)

    # cal statistics
    # cal_stat(first_50, combined)

    # take 1350 for train, and rest 150 for dev
    train, dev = sample_train(combined)
    train = pd.concat([train, first_50]).sample(frac=1, random_state=1)

    # verify
    # stat = train['qtype'].value_counts()
    # stat /= 7
    # print(stat)
    # print(dev['qtype'].value_counts())

    # write to file
    train.to_csv(os.path.join(output_path, 'train.csv'), index=False, sep='\t', encoding='utf-8')
    dev.to_csv(os.path.join(output_path, 'dev.csv'), index=False, sep='\t', encoding='utf-8')

    # write data for the classifier
    df_list = [train, dev]
    output_path = os.path.join(output_path, 'context_source_span_question')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for idx, file_name in enumerate(['train', 'dev']):
        df = proc_file(df_list[idx])
        src_file = os.path.join(output_path, file_name + '.input')
        tgt_file = os.path.join(output_path, file_name + '.label')
        with open(src_file, 'w+', encoding='utf-8') as output1, open(tgt_file, 'w+', encoding='utf-8') as output2:
            output1.write('\n'.join(df['all'].tolist()))
            output2.write('\n'.join(df['qtype'].tolist()))

