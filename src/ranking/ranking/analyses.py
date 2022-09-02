"""
This file is to get best questions and generate MTurk files
"""
import os
import glob
import json
import numpy as np
import argparse
import pandas as pd
from data_utils import clean_unnecessary_spaces

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "MTurk sample generations （ranking)"
    parser.add_argument('-data_dir', type=str, default='/home-nfs/users/output/inquisitive/')
    parser.add_argument('-data_path', default='/home-nfs/users/data/inquisitive/fairseq'
                                              '/context_source_span_qtype/test.csv',
                        help='path of the 50 expert questions')
    parser.add_argument('-log_path', default='/home-nfs/users/output/inquisitive/MTurk/',
                        help='path of the log file')
    return parser.parse_args()


def write_json(df, file_name):
    output_path = os.path.join(args.log_path, file_name)

    # add extra content to each line
    extra_content = {"feedback": [
        "Does the question seem grammatically correct?",
        "Does the question make sense?",
        "Does the question seem relevant to the source?",
        "Does the question show curiosity to learn more about the topic?"],
        "answer_options": [
            "Yes",
            "Somewhat",
            "No"
        ]}

    result = df.to_json(orient="records")
    parsed = json.loads(result)
    for ele in parsed:
        ele.update(extra_content)
    out_packer = {"URL_location": file_name,
                  "additional_feedback": "some comments left by the Turker",
                  "mturk_task_content": parsed}
    with open(output_path, 'w+') as f:
        json.dump(out_packer, f, indent=4)


def get_question(output_df):
    # read results
    df = pd.read_csv('/home-nfs/users/output/inquisitive/MTurk/rc_inquisitive_mturk_input_cssnqt_r.csv',
                     sep='\t', encoding='utf-8')
    df_score = pd.read_csv('/home-nfs/users/output/inquisitive/MTurk/rc_inquisitive_mturk_input_cssnqt_s.csv',
                           sep='\t', encoding='utf-8')

    # find max vote
    # df['max'] = df[["Definition", "Background", "Instantiation",
    #                 "Explanation", "Forward", "Elaboration"]].idxmax(axis=1)
    df['max_all'] = df.eq(df.max(1), axis=0).dot(df.columns + ',').str.rstrip(',')
    df['max'] = df['max_all'].copy()
    for row in range(0, df.shape[0]):
        if ',' in df.at[row, 'max_all']:
            col_values = df.at[row, 'max_all'].split(',')
            df.at[row, 'max'] = df_score[col_values].loc[row].idxmax()
    df.to_csv('/home-nfs/users/output/inquisitive/MTurk/rc_inquisitive_mturk_input_cssnqt_f.csv', sep='\t',
              encoding='utf-8', index=False)
    # get questions
    df_question = pd.read_csv('/home-nfs/users/output/inquisitive/MTurk/rc_inquisitive_mturk_input_cssnqt.csv',
                        sep='\t', encoding='utf-8')
    df_question = df_question.assign(**dict.fromkeys(['selected'], ' '))
    for row in range(0, df_question.shape[0]):
        df_question.at[row, 'selected'] = df_question.at[row, df['max'].loc[row]]
    return df_question['selected']


if __name__ == '__main__':
    args = get_args()

    # get other parts for json files
    np.random.seed(1)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path, exist_ok=True)

    model_dir = {'cssnqt': 'qtype_allgen/context_source_span_qtype'}

    # first sample 500 sentences from test set, make sure that the 300 expert questions are excluded
    data_df = pd.read_csv(args.data_path, sep='\t', encoding='utf-8')

    # get 1st batch index
    new_df_1 = data_df.sample(n=50, random_state=1)

    # get unique source and exclude 1st batch
    for i in new_df_1.index:
        if i in data_df.index:
            Article_Id = data_df.at[i, 'Article_Id']
            Sentence_Id = data_df.at[i, 'Sentence_Id']
            data_df = data_df.loc[(data_df['Article_Id'] != Article_Id) | (data_df['Sentence_Id'] != Sentence_Id)]

    data_df_unique = data_df.groupby(['Article_Id', 'Sentence_Id'], group_keys=False).apply(
        lambda data_df: data_df.sample(1, random_state=1))
    data_df_2 = data_df_unique.sample(n=250, random_state=1)

    # get the unique index part (417)
    data_df_unique = data_df_unique.drop(data_df_2.index)
    final_idx = data_df_unique.index

    # get the non-unique index part (83)
    for i in data_df_2.index:
        if i in data_df.index:
            new_df = data_df.loc[(data_df['Article_Id'] == data_df_2.at[i, 'Article_Id']) &
                                 (data_df['Sentence_Id'] == data_df_2.at[i, 'Sentence_Id']) &
                                 (data_df['Span'] != data_df_2.at[i, 'Span'])]
            if new_df.shape[0] > 0:
                final_idx = final_idx.union(new_df.sample(1, random_state=1).index)
                if final_idx.shape[0] >= 500:
                    break

    # get sentences and context, find span position
    final_df = pd.read_csv(args.data_path.replace('context_source_span_qtype', 'context_source_span'),
                          sep='\t', encoding='utf-8').loc[final_idx]
    final_df['context_id'] = 'context_' + final_df['Article_Id'].astype('str') + '_' \
                             + final_df['Sentence_Id'].astype('str')
    final_df['sentence_id'] = 'source_' + final_df['Article_Id'].astype('str') + '_' \
                              + final_df['Sentence_Id'].astype('str')
    final_df['question_id'] = 'question_' + final_df.index.to_series().astype('str')
    final_df.rename(columns={"prev_sent": "context"}, inplace=True)
    final_df['source'] = final_df['source'].str.replace(' [SEP] ', '\t',
                                                       regex=False).str.split('\t').apply(lambda x: x[1])
    final_df['sentence'] = final_df['source'].copy()
    final_df = final_df.reset_index(drop=True)
    for row in range(0, final_df.shape[0]):
        # prepare span
        sentence = final_df['sentence'][row].split(' ')
        span_start, span_end = final_df['Span_Start_Position'][row], final_df['Span_End_Position'][row]
        assert(' '.join(sentence[span_start:span_end]) == final_df['Span'][row])
        sentence[span_start] = '<' + sentence[span_start]
        sentence[span_end - 1] += '>'
        final_df.at[row, 'sentence'] = clean_unnecessary_spaces(' '.join(sentence))

        # prepare context and context id
        if final_df.at[row, 'context'] == 'NO_CONTEXT':
            final_df.at[row, 'context_id'] = 'context_0_0'
        else:
            final_df.at[row, 'context'] = clean_unnecessary_spaces(final_df.at[row, 'context'])
    final_df = final_df[['context', 'context_id', 'sentence', 'sentence_id', 'question_id', 'target']]

    # for each model get predictions
    for i, key in enumerate(model_dir):
        # make sure the ids are unique
        final_df = final_df.assign(**dict.fromkeys(['question'], ' '))
        output_df = final_df[['context', 'context_id', 'sentence', 'sentence_id', 'question', 'question_id']].copy()
        output_df['question_id'] = output_df['question_id'] + '_' + str(i)

        # # save order
        # rank_index = output_df.sample(frac=1, random_state=3).sample(frac=1, random_state=3)
        # rqt_index = output_df.sample(frac=1, random_state=2)
        # nqt_index = output_df.sample(frac=1, random_state=3)
        # rank_index.index.to_series().to_csv('/home-nfs/users/output/inquisitive/MTurk/rank_index.csv',
        #                                     sep='\t', encoding='utf-8', index=False)
        # rqt_index.index.to_series().to_csv('/home-nfs/users/output/inquisitive/MTurk/rqt_index.csv',
        #                                    sep='\t', encoding='utf-8', index=False)
        # nqt_index.index.to_series().to_csv('/home-nfs/users/output/inquisitive/MTurk/nqt_index.csv',
        #                                    sep='\t', encoding='utf-8', index=False)

        # shuffle order
        output_df = output_df.sample(frac=1, random_state=3).reset_index(drop=True)

        # get questions
        output_df['question'] = get_question(output_df)

        # shuffle again
        output_df = output_df.sample(frac=1, random_state=3)

        # then create 100 json files containing 5 examples in each
        for j in range(1, 101, 1):
            file_name = 'rc_inquisitive_mturk_input_cssrank_' + str(j) + '.json'
            write_json(output_df[5 * j - 5: 5 * j], file_name)

