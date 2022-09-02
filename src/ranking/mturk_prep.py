"""
This is to generate all the questions (using four different models) for MTurk
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
    parser.description = "MTurk sample generations"
    parser.add_argument('-data_dir', type=str, default='/home-nfs/users/output/inquisitive/')
    parser.add_argument('-data_path', default='/home-nfs/users/data/inquisitive/fairseq'
                                              '/context_source_span_qtype/test.csv',
                        help='path of the 50 expert questions')
    parser.add_argument('-log_path', default='/home-nfs/users/output/inquisitive/MTurk/',
                        help='path of the log file')

    return parser.parse_args()


def random_question(file_path):
    "random select question"
    f_path_list = glob.glob(os.path.join(file_path, 'test_*.txt'))
    for f_path in f_path_list:
        lname = os.path.basename(f_path)[5:-4]
        if 'df_question' in locals():
            df_question[lname] = pd.read_csv(f_path, sep='\t', encoding='utf-8', header=None).squeeze()
        else:
            df_question = pd.read_csv(f_path, sep='\t', encoding='utf-8', header=None).squeeze().rename(lname).to_frame()
    return pd.Series([np.random.choice(i, 1)[0] for i in df_question.values])


def get_nonref_question(file_path):
    df_top1 = pd.read_excel('/home-nfs/users/output/inquisitive/qtype_allgen'
                            '/cssq_new_inqusitive_test_scores.xlsx').idxmax(axis=1)
    f_path_list = glob.glob(os.path.join(file_path, 'test_*.txt'))
    for f_path in f_path_list:
        lname = os.path.basename(f_path)[5:-4]
        if 'df_question' in locals():
            df_question[lname] = pd.read_csv(f_path, sep='\t', encoding='utf-8', header=None).squeeze()
        else:
            df_question = pd.read_csv(f_path, sep='\t', encoding='utf-8', header=None).squeeze().rename(lname).to_frame()

    df_question = df_question.assign(**dict.fromkeys(['selected'], ' '))
    for row in range(0, df_question.shape[0]):
        df_question.at[row, 'selected'] = df_question.at[row, df_top1.loc[row]]
    return df_question['selected']


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


if __name__ == "__main__":
    args = get_args()
    np.random.seed(1)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path, exist_ok=True)

    model_dir = {'cs': 'fairseq/context_source', 'css': 'fairseq/context_source_span',
                 'cssrqt':'fairseq/context_source_span_qtype', 'cssnqt': 'qtype_allgen/context_source_span_qtype'}

    # first sample 500 sentences from test set, make sure that the 300 expert questions are excluded
    data_df = pd.read_csv(args.data_path, sep='\t', encoding='utf-8')

    # new_df = data_df.sample(n=50, random_state=1)
    # index_50 = new_df.index
    # new_df = data_df.drop(index_50).sample(n=450, random_state=1)
    # final_idx = index_50.union(new_df.index)

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

    # write gold questions
    output_df = final_df[['context', 'context_id', 'sentence', 'sentence_id', 'target', 'question_id']].copy()
    output_df = output_df.rename(columns={'target': 'question'})
    output_df['question_id'] = output_df['question_id'] + '_4'
    output_df = output_df.sample(frac=1, random_state=1)
    for j in range(1, 101, 1):
        file_name = 'rc_inquisitive_mturk_input_' + '_'.join(['g', str(j)]) + '.json'
        write_json(output_df[5 * j - 5: 5 * j], file_name)

    # for each model get predictions
    for i, key in enumerate(model_dir):
        if key != 'cssnqt':
            continue
            f_path = os.path.join(args.data_dir + model_dir[key], 'test_30maxlen.txt')
            df_question = pd.read_csv(f_path, sep='\t', encoding='utf-8', header=None).squeeze()
            final_df['question'] = df_question.loc[final_idx].reset_index(drop=True)
        else:
            final_df['question'] = get_nonref_question(args.data_dir + model_dir[key]).loc[final_idx].reset_index(drop=True)

        # make sure the ids are unique
        output_df = final_df[['context', 'context_id', 'sentence', 'sentence_id', 'question', 'question_id']].copy()
        output_df['question_id'] = output_df['question_id'] + '_' + str(i)

        # shuffle order
        output_df = output_df.sample(frac=1, random_state=i)

        # then create 100 json files containing 5 examples in each
        for j in range(1, 101, 1):
            file_name = 'rc_inquisitive_mturk_input_' + '_'.join([key, str(j)]) + '.json'
            write_json(output_df[5 * j - 5: 5 * j], file_name)