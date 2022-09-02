"""
This file is to predict the question types
"""
import os
import argparse
import pandas as pd
from fairseq.models.roberta import RobertaModel

from data_utils import clean_unnecessary_spaces


def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "classfication for question types"

    # load roberta model
    parser.add_argument('-model_type', type=str, default='context_source_span_q')
    parser.add_argument('-model_dir', type=str, default='/home-nfs/users/output/inquisitive/question_type/')
    parser.add_argument('-checkpoint_name', type=str, default='checkpoint_best.pt')
    parser.add_argument('-data_path', type=str, default='/home-nfs/users/data/inquisitive/annotations/question_type/')

    # data path need to pred
    parser.add_argument('-input_path', type=str, default='/home-nfs/users/data/inquisitive/annotations/qtype_pred/origin')

    # output path for model
    parser.add_argument('-output_path', type=str,
                        default='/home-nfs/users/data/inquisitive/annotations/qtype_pred/pred',
                        help='question_types_predictions')
    return parser.parse_args()

def proc_file(df):
    df['prev_sent'].fillna('NO_CONTEXT', inplace=True)
    df['prev_sent'].replace(' ', 'NO_CONTEXT', inplace=True)
    df['prev_sent'] = df['prev_sent'].astype('str')
    df['Span'] = df['Span'].astype('str')
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df['all'] = df['prev_sent'] + ' [SEP] ' + df['source'] + ' [SEP] ' + df['Span'] + ' [SEP] ' + df['target']
    df['all'] = df['all'].apply(clean_unnecessary_spaces)
    return df

def get_score(tokens, label_fn, roberta):
    tokens_encoded = roberta.encode(tokens)
    return label_fn(roberta.predict('qtype_head', tokens_encoded).argmax().item())


if __name__ == '__main__':
    args = get_args()
    args.model_dir = os.path.join(args.model_dir, args.model_type)
    args.data_path = os.path.join(args.data_path, args.model_type)

    # load roberta
    roberta = RobertaModel.from_pretrained(args.model_dir, checkpoint_file=args.checkpoint_name,
                                           data_name_or_path=args.data_path)
    roberta.eval()  # disable dropout
    roberta.cuda()
    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )

    # do the predictions and write to output
    for fname in ['train', 'dev', 'test']:
        df = pd.read_csv(os.path.join(args.input_path, fname + '.csv'), sep='\t', encoding='utf-8')
        df = proc_file(df)
        df['label'] = df['all'].apply(lambda x: get_score(x, label_fn, roberta))
        df.to_csv(os.path.join(args.output_path, fname + '.csv'), sep='\t', encoding='utf-8', index=False)

