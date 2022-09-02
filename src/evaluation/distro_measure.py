"""
Input generated questions,
and then print probability distributions over predicted question type labels (with the best Roberta classifier)
"""

import os
import glob
import torch
import argparse
import pandas as pd
from fairseq.models.roberta import RobertaModel
from data_utils import clean_unnecessary_spaces

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "question type distributions"

    # load roberta model
    parser.add_argument('-model_type', type=str, default='context_source_span_q')
    parser.add_argument('-model_dir', type=str, default='/home-nfs/users/output/inquisitive/question_type/')
    parser.add_argument('-checkpoint_name', type=str, default='checkpoint_best.pt')
    parser.add_argument('-data_path', type=str, default='/home-nfs/users/data/inquisitive/annotations/question_type/')

    # data set directory
    parser.add_argument('-log_path', default='/home-nfs/users/output/inquisitive/fairseq/',
                        help='path of the log file')
    parser.add_argument('-log_name', default='test_30maxlen.txt',
                        help='name of the log file')

    return parser.parse_args()

def get_score(tokens, roberta):
    tokens_encoded = roberta.encode(tokens)
    return torch.exp(roberta.predict('qtype_head', tokens_encoded).data.to('cpu'))[0].tolist()

def output_csv(df, cname, roberta, output_path):
    df['score'] = df[cname].apply(lambda x: get_score(x, roberta))
    df[['Explanation', 'Background', 'Elaboration', 'Instantiation', 'Definition', 'Other', 'Forward']] = \
        pd.DataFrame(df.score.tolist(), index=df.index)
    df.to_csv(output_path, index=False, sep='\t', encoding='utf-8')
    print('finished')

if __name__ == "__main__":
    args = get_args()
    args.model_dir = os.path.join(args.model_dir, args.model_type)
    args.data_path = os.path.join(args.data_path, args.model_type)

    # load model
    roberta = RobertaModel.from_pretrained(args.model_dir, checkpoint_file=args.checkpoint_name,
                                           data_name_or_path=args.data_path)
    roberta.eval()  # disable dropout

    roberta.cuda()

    # get path
    file_path = glob.glob(os.path.join(args.log_path, '*/' + args.log_name))

    # set parameters and process context data
    cname = 'generation'
    read_data_path = args.log_path.replace('/output/', '/data/')
    if 'dev' in args.log_name:
        output_name = 'dev_qtype_dis.csv'
        df_data = pd.read_csv(os.path.join(read_data_path, 'context_source_span/' + 'val.csv'),
                              sep='\t', encoding='utf-8')
        df_rel_data = pd.read_csv(os.path.join(read_data_path, 'context_source_span_rel/' + 'val.csv'),
                                  sep='\t', encoding='utf-8')
    else:
        output_name = 'test_qtype_dis.csv'
        df_data = pd.read_csv(os.path.join(read_data_path, 'context_source_span/' + 'test.csv'),
                              sep='\t', encoding='utf-8')
        df_rel_data = pd.read_csv(os.path.join(read_data_path, 'context_source_span_rel/' + 'test.csv'),
                                  sep='\t', encoding='utf-8')
    df_rel_data['source'] = df_rel_data['source'].str.replace(' [SEP] ', '\t', regex=False).str.split('\t')
    df_rel_data['source'] = df_rel_data['source'].apply((lambda x: ' [SEP] '.join(x[:-1])))


    # read data and cal scores
    for fpath in file_path:
        for key in ['/source/', '/source_qtype/', '/source_span/', '/source_span_qtype/',
                    '/context_source/', '/context_source_span/', '/context_source_qtype/',
                    '/context_source_span_qtype/']:
            if key in fpath:
                df = pd.read_csv(fpath, names=[cname], sep='\t', encoding='utf-8')
                if 'rel' in fpath:
                    df[cname] = (df_data['source'] + ' [SEP] ' + df[cname]).astype('str').apply(clean_unnecessary_spaces)
                else:
                    df[cname] = (df_rel_data['source'] + ' [SEP] ' + df[cname]).astype('str').apply(clean_unnecessary_spaces)
                output_csv(df, cname, roberta, fpath.replace(args.log_name, output_name))
