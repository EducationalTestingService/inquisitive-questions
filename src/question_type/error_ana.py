"""
This file is to compute error analysis
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
# from fairseq.models.roberta import RobertaModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.description = "Analyses for question type classification"

    # load roberta model
    parser.add_argument('-input_name', type=str, default='context_source_span_q')
    parser.add_argument('-model_dir', type=str, default='/home-nfs/users/output/inquisitive/question_type/')
    parser.add_argument('-checkpoint_name', type=str, default='checkpoint_best.pt')
    parser.add_argument('-data_path', type=str, default='/home-nfs/users/data/inquisitive/annotations/question_type/')
    return parser.parse_args()


def predict(tokens, label_fn, roberta):
    tokens_encoded = roberta.encode(tokens)
    pred = label_fn(roberta.predict('qtype_head', tokens_encoded).argmax().item())
    return pred


if __name__ == '__main__':
    args = get_args()
    args.model_dir = os.path.join(args.model_dir, args.input_name)
    args.data_path = os.path.join(args.data_path, args.input_name)

    # # load model
    # roberta = RobertaModel.from_pretrained(args.model_dir, checkpoint_file=args.checkpoint_name,
    #                                        data_name_or_path=args.data_path)
    # roberta.eval()  # disable dropout
    # roberta.cuda()
    #
    # # get func
    # label_fn = lambda label: roberta.task.label_dictionary.string(
    #     [label + roberta.task.label_dictionary.nspecial]
    # )
    #
    # # get predictions
    # df = pd.read_csv(os.path.join(args.data_path, 'dev.input'), names=['source'], sep='\t\n', encoding='utf-8')
    # df['label'] = pd.read_csv(os.path.join(args.data_path, 'dev.label'), names=['label'], sep='\t', encoding='utf-8')
    # df['pred'] = df['source'].apply(lambda x: predict(x, label_fn, roberta))
    #
    # # start analyze
    # df.to_csv(os.path.join(args.data_path, 'dev_analysis.csv'), index=False, sep='\t', encoding='utf-8')

    df = pd.read_csv(os.path.join(args.data_path, 'dev_analysis.csv'), sep='\t', encoding='utf-8')
    # df_dict = {"B": 'Background', "C": 'Explanation', "D": 'Definition', "E": 'Elaboration', "F": 'Forward',
    #         "I": 'Instantiation', "O": 'Others'}
    # df = df.replace({"label": df_dict, "pred": df_dict})

    confusion_matrix = pd.crosstab(df['label'], df['pred'], rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)

    plt.clf()

    mycmap = sns.light_palette("seagreen", as_cmap=True)
    res = sns.heatmap(confusion_matrix.T, annot=True, fmt='.0f', cmap=mycmap, cbar=False)

    plt.tick_params(axis='x', labelrotation=0)
    plt.tick_params(axis='y', labelrotation=0)
    # plt.savefig("output/crosstab_pandas.png", bbox_inches='tight', dpi=100)

    plt.show(bbox_inches='tight')

