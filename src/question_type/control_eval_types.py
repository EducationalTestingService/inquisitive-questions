"""
This file is to evaluate the controllability of question types
so only focus on types
"""
import os
import glob
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
    parser.add_argument('-model_data_path', type=str, default='/home-nfs/users/data/inquisitive/annotations/question_type/')

    # specify data path
    parser.add_argument('-label_path', default='/home-nfs/users/data/inquisitive/fairseq/context_source_span_qtype',
                        help='path of the log file')
    parser.add_argument('-data_path', default='/home-nfs/users/output/inquisitive/qtype_allgen/context_source_span_qtype',
                        help='path of the log file')

    # parser.add_argument('-data_name', default='test_Background.txt',
    #                     help='name of the log file')
    return parser.parse_args()


def predict(tokens, label_fn, roberta):
    tokens_encoded = roberta.encode(tokens)
    pred = label_fn(roberta.predict('qtype_head', tokens_encoded).argmax().item())
    return pred


if __name__ == '__main__':
    args = get_args()
    args.model_dir = os.path.join(args.model_dir, args.input_name)
    args.model_data_path = os.path.join(args.model_data_path, args.input_name)

    # # load model
    # roberta = RobertaModel.from_pretrained(args.model_dir, checkpoint_file=args.checkpoint_name,
    #                                        data_name_or_path=args.model_data_path)
    # roberta.eval()  # disable dropout
    # roberta.cuda()
    #
    # # get func
    # label_fn = lambda label: roberta.task.label_dictionary.string(
    #     [label + roberta.task.label_dictionary.nspecial]
    # )
    #
    # # read data (no filtering)
    # file_path = glob.glob(os.path.join(args.data_path, 'test_*.txt'))
    # for f_path in file_path:
    #     df = pd.read_csv(f_path, names=['question'], sep='\t\n', encoding='utf-8')
    #     df['source'] = pd.read_csv(os.path.join(args.label_path, 'test.source'), names=['source'], sep='\t',
    #                               encoding='utf-8')
    #     df['source'] = df['source'].str.replace(' [SEP] ', '/t', regex=False).str.split('/t')
    #     df['source'] = [' [SEP] '.join(map(str, l[:-1])) for l in df['source']]
    #     df['source'] = df['source'] + ' [SEP] ' + df['question']
    #     plot_name = os.path.basename(f_path)[5:-4]
    #     df['label'] = plot_name
    #
    #     # get predictions
    #     df['pred'] = df['source'].apply(lambda x: predict(x, label_fn, roberta))
    #
    #     # start analyze
    #     # df.to_csv(os.path.join(args.data_path, 'test_control_eval.csv'), index=False, sep='\t', encoding='utf-8')
    #     #
    #     # df = pd.read_csv(os.path.join(args.data_path, 'test_control_eval.csv'), sep='\t', encoding='utf-8')
    #     df_dict = {"B": 'Background', "C": 'Explanation', "D": 'Definition', "E": 'Elaboration', "F": 'Forward-looking',
    #             "I": 'Instantiation', "O": 'Others'}
    #     df = df.replace({"label": df_dict, "pred": df_dict})
    #     test_acc = (df.loc[df.pred == df.label].shape[0]) / df.shape[0]
    #     print(test_acc)
    #     # confusion_matrix = pd.crosstab(df['label'], df['pred'], rownames=['Actual'], colnames=['Predicted'])
    #
    #     try:
    #         final_df
    #         final_df = final_df.append(df[['label', 'pred']])
    #     except NameError:
    #         final_df = df[['label', 'pred']]
    #
    #     confusion_matrix = pd.crosstab(df['label'], df['pred'], rownames=['Actual'], colnames=['Predicted'])
    #
    # final_df.to_csv(os.path.join(args.data_path, 'test_control_eval.csv'), index=False, sep='\t', encoding='utf-8')
    final_df = pd.read_csv(os.path.join(args.data_path, 'test_control_eval.csv'), sep='\t', encoding='utf-8')
    confusion_matrix = pd.crosstab(final_df['label'], final_df['pred'], rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)
    new_orders = ['Explanation', 'Elaboration', 'Background', 'Definition', 'Instantiation', 'Forward']
    confusion_matrix = confusion_matrix.reindex(new_orders)[new_orders[:-1]]
    confusion_matrix = confusion_matrix.rename({'Forward': 'F', 'Explanation': 'C',
                                                'Elaboration': 'E', 'Background': 'B',
                                                'Definition': 'D', 'Instantiation': 'I'}, axis='index')
    confusion_matrix = confusion_matrix.rename({'Forward': 'F', 'Explanation': 'C',
                                                'Elaboration': 'E', 'Background': 'B',
                                                'Definition': 'D', 'Instantiation': 'I'}, axis=1)
    plt.clf()

    mycmap = sns.light_palette("seagreen", as_cmap=True)
    res = sns.heatmap(confusion_matrix.T, annot=True, fmt='.0f', cmap=mycmap, cbar=False)

    plt.tick_params(axis='x', labelrotation=0) #, labelsize=8)
    plt.tick_params(axis='y', labelrotation=0) #, labelsize=8)
    plot_name = os.path.basename(args.data_path)
    plt.savefig("control/" + plot_name + ".png", bbox_inches='tight', dpi=100)

    plt.show(bbox_inches='tight')

