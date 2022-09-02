"""
prepare data for ranking
Use Roberta for regression
"""

import re
import os
import pandas as pd
from data_utils import clean_unnecessary_spaces

def sep_data(sep_df):
    sep_dev = sep_df.sample(frac=0.1, random_state=1)
    sep_train = sep_df.drop(sep_dev.index)
    return sep_dev, sep_train


if __name__ == "__main__":
    df_1 = pd.read_excel('/home-nfs/users/data/inquisitive/ranking/'
                         'context_source_span_qtype_output_50_HM.xlsx').rename(columns={"Definition (D)": "D",
                                                                                        "Background (B)": "B",
                                                                                        "Instantiation (I)": "I",
                                                                                        "Explanation ( C )": "C",
                                                                                        "Forward (F)": "F",
                                                                                        "Elaboration ( E )": "E"})
    df_2 = pd.read_excel('/home-nfs/users/data/inquisitive/ranking/'
                         'context_source_span_qtype_output_250_HM.xlsx').rename(columns={"Definition": "D",
                                                                                        "Background": "B",
                                                                                        "Instantiation": "I",
                                                                                        "Explanation ( C )": "C",
                                                                                        "Forward": "F",
                                                                                        "Elaboration ( E ) ": "E"})
    df = df_1.append(df_2, ignore_index=True)
    # generate data
    output_df = pd.DataFrame(columns=('source', 'q1', 'q2'))
    for row in range(0, 300):
        rank_num = re.sub('[^0-9]+', '', df['Rank'][row])
        rank_str = re.sub('[^a-zA-Z]+', '', df['Rank'][row])
        source = clean_unnecessary_spaces(df.at[row, 'source'])
        question_pool = ['D', 'B', 'I', 'C', 'F', 'E']

        ##  method 1
        # for i, rank in enumerate(rank_num):
        #     q1 = df.at[row, rank_str[i]]
        #     if rank_str[i] in question_pool:
        #         question_pool.remove(rank_str[i])
        #     j = i + 1
        #     while(j < len(rank_num)):
        #         if rank == rank_num[j]:
        #             if rank_str[j] in question_pool:
        #                 question_pool.remove(rank_str[j])
        #             j += 1
        #         else:
        #             break
        #     for qk in question_pool:
        #         q2 = df.at[row, qk]
        #         output_df = output_df.append(pd.Series({'source': source, 'q1': q1, 'q2': q2}), ignore_index=True)

        ## method 2
        # rank_num = re.sub('[^0-3]+', '', df['Rank'][row])
        # rank_str = rank_str[:len(rank_num)]
        for rank in rank_str:
            question_pool.remove(rank)
        for rank in rank_str:
            for other in question_pool:
                q1 = df.at[row, rank]
                q2 = df.at[row, other]
                output_df = output_df.append(pd.Series({'source': source, 'q1': q1, 'q2': q2}), ignore_index=True)


        # ## method 3, modify
        # for i, rank in enumerate(rank_num):
        #     q1 = df.at[row, rank_str[i]]
        #     if rank_str[i] in question_pool:
        #         question_pool.remove(rank_str[i])
        #     j = i + 1
        #     while(j < len(rank_num)):
        #         if int(rank) + 1 >= int(rank_num[j]):
        #             if rank_str[j] in question_pool:
        #                 question_pool.remove(rank_str[j])
        #             j += 1
        #         else:
        #             break
        #     for qk in question_pool:
        #         q2 = df.at[row, qk]
        #         output_df = output_df.append(pd.Series({'source': source, 'q1': q1, 'q2': q2}), ignore_index=True)

    # sep train and dev, make sure label distribution similar
    output_df = output_df.drop_duplicates().reset_index(drop=True)
    output_df['label'] = 1

    # train = output_df[:-286].sample(frac=1, random_state=1)   # this is for high rank vs. low rank (rank diff >= 2)
    # dev = output_df[-286:].sample(frac=1, random_state=1)
    # train = output_df[:].sample(frac=1, random_state=1) # this is for top 3
    # dev = output_df[-:].sample(frac=1, random_state=1)
    train = output_df[:-257].sample(frac=1, random_state=1) # this is for binary (selected vs. non selected)
    dev = output_df[-257:].sample(frac=1, random_state=1)
    # train = output_df[:-].sample(frac=1, random_state=1) # this is for high rank vs. low rank
    # dev = output_df[-:].sample(frac=1, random_state=1)

    # create data
    train_idx = train.sample(frac=0.5, random_state=1).index
    train.loc[train_idx, ['q1', 'q2']] = train.loc[train_idx, ['q2', 'q1']].values
    train.loc[train_idx, 'label'] = 0
    train['source'] = train['source'] + ' [SEP] ' + train['q1'] + ' [SEP] ' + train['q2']
    dev_idx = dev.sample(frac=0.5, random_state=1).index
    dev.loc[dev_idx, ['q1', 'q2']] = dev.loc[dev_idx, ['q2', 'q1']].values
    dev.loc[dev_idx, 'label'] = 0
    dev['source'] = dev['source'] + ' [SEP] ' + dev['q1'] + ' [SEP] ' + dev['q2']

    # write data for the classifier
    df_list = [train, dev]
    output_path = os.path.join('/home-nfs/users/data/inquisitive/ranking/HM', 'source_q1_q2_binary')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for idx, file_name in enumerate(['train', 'dev']):
        df = df_list[idx]
        df["source"] = df["source"].astype(str)
        df["label"] = df["label"].astype(str)
        src_file = os.path.join(output_path, file_name + '.input')
        tgt_file = os.path.join(output_path, file_name + '.label')
        with open(src_file, 'w+', encoding='utf-8') as output1, open(tgt_file, 'w+', encoding='utf-8') as output2:
            output1.write('\n'.join(df['source'].tolist()))
            output2.write('\n'.join(df['label'].tolist()))
