#!/usr/bin/env python
__author__ = 'xinya'

from .meteor.meteor import Meteor
from .bleu.bleu import Bleu
from .rouge.rouge import Rouge
from .cider.cider import Cider
from collections import defaultdict
from argparse import ArgumentParser

import sys
import imp
from nltk.tokenize import word_tokenize
imp.reload(sys)
#sys.setdefaultencoding('utf-8')

def tokenize_str(input_str):
    return ' '.join(word_tokenize(input_str))

def eval(out_file, src_file, tgt_file, isDIn = False, num_pairs = 500):
    """
        Given a filename, calculate the metric scores for that prediction file

        isDin: boolean value to check whether input file is DirectIn.txt
    """

    pairs = []
    with open(src_file, 'r') as infile:
        for line in infile:
            pair = {}
            pair['tokenized_sentence'] = tokenize_str(line.strip())
            pairs.append(pair)

    with open(tgt_file, "r") as infile:
        cnt = 0
        for line in infile:
            pairs[cnt]['tokenized_question'] = tokenize_str(line.strip())
            cnt += 1

    output = []
    with open(out_file, 'r') as infile:
        for line in infile:
            line = tokenize_str(line.strip())
            output.append(line)


    for idx, pair in enumerate(pairs):
        pair['prediction'] = output[idx]


    ## eval
    #from eval import QGEvalCap
    import json
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])
    for i, pair in enumerate(pairs[:]):
        # key = pair['tokenized_sentence']
        # #res[key] = [pair['prediction'].encode('utf-8')]
        # res[key] = [pair['prediction']]
        #
        # ## gts
        # #gts[key].append(pair['tokenized_question'].encode('utf-8'))
        # gts[key].append(pair['tokenized_question'])
        res[i] = [pair['prediction']]
        gts[i] = [pair['tokenized_question']]
    return gts, res

class QGEvalCap:
    def __init__(self, gts, res, out_file, tgt_file, measurement):
        self.gts = gts
        self.res = res
        self.out_file = out_file
        self.tgt_file = tgt_file
        self.measurement = measurement

    def evaluate(self):
        idx = {'bleu':0, 'meteor':1, 'rouge':2}
        output = []
        scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(), "METEOR"),
                (Rouge(), "ROUGE_L")
            ]
        idx = [idx[m] for m in self.measurement]
        scorers = [scorers[i] for i in idx]
        

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.5f"%(m, sc))
                    output.append(sc)
            else:
                print("%s: %0.5f"%(method, score))
                output.append(score)
        #return output


'''
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-out", "--out_file", dest="out_file", default="./output/pred.txt", help="output file to compare")
    #parser.add_argument("-out", "--out_file", dest="out_file", default="/home/nlp-text/dynamic/mzhao/AQG/pytorch-seq2seq/experiment/checkpoints/2021_03_03_21_43_22_batch32_epoch50_attn/test_result.txt", help="output file to compare")
    parser.add_argument("-src", "--src_file", dest="src_file", default="/home/nlp-text/dynamic/mzhao/AQG/nqg/data/processed/src-test.txt", help="src file")
    parser.add_argument("-tgt", "--tgt_file", dest="tgt_file", default="/home/nlp-text/dynamic/mzhao/AQG/nqg/data/processed/tgt-test.txt", help="target file")
    args = parser.parse_args()

    print("scores: \n")
    
'''

