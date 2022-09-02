'''
This script aggregates below metrics for question generation evaluation: 
Bleu, Meteor, Rouge (from nqg)
BERTScore (from BERTScore)

how to install bert-score:
    pip install bert-score

how to use:
    python aqg_evaluation.py -src <source file> -tgt <target file> -out <output/predicted file> -m <measurement>
'''
import os
import sys
import subprocess
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-src", "--src_file", dest="src_file", help="The source file that contains the contexts questions are generated from.")
parser.add_argument("-tgt", "--tgt_file", dest="tgt_file", help="The target file that contains the questions.")
parser.add_argument("-pred", "--pred_file", dest="out_file", help="The output file that contains the questions predicted by the modol and whose quality is to be measured.")
parser.add_argument("-m", "--measurement", dest="measurement", default="all", choices=['bleu', 'meteor', 'rouge', 'bert', 'all'], help='Choose one of the metrics you want to use to evaluate the prediction, or "all" to use all metrics.')
parser.add_argument('-qsub_script', dest='qsub_script', default='/home/nlp-text/dynamic/lgao/util/qsub_BERTScore.txt', help='The full path to store a qsub script for BERTScore.')
args = parser.parse_args()

sys.path.append('/home/nlp-text/dynamic/mzhao/AQG/nqg/qgevalcap')
from eval import eval, QGEvalCap

if args.measurement != 'all':
    measurement = [args.measurement]
else:
    measurement = ['bleu', 'meteor', 'rouge']

if 'bert' in measurement:
    cmd = '#!/bin/bash\n\n\
#$ -S /bin/bash\n\
#$ -j y\n\
#$ -m n\n\
#$ -N bert_score\n\
#$ -q gpu.q\n\
#$ -l gpu=1\n\
#$ -l h=vor\n\
#$ -o {0}\n\n\
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /home/conda/mzhao/envs/aqg/bin/bert-score -r {1} -c {2} --lang en --rescale_with_baseline'.format(os.path.abspath(os.path.dirname(args.qsub_script)), args.tgt_file, args.out_file)
    with open(args.qsub_script, 'w') as output:
        output.write(cmd)
    print('*'*12+'\ninstructions\n'+'*'*12)
    print('{} is generated. You need to qsub it to a gpu node to calculate BERTScore. A log file containing BERTScore results will be created by the grid engine in the same directory.\n'.format(os.path.abspath(args.qsub_script)))
    measurement.pop()
if measurement:
    gts, res = eval(args.out_file, args.src_file, args.tgt_file)
    QGEval = QGEvalCap(gts, res, args.out_file, args.tgt_file, measurement)
    QGEval.evaluate()
