# Type-controlled Inquistive Question Generation
## Introduction
This repository contains data and code used in the paper [“What makes a question inquisitive?” A Study on Type-Controlled Inquisitive Question Generation](https://aclanthology.org/2022.starsem-1.22.pdf). The original dataset of inquisitive questions are collected from Ko et al. (2020). In our research we added annotations of question types. 
## Data
Inside the *data* directory,
- The question_types folder contains expert annotations for question types (e.g., background, elaboration, causal, etc.).
- The annotated_questions folder contains ``silver labels'' of all the questions used. 
We trained a RoBERTa model on the expert annotations to create the silver labels. For more information, please see the following.

## Code
Inside the *code* directory, mostly are for processing data and evaluate model generations.
`check_types` contains files to analyze inquisitive v.s. informative classifier. 
`evaluation`, `utils` and `evaluation.py` are used for evaluations.
We didn't include below repositories:
1. [fairseq](https://github.com/pytorch/fairseq), which are used for BART model and RoBERTa model.
2. [BERTScore](https://github.com/Tiiiger/bert_score), which is used for evaluation.

### Script Example
BPE encode
```
data_direct=context_source_span_q
for SPLIT in train dev; do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "/home-nfs/lygao/data/inquisitive/annotations/question_type/$data_direct/$SPLIT.input" \
        --outputs "/home-nfs/lygao/data/inquisitive/annotations/question_type/$data_direct/$SPLIT.bpe" \
        --workers 60 \
        --keep-empty
done
```
Preprocess data
```
fairseq-preprocess \
    --only-source \
    --trainpref "/home-nfs/lygao/data/inquisitive/annotations/question_type/$data_direct/train.bpe" \
    --validpref "/home-nfs/lygao/data/inquisitive/annotations/question_type/$data_direct/dev.bpe" \
    --destdir "/home-nfs/lygao/data/inquisitive/annotations/question_type/$data_direct/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "/home-nfs/lygao/data/inquisitive/annotations/question_type/$data_direct/train.label" \
    --validpref "/home-nfs/lygao/data/inquisitive/annotations/question_type/$data_direct/dev.label" \
    --destdir "/home-nfs/lygao/data/inquisitive/annotations/question_type/$data_direct/label" \
    --workers 60
```
Example scripts for classification of question types
```
TOTAL_NUM_UPDATES=2625  # 15 epochs for bsz 8
WARMUP_UPDATES=157      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
HEAD_NAME=qtype_head     # Custom name for the classification head.
NUM_CLASSES=7           # Number of classes for the classification task.
MAX_SENTENCES=2         # Batch size.
data_direct=all_questions
ROBERTA_PATH=/home-nfs/lygao/project/inquisitive/roberta.large/model.pt
datadir=/home-nfs/lygao/data/inquisitive/annotations/question_type/$data_direct/
savedir=/home-nfs/lygao/output/inquisitive/question_type/$data_direct/

python /home-nfs/lygao/project/inquisitive/fairseq/train.py $datadir \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 15 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --truncate-sequence  \
    --find-unused-parameters \
    --update-freq 4 \
    --save-dir $savedir
```

Example scripts for training of TYPE model
```
dir=/home-nfs/lygao/data/inquisitive/fairseq/context_source_span_qtype
savedir=/home-nfs/lygao/output/inquisitive/fairseq/context_source_span_qtype
BART_PATH=/home-nfs/lygao/project/inquisitive/bart.large/model.pt

python /home-nfs/lygao/project/inquisitive/fairseq/train.py $dir \
    --restore-file $BART_PATH \
    --max-tokens 1024 \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --truncate-target \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay \
    --lr 3e-05 --total-num-update 20000 \
    --warmup-updates 500 \
    --memory-efficient-fp16 --update-freq 16 \
    --save-dir $savedir \
    --ddp-backend=no_c10d  \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --max-epoch 15 \
    --tensorboard-logdir $savedir
```

## Citation
```
@inproceedings{gao-etal-2022-makes,
    title = "{``}What makes a question inquisitive?{''} A Study on Type-Controlled Inquisitive Question Generation",
    author = "Gao, Lingyu  and
      Ghosh, Debanjan  and
      Gimpel, Kevin",
    booktitle = "Proceedings of the 11th Joint Conference on Lexical and Computational Semantics",
    month = jul,
    year = "2022",
    address = "Seattle, Washington",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.starsem-1.22",
    pages = "240--257"
}
```
