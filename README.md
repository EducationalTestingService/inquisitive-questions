# Inquisitive Question Generation
Repository for research on Inquisitive Question Generation
## Introduction
This repository contains data and code used in the paper [“What makes a question inquisitive?” A Study on Type-Controlled Inquisitive Question Generation](https://aclanthology.org/2022.starsem-1.22.pdf). The original dataset of inquisitive questions are collected from Ko et al. (2020). In our research we added annotations of question types. 
### Data description

The question_types folder contains expert annotations for question types (e.g., background, elaboration, causal, etc.).
The annotated_questions folder contains ``silver labels'' of all the questions used. We trained a RoBERTa model on the expert annotations to create the silver labels. For more information, please see the following.
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
