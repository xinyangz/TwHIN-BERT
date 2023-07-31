# TwHIN-BERT: A Socially-Enriched Pre-trained Language Model for Multilingual Tweet Representations
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-green.svg?style=flat-square)](https://github.com/xinyangz/TwHIN-BERT/pulls)
[![arXiv](https://img.shields.io/badge/arXiv-2203.15827-b31b1b.svg)](https://arxiv.org/abs/2209.07562)
[![Huggingface-base](https://img.shields.io/badge/HuggingFace-twhin--bert--base-yellow)](https://huggingface.co/Twitter/twhin-bert-base)
[![Huggingface-large](https://img.shields.io/badge/HuggingFace-twhin--bert--large-yellow)](https://huggingface.co/Twitter/twhin-bert-large)


This repo contains models, code and pointers to datasets from our paper: [TwHIN-BERT: A Socially-Enriched Pre-trained Language Model for Multilingual Tweet Representations](https://arxiv.org/abs/2209.07562).
[[PDF]](https://arxiv.org/pdf/2209.07562.pdf)
[[HuggingFace Models]](https://huggingface.co/Twitter)
[[Video]](https://www.youtube.com/watch?v=bjpq1Y4obi80)

### Overview
TwHIN-BERT is a new multi-lingual Tweet language model that is trained on 7 billion Tweets from over 100 distinct languages. TwHIN-BERT differs from prior pre-trained language models as it is trained with not only text-based self-supervision (e.g., MLM), but also with a social objective based on the rich social engagements within a Twitter Heterogeneous Information Network (TwHIN).

TwHIN-BERT can be used as a drop-in replacement for BERT in a variety of NLP and recommendation tasks. It not only outperforms similar models semantic understanding tasks such text classification), but also **social recommendation **tasks such as predicting user to Tweet engagement.

## 1. Pretrained Models

We initially release two pretrained TwHIN-BERT models (base and large) that are compatible wit the [HuggingFace BERT models](https://github.com/huggingface/transformers).


| Model | Size | Download Link (🤗 HuggingFace) |
| ------------- | ------------- | --------- |
| TwHIN-BERT-base   | 280M parameters | [Twitter/TwHIN-BERT-base](https://huggingface.co/Twitter/twhin-bert-base) |
| TwHIN-BERT-large  | 550M parameters | [Twitter/TwHIN-BERT-large](https://huggingface.co/Twitter/twhin-bert-large) |


To use these models in 🤗 Transformers:
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('Twitter/twhin-bert-base')
model = AutoModel.from_pretrained('Twitter/twhin-bert-base')
inputs = tokenizer("I'm using TwHIN-BERT! #TwHIN-BERT #NLP", return_tensors="pt")
outputs = model(**inputs)
```



## 2. Benchmark Datasets
The datasets are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

### 2.1 Multilingual Hashtag Prediction
Please check the official dataset repo on HuggingFace ([link](https://huggingface.co/datasets/Twitter/HashtagPrediction)) for dataset description and download.

A hydrated version of the dataset can be downloaded [here](https://www.dropbox.com/s/wnfgz2ry369r6ps/hashtag-classification.zip?dl=0). You must follow Twitter's term of service if using the hydrated dataset.

### 2.2 Engagement Prediction
A hydrated version of the dataset can be downloaded [here](https://www.dropbox.com/s/7fnbaenl11j0yuf/engagement-dataset.zip?dl=0). You must follow Twitter's term of service if using the hydrated dataset.


## Citation
If you use TwHIN-BERT or out datasets in your work, please cite the following:
```bib
@article{zhang2022twhin,
  title={TwHIN-BERT: A Socially-Enriched Pre-trained Language Model for Multilingual Tweet Representations},
  author={Zhang, Xinyang and Malkov, Yury and Florez, Omar and Park, Serim and McWilliams, Brian and Han, Jiawei and El-Kishky, Ahmed},
  journal={arXiv preprint arXiv:2209.07562},
  year={2022}
}
```
