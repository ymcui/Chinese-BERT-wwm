[**中文说明**](https://github.com/ymcui/Chinese-BERT-wwm/) | [**English**](https://github.com/ymcui/Chinese-BERT-wwm/blob/master/README_EN.md)

## Chinese BERT with Whole Word Masking
For further accelerating Chinese natural language processing, we provide **Chinese pre-trained BERT with Whole Word Masking**. Meanwhile, we also compare the state-of-the-art Chinese pre-trained models in depth, including [BERT](https://github.com/google-research/bert)、[ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)、[BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm).

- **[Pre-Training with Whole Word Masking for Chinese BERT](https://ieeexplore.ieee.org/document/9599397)**  
- Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Ziqing Yang
- Published in *IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)*

This repository is developed based on：https://github.com/google-research/bert

----

[Chinese LERT](https://github.com/ymcui/LERT) | [Chinese/English PERT](https://github.com/ymcui/PERT) [Chinese MacBERT](https://github.com/ymcui/MacBERT) | [Chinese ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [Chinese XLNet](https://github.com/ymcui/Chinese-XLNet) | [Chinese BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [TextBrewer](https://github.com/airaria/TextBrewer) | [TextPruner](https://github.com/airaria/TextPruner)

More resources by HFL: https://github.com/ymcui/HFL-Anthology

## News
**Mar 28, 2023 We open-sourced Chinese LLaMA&Alpaca LLMs, which can be quickly deployed on PC. Check: https://github.com/ymcui/Chinese-LLaMA-Alpaca**

2022/10/29 We release a new pre-trained model called LERT, check https://github.com/ymcui/LERT/

2022/3/30 We release a new pre-trained model called PERT, check https://github.com/ymcui/PERT 

2021/12/17 We release a model pruning toolkit - TextPruner, check https://github.com/airaria/TextPruner

2021/1/27 All models support TensorFlow 2 now. Please use transformers library to access them or download from https://huggingface.co/hfl

2020/9/15 Our paper ["Revisiting Pre-Trained Models for Chinese Natural Language Processing"](https://arxiv.org/abs/2004.13922) is accepted to [Findings of EMNLP](https://2020.emnlp.org) as a long paper.

2020/8/27 We are happy to announce that our model is on top of GLUE benchmark, check [leaderboard](https://gluebenchmark.com/leaderboard).

<details>
<summary>Past News</summary>
2020/3/23 The models in this repository now can be easily accessed through [PaddleHub](https://github.com/PaddlePaddle/PaddleHub), check [Quick Load](#Quick-Load)

2020/2/26 We release a knowledge distillation toolkit [TextBrewer](https://github.com/airaria/TextBrewer)

2020/1/20 Happy Chinese New Year! We've released RBT3 and RBTL3 (3-layer RoBERTa-wwm-ext-base/large), check [Small Models](#Small-Models)

2019/12/19 The models in this repository now can be easily accessed through [Huggingface-Transformers](https://github.com/huggingface/transformers), check [Quick Load](#Quick-Load)

2019/10/14 We release `RoBERTa-wwm-ext-large`, check [Download](#Download)

2019/9/10 We release `RoBERTa-wwm-ext`, check [Download](#Download)

2019/7/30 We release `BERT-wwm-ext`, which was trained on larger data, check [Download](#Download)

2019/6/20 Initial version, pre-trained models could be downloaded through Google Drive, check [Download](#Download)
</details>

## Guide
| Section | Description |
|-|-|
| [Introduction](#Introduction) | Introduction to BERT with Whole Word Masking (WWM) |
| [Download](#Download) | Download links for Chinese BERT-wwm |
| [Quick Load](#Quick-Load) | Learn how to quickly load our models through [🤗Transformers](https://github.com/huggingface/transformers) or [PaddleHub](https://github.com/PaddlePaddle/PaddleHub) |
| [Model Comparison](#Model-Comparison) | Compare the models published in this repository |
| [Baselines](#Baselines) | Baseline results for several Chinese NLP datasets (partial) |
| [Small Models](#Small-Models) | 3-layer Transformer models |
| [Useful Tips](#Useful-Tips) | Provide several useful tips for using Chinese pre-trained models |
| [English BERT-wwm](#English-BERT-wwm) | Download English BERT-wwm (by Google) |
| [FAQ](#FAQ) | Frequently Asked Questions |
| [Citation](#Citation) | Citation |


## Introduction
**Whole Word Masking (wwm)** is an upgraded version by [BERT](https://github.com/google-research/bert) released on late May 2019.

The following introductions are copied from BERT repository.
```
In the original pre-processing code, we randomly select WordPiece tokens to mask. For example:

Input Text: the man jumped up , put his basket on phil ##am ##mon ' s head 

Original Masked Input: [MASK] man [MASK] up , put his [MASK] on phil [MASK] ##mon ' s head

The new technique is called Whole Word Masking. In this case, we always mask all of the the tokens corresponding to a word at once. The overall masking rate remains the same.

Whole Word Masked Input: the man [MASK] up , put his basket on [MASK] [MASK] [MASK] ' s head

The training is identical -- we still predict each masked WordPiece token independently. The improvement comes from the fact that the original prediction task was too 'easy' for words that had been split into multiple WordPieces.

```

**Important Note: Terminology `Masking` does not ONLY represent replace a word into `[MASK]` token.
It could also be in another form, such as `keep original word` or `randomly replaced by another word`.**

In the Chinese language, it is straightforward to utilize whole word masking, as traditional text processing in Chinese should include `Chinese Word Segmentation (CWS)`.
In the original `BERT-base, Chinese` by Google, the segmentation is done by splitting the Chinese characters while neglecting the importance of CWS.
In this repository, we utilize [Language Technology Platform (LTP)](http://ltp.ai) by Harbin Institute of Technology for CWS, and adapt whole word masking in Chinese text.


## Download
As all models are 'BERT-base' variants, we do not incidate 'base' in the following model names.

* **`BERT-base`**：12-layer, 768-hidden, 12-heads, 110M parameters

| Model | Data | Google Drive | iFLYTEK Cloud |
| :------- | :--------- | :---------: | :---------: |
| **`RBT6, Chinese`** | **Wikipedia+Extended data<sup>[1]</sup>** | - | **[TensorFlow（pw:hniy）](https://pan.baidu.com/s/1_MDAIYIGVgDovWkSs51NDA?pwd=hniy)** |
| **`RBT4, Chinese`** | **Wikipedia+Extended data<sup>[1]</sup>** | - | **[TensorFlow（pw:sjpt）](https://pan.baidu.com/s/1MUrmuTULnMn3L1aw_dXxSA?pwd=sjpt)** |
| **`RBTL3, Chinese`** | **Wikipedia+Extended data<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1Jzn1hYwmv0kXkfTeIvNT61Rn1IbRc-o8)**<br/>**[PyTorch](https://drive.google.com/open?id=1qs5OasLXXjOnR2XuGUh12NanUl0pkjEv)** | **[TensorFlow（pw:s6cu）](https://pan.baidu.com/s/1vV9ClBMbsSpt8wUpfQz62Q?pwd=s6cu)** |
| **`RBT3, Chinese`** | **Wikipedia+Extended data<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1-rvV0nBDvRCASbRz8M9Decc3_8Aw-2yi)**<br/>**[PyTorch](https://drive.google.com/open?id=1_LqmIxm8Nz1Abvlqb8QFZaxYo-TInOed)** | **[TensorFlow（pw:5a57）](https://pan.baidu.com/s/1AnapwWj1YBZ_4E6AAtj2lg?pwd=5a57)** |
| **`RoBERTa-wwm-ext-large, Chinese`** | **Wikipedia+Extended data<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1dtad0FFzG11CBsawu8hvwwzU2R0FDI94)**<br/>**[PyTorch](https://drive.google.com/open?id=1-2vEZfIFCdM1-vJ3GD6DlSyKT4eVXMKq)** | **[TensorFlow（pw:dqqe）](https://pan.baidu.com/s/1F68xzCLWEonTEVP7HQ0Ciw?pwd=dqqe)** |
| **`RoBERTa-wwm-ext, Chinese`** | **Wikipedia+Extended data<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1jMAKIJmPn7kADgD3yQZhpsqM-IRM1qZt)** <br/>**[PyTorch](https://drive.google.com/open?id=1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25)** | **[TensorFlow（pw:vybq）](https://pan.baidu.com/s/1oR0cgSXE3Nz6dESxr98qVA?pwd=vybq)** |
| **`BERT-wwm-ext, Chinese`** | **Wikipedia+Extended data<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1buMLEjdtrXE2c4G1rpsNGWEx7lUQ0RHi)** <br/>**[PyTorch](https://drive.google.com/open?id=1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_)** | **[TensorFlow（pw:wgnt）](https://pan.baidu.com/s/1x-jIw1X2yNYHGak2yiq4RQ?pwd=wgnt)** |
| **`BERT-wwm, Chinese`** | **Wikipedia** | **[TensorFlow](https://drive.google.com/open?id=1RoTQsXp2hkQ1gSRVylRIJfQxJUgkfJMW)** <br/>**[PyTorch](https://drive.google.com/open?id=1AQitrjbvCWc51SYiLN-cJq4e0WiNN4KY)** | **[TensorFlow（pw:qfh8）](https://pan.baidu.com/s/1HDdDXiYxGT5ub5OeO7qdWw?pwd=qfh8)** |
| `BERT-base, Chinese`<sup>Google</sup> | Wikipedia | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) | - |
| `BERT-base, Multilingual Cased`<sup>Google</sup>  | Wikipedia | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) | - |
| `BERT-base, Multilingual Uncased`<sup>Google</sup>  | Wikipedia | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip) | - |

### PyTorch Version

If you need these models in PyTorch,

1) Convert TensorFlow checkpoint into PyTorch, using [🤗Transformers](https://github.com/huggingface/transformers)

2) Download from https://huggingface.co/hfl

Steps: select one of the model in the page above → click "list all files in model" at the end of the model page → download bin/json files from the pop-up window

### Note

The whole zip package roughly takes ~400M.
ZIP package includes the following files:

```
chinese_wwm_L-12_H-768_A-12.zip
    |- bert_model.ckpt      # Model Weights
    |- bert_model.meta      # Meta info
    |- bert_model.index     # Index info
    |- bert_config.json     # Config file
    |- vocab.txt            # Vocabulary
```

`bert_config.json` and `vocab.txt` are identical to the original **`BERT-base, Chinese`** by Google。


## Quick Load
### Huggingface-Transformers

With [Huggingface-Transformers](https://github.com/huggingface/transformers), the models above could be easily accessed and loaded through the following codes.
```
tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BertModel.from_pretrained("MODEL_NAME")
```
**Notice: Please use BertTokenizer and BertModel for loading these model. DO NOT use RobertaTokenizer/RobertaModel!**

The actual model and its `MODEL_NAME` are listed below.

| Original Model | MODEL_NAME |
| - | - |
| RoBERTa-wwm-ext-large | hfl/chinese-roberta-wwm-ext-large |
| RoBERTa-wwm-ext | hfl/chinese-roberta-wwm-ext |
| BERT-wwm-ext | hfl/chinese-bert-wwm-ext |
| BERT-wwm | hfl/chinese-bert-wwm |
| RBT3 | hfl/rbt3 |
| RBTL3 | hfl/rbtl3 |

### PaddleHub

With [PaddleHub](https://github.com/PaddlePaddle/PaddleHub), we can download and install the model with one line of code.

```
import paddlehub as hub
module = hub.Module(name=MODULE_NAME)
```

The actual model and its `MODULE_NAME` are listed below.

| Original Model | MODULE_NAME |
| - | - |
| RoBERTa-wwm-ext-large | [chinese-roberta-wwm-ext-large](https://www.paddlepaddle.org.cn/hubdetail?name=chinese-roberta-wwm-ext-large&en_category=SemanticModel) |
| RoBERTa-wwm-ext       | [chinese-roberta-wwm-ext](https://www.paddlepaddle.org.cn/hubdetail?name=chinese-roberta-wwm-ext&en_category=SemanticModel) |
| BERT-wwm-ext          | [chinese-bert-wwm-ext](https://www.paddlepaddle.org.cn/hubdetail?name=chinese-bert-wwm-ext&en_category=SemanticModel) |
| BERT-wwm              | [chinese-bert-wwm](https://www.paddlepaddle.org.cn/hubdetail?name=chinese-bert-wwm&en_category=SemanticModel) |
| RBT3                  | [rbt3](https://www.paddlepaddle.org.cn/hubdetail?name=rbt3&en_category=SemanticModel) |
| RBTL3                 | [rbtl3](https://www.paddlepaddle.org.cn/hubdetail?name=rbtl3&en_category=SemanticModel) |


## Model Comparison
We list comparisons on the models that were released in this project.
`~BERT` means to inherit the attributes from original Google's BERT.

| - | BERT<sup>Google</sup> | BERT-wwm | BERT-wwm-ext | RoBERTa-wwm-ext | RoBERTa-wwm-ext-large |
| :------- | :---------: | :---------: | :---------: | :---------: | :---------: |
| Masking | WordPiece | WWM<sup>[1]</sup> | WWM | WWM | WWM |
| Type | BERT-base | BERT-base | BERT-base | BERT-base | **BERT-large** |
| Data Source | wiki | wiki | wiki+ext<sup>[2]</sup> | wiki+ext | wiki+ext |
| Training Tokens # | 0.4B | 0.4B | 5.4B | 5.4B | 5.4B |
| Device | TPU Pod v2 | TPU v3 | TPU v3 | TPU v3 | **TPU Pod v3-32<sup>[3]</sup>** |
| Training Steps | ? | 100K<sup>MAX128</sup> <br/>+100K<sup>MAX512</sup> | 1M<sup>MAX128</sup> <br/>+400K<sup>MAX512</sup> | 1M<sup>MAX512</sup> | 2M<sup>MAX512</sup> |
| Batch Size | ? | 2,560 / 384 | 2,560 / 384 | 384 | 512 |
| Optimizer | AdamW | LAMB | LAMB | AdamW | AdamW |
| Vocabulary | 21,128 | ~BERT<sup>[4]</sup> vocab | ~BERT vocab | ~BERT vocab | ~BERT vocab |
| Init Checkpoint | Random Init | ~BERT weight | ~BERT weight | ~BERT weight | Random Init |


## Baselines
We experiment on several Chinese datasets, including sentence-level to document-level tasks.

**We only list partial results here and kindly advise the readers to read our [technical report](https://arxiv.org/abs/1906.08101).**

Best Learning Rate:  

| Model | BERT | ERNIE | BERT-wwm* |
| :------- | :---------: | :---------: | :---------: |
| CMRC 2018 | 3e-5 | 8e-5 | 3e-5 |
| DRCD | 3e-5 | 8e-5 | 3e-5 |
| CJRC | 4e-5 | 8e-5 | 4e-5 |
| XNLI | 3e-5 | 5e-5 | 3e-5 |
| ChnSentiCorp | 2e-5 | 5e-5 | 2e-5 |
| LCQMC  | 2e-5 | 3e-5 | 2e-5 |
| BQ Corpus | 3e-5 | 5e-5 | 3e-5 |
| THUCNews | 2e-5 | 5e-5 | 2e-5 |
* represents all related models (BERT-wwm, BERT-wwm-ext, RoBERTa-wwm-ext, RoBERTa-wwm-ext-large)


- [**CMRC 2018**：Span-Extraction Machine Reading Comprehension (Simplified Chinese)](https://github.com/ymcui/cmrc2018)
- [**DRCD**：Span-Extraction Machine Reading Comprehension (Traditional Chinese)](https://github.com/DRCSolutionService/DRCD)
- [**CJRC**: Chinese Judiciary Reading Comprehension](http://cail.cipsc.org.cn)
- [**XNLI**：Natural Langauge Inference](https://github.com/google-research/bert/blob/master/multilingual.md)
- [**ChnSentiCorp**：Sentiment Analysis](https://github.com/pengming617/bert_classification)
- [**LCQMC**：Sentence Pair Matching](http://icrc.hitsz.edu.cn/info/1037/1146.htm)
- [**BQ Corpus**：Sentence Pair Matching](http://icrc.hitsz.edu.cn/Article/show/175.html)
- [**THUCNews**：Document-level Text Classification](http://thuctc.thunlp.org)

**Note: To ensure the stability of the results, we run 10 times for each experiment and report maximum and average scores.**

**Average scores are in brackets, and max performances are the numbers that out of brackets.**

### [CMRC 2018](https://github.com/ymcui/cmrc2018)
CMRC 2018 dataset is released by Joint Laboratory of HIT and iFLYTEK Research.
The model should answer the questions based on the given passage, which is identical to SQuAD.
Evaluation Metrics: EM / F1

| Model | Development | Test | Challenge |
| :------- | :---------: | :---------: | :---------: |
| BERT | 65.5 (64.4) / 84.5 (84.0) | 70.0 (68.7) / 87.0 (86.3) | 18.6 (17.0) / 43.3 (41.3) |
| ERNIE | 65.4 (64.3) / 84.7 (84.2) | 69.4 (68.2) / 86.6 (86.1) | 19.6 (17.0) / 44.3 (42.8) |
| **BERT-wwm** | 66.3 (65.0) / 85.6 (84.7) | 70.5 (69.1) / 87.4 (86.7) | 21.0 (19.3) / 47.0 (43.9) |
| **BERT-wwm-ext** | 67.1 (65.6) / 85.7 (85.0) | 71.4 (70.0) / 87.7 (87.0) | 24.0 (20.0) / 47.3 (44.6) |
| **RoBERTa-wwm-ext** | 67.4 (66.5) / 87.2 (86.5) | 72.6 (71.4) / 89.4 (88.8) | 26.2 (24.6) / 51.0 (49.1) |
| **RoBERTa-wwm-ext-large** | **68.5 (67.6) / 88.4 (87.9)** | **74.2 (72.4) / 90.6 (90.0)** | **31.5 (30.1) / 60.1 (57.5)** |


### [DRCD](https://github.com/DRCKnowledgeTeam/DRCD)
DRCD is also a span-extraction machine reading comprehension dataset, released by Delta Research Center. The text is written in Traditional Chinese.
Evaluation Metrics: EM / F1

| Model | Development | Test |
| :------- | :---------: | :---------: |
| BERT | 83.1 (82.7) / 89.9 (89.6) | 82.2 (81.6) / 89.2 (88.8) |
| ERNIE | 73.2 (73.0) / 83.9 (83.8) | 71.9 (71.4) / 82.5 (82.3) |
| **BERT-wwm** | 84.3 (83.4) / 90.5 (90.2) | 82.8 (81.8) / 89.7 (89.0) |
| **BERT-wwm-ext** | 85.0 (84.5) / 91.2 (90.9) | 83.6 (83.0) / 90.4 (89.9) |
| **RoBERTa-wwm-ext** | 86.6 (85.9) / 92.5 (92.2) | 85.6 (85.2) / 92.0 (91.7) |
| **RoBERTa-wwm-ext-large** | **89.6 (89.1) / 94.8 (94.4)** | **89.6 (88.9) / 94.5 (94.1)** |


### CJRC
[**CJRC**](http://cail.cipsc.org.cn) is a Chinese judiciary reading comprehension dataset, released by Joint Laboratory of HIT and iFLYTEK Research. Note that, the data used in these experiments are NOT identical to the official one.
Evaluation Metrics: EM / F1

| Model | Development | Test |
| :------- | :---------: | :---------: |
| BERT | 54.6 (54.0) / 75.4 (74.5) | 55.1 (54.1) / 75.2 (74.3) |
| ERNIE | 54.3 (53.9) / 75.3 (74.6) | 55.0 (53.9) / 75.0 (73.9) |
| **BERT-wwm** | 54.7 (54.0) / 75.2 (74.8) | 55.1 (54.1) / 75.4 (74.4) |
| **BERT-wwm-ext** | 55.6 (54.8) / 76.0 (75.3) | 55.6 (54.9) / 75.8 (75.0) |
| **RoBERTa-wwm-ext** | 58.7 (57.6) / 79.1 (78.3) | 59.0 (57.8) / 79.0 (78.0) |
| **RoBERTa-wwm-ext-large** | **62.1 (61.1) / 82.4 (81.6)** | **62.4 (61.4) / 82.2 (81.0)** |


### XNLI
We use XNLI data for testing NLI task.
Evaluation Metrics: Accuracy

| Model | Development | Test |
| :------- | :---------: | :---------: |
| BERT | 77.8 (77.4) | 77.8 (77.5) |
| ERNIE | 79.7 (79.4) | 78.6 (78.2) |
| **BERT-wwm** | 79.0 (78.4) | 78.2 (78.0) |
| **BERT-wwm-ext** | 79.4 (78.6) | 78.7 (78.3) |
| **RoBERTa-wwm-ext** | 80.0 (79.2) | 78.8 (78.3) |
| **RoBERTa-wwm-ext-large** | **82.1 (81.3)** | **81.2 (80.6)** |

### ChnSentiCorp
We use ChnSentiCorp data for testing sentiment analysis.
Evaluation Metrics: Accuracy

| Model | Development | Test |
| :------- | :---------: | :---------: |
| BERT | 94.7 (94.3) | 95.0 (94.7) |
| ERNIE | 95.4 (94.8) | 95.4 **(95.3)** |
| **BERT-wwm** | 95.1 (94.5) | 95.4 (95.0) |
| **BERT-wwm-ext** | 95.4 （94.6) | 95.3 (94.7) |
| **RoBERTa-wwm-ext** | 95.0 (94.6) | 95.6 (94.8) |
| **RoBERTa-wwm-ext-large** | **95.8 (94.9)** | **95.8** (94.9) |


### Sentence Pair Matching：LCQMC, BQ Corpus

#### LCQMC
Evaluation Metrics: Accuracy

| Model | Development | Test |
| :------- | :---------: | :---------: |
| BERT | 89.4 (88.4) | 86.9 (86.4) |
| ERNIE | 89.8 (89.6) | **87.2 (87.0)** |
| **BERT-wwm** | 89.4 (89.2) | 87.0 (86.8) |
| **BERT-wwm-ext** | 89.6 (89.2) | 87.1 (86.6) |
| **RoBERTa-wwm-ext** | 89.0 (88.7) | 86.4 (86.1) |
| **RoBERTa-wwm-ext-large** | **90.4 (90.0)** | 87.0 (86.8) |

#### BQ Corpus 
Evaluation Metrics: Accuracy

| Model | Development | Test |
| :------- | :---------: | :---------: |
| BERT | 86.0 (85.5) | 84.8 (84.6) |
| ERNIE | 86.3 (85.5) | 85.0 (84.6) |
| **BERT-wwm** | 86.1 (85.6) | 85.2 **(84.9)** |
| **BERT-wwm-ext** | **86.4** (85.5) | 85.3 (84.8) |
| **RoBERTa-wwm-ext** | 86.0 (85.4) | 85.0 (84.6) |
| **RoBERTa-wwm-ext-large** | 86.3 **(85.7)** | **85.8 (84.9)** |


### THUCNews
Released by Tsinghua University, which contains news in 10 categories.
Evaluation Metrics: Accuracy

| Model | Development | Test |
| :------- | :---------: | :---------: |
| BERT | 97.7 (97.4) | 97.8 (97.6) |
| ERNIE | 97.6 (97.3) | 97.5 (97.3) |
| **BERT-wwm** | 98.0 (97.6) | 97.8 (97.6) |
| **BERT-wwm-ext** | 97.7 (97.5) | 97.7 (97.5) |
| **RoBERTa-wwm-ext** | 98.3 (97.9) | 97.7 (97.5) |
| **RoBERTa-wwm-ext-large** | 98.3 (97.7) | 97.8 (97.6) |

### Small Models
We list RBT3 and RBTL3 results on several NLP tasks. Note that, we only list test set results.

| Model | CMRC 2018 | DRCD | XNLI | CSC | LCQMC | BQ | Average | Params |
| :------- | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| RoBERTa-wwm-ext-large | 74.2 / 90.6 | 89.6 / 94.5 | 81.2 | 95.8 | 87.0 | 85.8 | 87.335 | 325M |
| RoBERTa-wwm-ext | 72.6 / 89.4 | 85.6 / 92.0 | 78.8 | 95.6 | 86.4 | 85.0 | 85.675 | 102M |
| RBTL3 | 63.3 / 83.4 | 77.2 / 85.6 | 74.0 | 94.2 | 85.1 | 83.6 | 80.800 | 61M (59.8%) |
| RBT3 | 62.2 / 81.8 | 75.0 / 83.9 | 72.3 | 92.8 | 85.1 | 83.3 | 79.550 | 38M (37.3%) |

Relative performance:

| Model | CMRC 2018 | DRCD | XNLI | CSC | LCQMC | BQ | Average | AVG-C |
| :------- | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| RoBERTa-wwm-ext-large | 102.2% / 101.3% | 104.7% / 102.7% | 103.0% | 100.2% | 100.7% | 100.9% | 101.9% | 101.2% |
| RoBERTa-wwm-ext | 100% / 100% | 100% / 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| RBTL3 | 87.2% / 93.3% | 90.2% / 93.0% | 93.9% | 98.5% | 98.5% | 98.4% | 94.3% | 97.35% |
| RBT3 | 85.7% / 91.5% | 87.6% / 91.2% | 91.8% | 97.1% | 98.5% | 98.0% | 92.9% | 96.35% |

* AVG-C: average score of classification tasks: XNLI, CSC, LCQMC, BQ

- The numbers of parameter are calculated based on XNLI classification task.
- Relative parameter percentage is calculated based on RoBERTa-wwm-ext model.
- RBT3: We use RoBERTa-wwm-ext for initializing the first three layers, and continue to train 1M steps.
- RBTL3: We use RoBERTa-wwm-ext-large for initializing the first three layers, and continue to train 1M steps.
- The name of RBT is the syllables of 'RoBERTa', and 'L' stands for large model.
- Directly using the first three layers of RoBERTa-wwm-ext-large to fine-tune the downstream task will result in a bad performance. For example, in CMRC 2018 task we could only achieve 42.9/65.3, while RBTL3 could reach 63.3/83.4.


## Useful Tips
* Initial learning rate is the most important hyper-parameters (regardless of BERT or other neural networks), and should ALWAYS be tuned for better performance.
* As shown in the experimental results, BERT and BERT-wwm share almost the same best initial learning rate, so it is straightforward to apply your initial learning rate in BERT to BERT-wwm. However, we find that ERNIE does not share the same characteristics, so it is STRONGLY recommended to tune the learning rate.
* As BERT and BERT-wwm were trained on Wikipedia data, they show relatively better performance on the formal text. While, ERNIE was trained on larger data, including web text, which will be useful on casual text, such as Weibo (microblogs).
* In long-sequence tasks, such as machine reading comprehension and document classification, we suggest using BERT or BERT-wwm.
* As these pre-trained models are trained in general domains, if the task data is extremely different from the pre-training data (Wikipedia for BERT/BERT-wwm), we suggest taking another pre-training steps on the task data, which was also suggested by Devlin et al. (2019).
* As there are so many possibilities in pre-training stage (such as initial learning rate, global training steps, warm-up steps, etc.), our implementation may not be optimal using the same pre-training data. Readers are advised to train their own model if seeking for another boost in performance. However, if it is unable to do pre-training, choose one of these pre-trained models which was trained on a similar domain to the downstream task.
* When dealing with Traditional Chinese text, use BERT or BERT-wwm.


## English BERT-wwm
We also repost English BERT-wwm (by Google official) here for your perusal.

*   **[`BERT-Large, Uncased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters

*   **[`BERT-Large, Cased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters


## FAQ
**Q: How to use this model?**  
A: Use it as if you are using original BERT. Note that, you don't need to do CWS for your text, as wwm only change the pre-training input but not the input for down-stream tasks.

**Q: Do you have any plans to release the code?**  
A: Unfortunately, I am not be able to release the code at the moment. As implementation is quite easy, I would suggest you to read #10 and #13.

**Q: How can I download XXXXX dataset?**  
A: We only provide the data that is publically available, check `data` directory. For copyright reasons, some of the datasets are not publically available. In that case, please search on GitHub or consult original authors for accessing.

**Q: How to use this model?**  
A: Use it as if you are using original BERT. Note that, you don't need to do CWS for your text, as wwm only change the pre-training input but not the input for down-stream tasks.

**Q: Do you have any plans on releasing the larger model? Say BERT-large-wwm?**  
A: If we could get significant gains from BERT-large, we will release a larger version in the future.

**Q: You lier! I can not reproduce the result! 😂**  
A: We use the simplist models in the downstream tasks. For example, in the classification task, we directly use `run_classifier.py` by Google. If you are not able to reach the average score that we reported, then there should be some bugs in your code. As there is randomness in reaching maximum scores, there is no guarantee that you will reproduce them.

**Q: I could get better performance than you!**  
A: Congratulations!

**Q: How long did it take to train such a model?**  
A: The training was done on Google Cloud TPU v3 with 128HBM, and it roughly takes 1.5 days. Note that, in the pre-training stage, we use [`LAMB Optimizer`](https://github.com/ymcui/LAMB_Optimizer_TF) which is optimized for the larger batch. In fine-tuning downstream task, we use normal `AdamWeightDecayOptimizer` as default.

**Q: Who is ERNIE?**  
A: The [ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE) in this repository refer to the model released by Baidu, but not the one that published by Tsinghua University which was also called [ERNIE](https://github.com/thunlp/ERNIE).

**Q: BERT-wwm does not perform well on some tasks.**  
A: The aim of this project is to provide researchers with a variety of pre-training models.
You are free to choose one of these models.
We only provide experimental results, and we strongly suggest trying these models in your own task.
One more model, one more choice.

**Q: Why not trying on more dataset?**  
A: To be honest: 1) no time to find more data; 2) no need; 3) no money;

**Q: Say something about these models**  
A: Each has its own emphasis and merits. Development of Chinese NLP needs joint efforts.

**Q: Any comments on the name of next generation of the pre-trained model?**  
A: Maybe ZOE: Zero-shOt Embeddings from language model

**Q: Tell me a little bit more about `RoBERTa-wwm-ext`**  
A: integrate whole word masking (wwm) into RoBERTa model, specifically:  
1) use whole word masking (but we did not use dynamic masking)  
2) remove Next Sentence Prediction (NSP)
3) directly use the data generated by `max_len=512` (but not from `max_len=128` for several steps then `max_len=512`)
4) extended training steps (1M steps)

## Citation
If you find the technical report or resource is useful, please cite our work in your paper.
- Primary (Journal extension): https://ieeexplore.ieee.org/document/9599397  
```
@journal{cui-etal-2021-pretrain,
  title={Pre-Training with Whole Word Masking for Chinese BERT},
  author={Cui, Yiming and Che, Wanxiang and Liu, Ting and Qin, Bing and Yang, Ziqing},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  year={2021},
  url={https://ieeexplore.ieee.org/document/9599397},
  doi={10.1109/TASLP.2021.3124365},
 }
```
- Secondary (conference paper): https://www.aclweb.org/anthology/2020.findings-emnlp.58
```
@inproceedings{cui-etal-2020-revisiting,
    title = "Revisiting Pre-Trained Models for {C}hinese Natural Language Processing",
    author = "Cui, Yiming  and
      Che, Wanxiang  and
      Liu, Ting  and
      Qin, Bing  and
      Wang, Shijin  and
      Hu, Guoping",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.58",
    pages = "657--668",
}
```

## Disclaimer
**This is NOT a project by Google official. Also, this is NOT an official product by HIT and iFLYTEK.**
The experiments only represent the empirical results in certain conditions and should not be regarded as the nature of the respective models. The results may vary using different random seeds, computing devices, etc. 
**The contents in this repository are for academic research purpose, and we do not provide any conclusive remarks. Users are free to use anythings in this repository within the scope of Apache-2.0 licence. However, we are not responsible for direct or indirect losses that was caused by using the content in this project.**


## Acknowledgement
The first author of this project is partially supported by [Google TensorFlow Research Cloud (TFRC) Program](https://www.tensorflow.org/tfrc).

## Issues
If there is any problem, please submit a GitHub Issue.

