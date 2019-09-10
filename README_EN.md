[**中文说明**](https://github.com/ymcui/Chinese-BERT-wwm/) | [**English**](https://github.com/ymcui/Chinese-BERT-wwm/blob/master/README_EN.md)

## Chinese BERT with Whole Word Masking
For further accelerating Chinese natural language processing, we provide **Chinese pre-trained BERT with Whole Word Masking**. Meanwhile, we also compare the state-of-the-art Chinese pre-trained models in depth, including [BERT](https://github.com/google-research/bert)、[ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)、[BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)

![./pics/header.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/header.png)

**Check our technical report on arXiv: https://arxiv.org/abs/1906.08101**


## News
**2019/9/10 We release `RoBERTa-wwm-ext`, check [Download](#Download)**

2019/7/30 We release `BERT-wwm-ext`, which was trained on larger data, check [Download](#Download)

2019/6/20 Initial version, pre-trained models could be downloaded through Google Drive, check [Download](#Download)


## Guide
| Section | Description |
|-|-|
| [Introduction](#Introduction) | Introduction to BERT with Whole Word Masking (WWM) |
| [Download](#Download) | Download links for Chinese BERT-wwm |
| [Model Comparison](#Model-Comparison) | Compare the models published in this repository |
| [Baselines](#Baselines) | Baseline results for several Chinese NLP datasets (partial) |
| [Useful Tips](#Useful-Tips) | Provide several useful tips for using Chinese pre-trained models |
| [English BERT-wwm](#English-BERT-wwm) | Download English BERT-wwm (by Google) |
| [FAQ](#FAQ) | Frequently Asked Questions |
| [Reference](#Reference) | Reference |


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
| **`RoBERTa-wwm-ext, Chinese`** | **Wikipedia+Extended data<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1jMAKIJmPn7kADgD3yQZhpsqM-IRM1qZt)** | **[TensorFlow（pw:peMe）](https://pan.iflytek.com:443/link/A136858D5F529E7C385C73EEE336F27B)** |
| **`BERT-wwm-ext, Chinese`** | **Wikipedia+Extended data<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1buMLEjdtrXE2c4G1rpsNGWEx7lUQ0RHi)** <br/>**[PyTorch](https://drive.google.com/open?id=1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_)** | **[TensorFlow（pw:thGd）](https://pan.iflytek.com:443/link/8AA4B23D9BCBCBA0187EE58234332B46)** <br/>**[PyTorch（pw:bJns）](https://pan.iflytek.com:443/link/4AB35DEBECB79C578BEC9952F78FB6F2)** |
| **`BERT-wwm, Chinese`** | **Wikipedia** | **[TensorFlow](https://drive.google.com/open?id=1RoTQsXp2hkQ1gSRVylRIJfQxJUgkfJMW)** <br/>**[PyTorch](https://drive.google.com/open?id=1AQitrjbvCWc51SYiLN-cJq4e0WiNN4KY)** | **[TensorFlow（pw:mva8）](https://pan.iflytek.com:443/link/4B172939D5748FB1A3881772BC97A898)** <br/>**[PyTorch（pw:8fX5）](https://pan.iflytek.com:443/link/8D4E8680433E6AD0F33D521EA920348E)** |
| `BERT-base, Chinese`<sup>Google</sup> | Wikipedia | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) | - |
| `BERT-base, Multilingual Cased`<sup>Google</sup>  | Wikipedia | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) | - |
| `BERT-base, Multilingual Uncased`<sup>Google</sup>  | Wikipedia | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip) | - |

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


#### Task Data
We only provide the data that is publically available, check `data` directory.


## Model Comparison
We list comparisons on the models that were released in this project.
`~BERT` means to inherit the attributes from original Google's BERT.

| - | BERT-wwm | BERT-wwm-ext | RoBERTa-wwm-ext |
| :------- | :---------: | :---------: | :---------: |
| Masking | whole word | whole word | whole word |
| Data | wiki | wiki+extended data | wiki+extended data |
| Device | TPU v3 | TPU v3 | TPU v3 |
| Training Steps | 100K (MAX128) <br/>+100K (MAX512) | 1M (MAX128) <br/>+400K (MAX512) | 1M (MAX512) |
| Batch Size | 2,560 / 384 | 2,560 / 384 | 384 |
| Optimizer | LAMB | LAMB | AdamW |
| Vocabulary | ~BERT vocab | ~BERT vocab | ~BERT vocab |
| Init Checkpoint | ~BERT weight | ~BERT weight | ~BERT weight |


## Baselines
We experiment on several Chinese datasets, including sentence-level to document-level tasks.

**We only list partial results here and kindly advise the readers to read our [technical report](https://arxiv.org/abs/1906.08101).**

- [**CMRC 2018**：Span-Extraction Machine Reading Comprehension (Simplified Chinese)](https://github.com/ymcui/cmrc2018)
- [**DRCD**：Span-Extraction Machine Reading Comprehension (Traditional Chinese)](https://github.com/DRCSolutionService/DRCD)
- [**CJRC**: Chinese Judiciary Reading Comprehension](http://cail.cipsc.org.cn)
- [**XNLI**：Natural Langauge Inference](https://github.com/google-research/bert/blob/master/multilingual.md)
- [**NER**：Chinese Named Entity Recognition](http://sighan.cs.uchicago.edu/bakeoff2006/)
- [**THUCNews**：Document-level Text Classification](http://thuctc.thunlp.org)

**Note: To ensure the stability of the results, we run 10 times for each experiment and report maximum and average scores.**


### [CMRC 2018](https://github.com/ymcui/cmrc2018)
CMRC 2018 dataset is released by Joint Laboratory of HIT and iFLYTEK Research.
The model should answer the questions based on the given passage, which is identical to SQuAD.

| Model | Development | Test | Challenge |
| :------- | :---------: | :---------: | :---------: |
| BERT | 65.5 (64.4) / 84.5 (84.0) | 70.0 (68.7) / 87.0 (86.3) | 18.6 (17.0) / 43.3 (41.3) | 
| ERNIE | 65.4 (64.3) / 84.7 (84.2) | 69.4 (68.2) / 86.6 (86.1) | 19.6 (17.0) / 44.3 (42.8) | 
| **BERT-wwm** | 66.3 (65.0) / 85.6 (84.7) | 70.5 (69.1) / 87.4 (86.7) | 21.0 (19.3) / 47.0 (43.9) | 
| **BERT-wwm-ext** | 67.1 (65.6) / 85.7 (85.0) | 71.4 (70.0) / 87.7 (87.0) | 24.0 (20.0) / 47.3 (44.6) |
| **RoBERTa-wwm-ext** | **67.4 (66.5) / 87.2 (86.5)** | **72.6 (71.4) / 89.4 (88.8)** | **26.2 (24.6) / 51.0 (49.1)** |


### [DRCD](https://github.com/DRCKnowledgeTeam/DRCD)
DRCD is also a span-extraction machine reading comprehension dataset, released by Delta Research Center. The text is written in Traditional Chinese.

| Model | Development | Test |
| :------- | :---------: | :---------: |
| BERT | 83.1 (82.7) / 89.9 (89.6) | 82.2 (81.6) / 89.2 (88.8) | 
| ERNIE | 73.2 (73.0) / 83.9 (83.8) | 71.9 (71.4) / 82.5 (82.3) | 
| **BERT-wwm** | 84.3 (83.4) / 90.5 (90.2) | 82.8 (81.8) / 89.7 (89.0) | 
| **BERT-wwm-ext** | 85.0 (84.5) / 91.2 (90.9) | 83.6 (83.0) / 90.4 (89.9) |
| **RoBERTa-wwm-ext** | **86.6 (85.9) / 92.5 (92.2)** | **85.6 (85.2) / 92.0 (91.7)** | 

### CJRC
[**CJRC**](http://cail.cipsc.org.cn) is a Chinese judiciary reading comprehension dataset, released by Joint Laboratory of HIT and iFLYTEK Research. Note that, the data used in these experiments are NOT identical to the official one.

| Model | Development | Test |
| :------- | :---------: | :---------: |
| BERT | 54.6 (54.0) / 75.4 (74.5) | 55.1 (54.1) / 75.2 (74.3) | 
| ERNIE | 54.3 (53.9) / 75.3 (74.6) | 55.0 (53.9) / 75.0 (73.9) | 
| **BERT-wwm** | 54.7 (54.0) / 75.2 (74.8) | 55.1 (54.1) / 75.4 (74.4) | 
| **BERT-wwm-ext** | 55.6 (54.8) / 76.0 (75.3) | 55.6 (54.9) / 75.8 (75.0) | 
| **RoBERTa-wwm-ext** | **58.7 (57.6) / 79.1 (78.3)** | **59.0 (57.8) / 79.0 (78.0)** |


### XNLI
We use XNLI data for testing NLI task.

| Model | Development | Test |
| :------- | :---------: | :---------: |
| BERT | 77.8 (77.4) | 77.8 (77.5) | 
| ERNIE | **79.7 (79.4)** | 78.6 (78.2) | 
| **BERT-wwm** | 79.0 (78.4) | 78.2 (78.0) | 
| **BERT-wwm-ext** | 79.4 (78.6) | **78.7 (78.3)** |
| **RoBERTa-wwm-ext** | **80.0** (79.2) | **78.8 (78.3)** |

### NER
We use People's Daily and MSRA-NER data for testing Chinese NER.

| Model | People's Daily | MSRA |
| :------- | :---------: | :---------: |
| BERT | 95.2 (94.9) | 95.3 (94.9) |  
| ERNIE | 95.7 (94.5) | 95.4 (95.1) |
| **BERT-wwm** | 95.3 (95.1) | 95.4 (95.1) |

### THUCNews
Released by Tsinghua University, which contains news in 10 categories.

| Model | Development | Test | 
| :------- | :---------: | :---------: | 
| BERT | 97.7 (97.4) | 97.8 (97.6) |
| ERNIE | 97.6 (97.3) | 97.5 (97.3) |
| **BERT-wwm** | 98.0 (97.6) | 97.8 (97.6) |


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
A: Please see `data` directory. For copyright reasons, some of the datasets are not publically available. In that case, please search on GitHub or consult original authors for accessing.

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

## Reference
If you find the technical report or resource is useful, please cite the following technical report in your paper.
https://arxiv.org/abs/1906.08101
```
@article{chinese-bert-wwm,
  title={Pre-Training with Whole Word Masking for Chinese BERT},
  author={Cui, Yiming and Che, Wanxiang and Liu, Ting and Qin, Bing and Yang, Ziqing and Wang, Shijin and Hu, Guoping},
  journal={arXiv preprint arXiv:1906.08101},
  year={2019}
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

