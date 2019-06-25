## Chinese BERT with Whole Word Masking
For further accelerating Chinese natural language processing, we provide **Chinese pre-trained BERT with Whole Word Masking**. Meanwhile, we also compare the state-of-the-art Chinese pre-trained models in depth, including [BERT](https://github.com/google-research/bert)、[ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)、[BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)

![./pics/header.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/header.png)

**Check our technical report on arXiv: https://arxiv.org/abs/1906.08101**


## News
2019/6/20 Initial version, pre-trained models could be downloaded through Google Storage, check [Download](#Download)


## Guide
| Section | Description |
|-|-|
| [Introduction](#Introduction) | Introduction to BERT with Whole Word Masking (WWM) |
| [Download](#Download) | Download links for Chinese BERT-wwm |
| [Baselines](#Baselines) | Baseline results for several Chinese NLP datasets (partial) |
| [Useful Tips](#Useful-Tips) | Provide several useful tips for using Chinese pre-trained models |
| [English BERT-wwm](#English-BERT-wwm) | Download English BERT-wwm (by Google) |
| [FAQ](#FAQ) | Frequently Asked Questions |
| [Reference](#Reference) | Reference |


## Introduction
**Whole Word Masking (wwm)** is an upgraded version by [BERT]() released on late May 2019.

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
We mainly provide the pre-trained weights on TensorFlow.

*   [**`BERT-base, Chinese (Whole Word Masking)`**](https://drive.google.com/open?id=1RoTQsXp2hkQ1gSRVylRIJfQxJUgkfJMW): 
    12-layer, 768-hidden, 12-heads, 110M parameters

#### PyTorch Version（Please use[PyTorch-BERT by 🤗](https://github.com/huggingface/pytorch-pretrained-BERT) > 0.6, otherwise you need to convert by yourself）
- Google: [download_link_for_google_storage](https://drive.google.com/open?id=1NlMd5GRG97N5BYJHDQR79EU41fEfzMCv)

The whole zip package roughly takes ~400M.
ZIP package (TensorFlow version) includes the following files:
```
chinese_wwm_L-12_H-768_A-12.zip
    |- bert_model.ckpt      # Model Weights
    |- bert_model.meta      # Meta info
    |- bert_model.index     # Index info
    |- bert_config.json     # Config file
    |- vocab.txt            # Vocabulary
```

`bert_config.json` and `vocab.txt` are identical to the original **`BERT-base, Chinese`** by Google。


## Baselines
We experiment on several Chinese datasets, including sentence-level to document-level tasks.

**We only list partial results here and kindly advise the readers to read our [technical report](https://arxiv.org/abs/1906.08101).**

- [**CMRC 2018**：Span-Extraction Machine Reading Comprehension (Simplified Chinese)](https://github.com/ymcui/cmrc2018)
- [**DRCD**：Span-Extraction Machine Reading Comprehension (Traditional Chinese)](https://github.com/DRCSolutionService/DRCD)
- [**NER**：Chinese Named Entity Recognition](http://sighan.cs.uchicago.edu/bakeoff2006/)
- [**THUCNews**：Document-level Text Classification](http://thuctc.thunlp.org)

**Note: To ensure the stability of the results, we run 10 times for each experiment and report maximum and average scores.**


### [CMRC 2018](https://github.com/ymcui/cmrc2018)
CMRC 2018 dataset is released by Joint Laboratory of HIT and iFLYTEK Research.
The model should answer the questions based on the given passage, which is identical to SQuAD.

![./pics/cmrc2018.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/cmrc2018.png)

### [DRCD](https://github.com/DRCKnowledgeTeam/DRCD)
DRCD is also a span-extraction machine reading comprehension dataset, released by Delta Research Center. The text is written in Traditional Chinese.

![./pics/drcd.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/drcd.png)

### NER
We use People Daily and MSRA-NER data for testing Chinese NER.

![./pics/ner.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/ner.png)

### THUCNews
Released by Tsinghua University, which contains news in 10 categories.

![./pics/thucnews.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/thucnews.png)

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
**Q: Do you have any plans on releasing the larger model? Say BERT-large-wwm?** </br>
A: If we could get significant gains from BERT-large, we will release a larger version in the future.

**Q: You lier! I can not reproduce the result! 😂** </br>
A: We use the simplist models in the downstream tasks. For example, in the classification task, we directly use `run_classifier.py` by Google. If you are not able to reach the average score that we reported, then there should be some bugs in your code. As there is randomness in reaching maximum scores, there is no guarantee that you will reproduce them.

**Q: I could get better performance than you!** </br>
A: Congratulations!

**Q: How long did it take to train such a model?** </br>
A: The training was done on Google Cloud TPU v3 with 128HBM, and it roughly takes 1.5 days. Note that, in the pre-training stage, we use [`LAMB Optimizer`](https://github.com/ymcui/LAMB_Optimizer_TF) which is optimized for the larger batch. In fine-tuning downstream task, we use normal `AdamWeightDecayOptimizer` as default.

**Q: Who is ERNIE?** </br>
A: The [ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE) in this repository refer to the model released by Baidu, but not the one that published by Tsinghua University which was also called [ERNIE](https://github.com/thunlp/ERNIE).

**Q: What's the difference between your model and ERNIE?** </br>
A: Baidu ERNIE was released earlier than BERT-wwm by Google (in terms of publication date). Baidu ERNIE uses additional web data (such as Baike, Zhidao, Tieba, etc.), but we only use Chinese Wikipedia.

**Q: How you use ERNIE?** </br>
A: We converted ERNIE from PaddlePaddle format to TensorFlow format and loaded it into the downstream task. Unfortunately, we could not provide PP to TF / PT code at this time, but there are some open-source implementations on GitHub. At the same time, for copyright reasons, we will not provide the pre-trained ERNIE weight in TensorFlow/PyTorch version for you to download. For questions about the use of ERNIE in PaddlePaddle, please consult [ERNIE Official](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE).

**Q: BERT-wwm does not perform well on some tasks.** </br>
A: The aim of this project is to provide researchers with a variety of pre-training models.
You are free to choose one of these models.
We only provide experimental results, and we strongly suggest trying these models in your own task.
One more model, one more choice.


**Q: Why not trying on more dataset?** </br>
A: To be honest: 1) no time to find more data; 2) no need; 3) no money;

**Q: Say something about these models** </br>
A: Each has its own emphasis and merits. Development of Chinese NLP needs joint efforts.

**Q: Any comments on the name of next generation of the pre-trained model?** </br>
A: Maybe ZOE: Zero-shOt Embeddings from language model


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
**This is NOT a project by Google official.**

The experiments only represent the empirical results in certain conditions and should not be regarded as the nature of the respective models. The results may vary using different random seeds, computing devices, etc. Note that, as we have not been testing ERNIE on PaddlePaddle, the results in this technical report may not reflect its true performance (Though we have reproduced several results on the datasets that they had tested.).


## Issues
If there is any problem, please submit a GitHub Issue.

