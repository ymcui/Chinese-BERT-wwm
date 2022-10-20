
<p>
<a href="https://console.tiyaro.ai/explore?q=hfl/&pub=hfl"> <img src="https://tiyaro-public-docs.s3.us-west-2.amazonaws.com/assets/try_on_tiyaro_badge.svg"></a>
</p>

[**中文说明**](https://github.com/ymcui/Chinese-BERT-wwm/) | [**English**](https://github.com/ymcui/Chinese-BERT-wwm/blob/master/README_EN.md)

<p align="center">
    <br>
    <img src="./pics/banner.png" width="500"/>
    <br>
</p>
<p align="center">
    <a href="https://github.com/ymcui/Chinese-BERT-wwm/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-BERT-wwm.svg?color=blue&style=flat-square">
    </a>
</p>

在自然语言处理领域中，预训练语言模型（Pre-trained Language Models）已成为非常重要的基础技术。为了进一步促进中文信息处理的研究发展，我们发布了基于全词掩码（Whole Word Masking）技术的中文预训练模型BERT-wwm，以及与此技术密切相关的模型：BERT-wwm-ext，RoBERTa-wwm-ext，RoBERTa-wwm-ext-large, RBT3, RBTL3等。  

- **[Pre-Training with Whole Word Masking for Chinese BERT](https://ieeexplore.ieee.org/document/9599397)**  
- *Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Ziqing Yang*
- Published in *IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)*

本项目基于谷歌官方BERT：https://github.com/google-research/bert

----

[中文MacBERT](https://github.com/ymcui/MacBERT) | [中文ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [中文XLNet](https://github.com/ymcui/Chinese-XLNet) | [知识蒸馏工具TextBrewer](https://github.com/airaria/TextBrewer) | [模型裁剪工具TextPruner](https://github.com/airaria/TextPruner)

查看更多哈工大讯飞联合实验室（HFL）发布的资源：https://github.com/ymcui/HFL-Anthology

## 新闻
**2022/3/30 我们开源了一种新预训练模型PERT。查看：https://github.com/ymcui/PERT**

2021/12/17 哈工大讯飞联合实验室推出模型裁剪工具包TextPruner。查看：https://github.com/airaria/TextPruner

2021/10/24 哈工大讯飞联合实验室发布面向少数民族语言的预训练模型CINO。查看：https://github.com/ymcui/Chinese-Minority-PLM

2021/7/21 由哈工大SCIR多位学者撰写的[《自然语言处理：基于预训练模型的方法》](https://item.jd.com/13344628.html)已出版，欢迎大家选购。

2021/1/27 所有模型已支持TensorFlow 2，请通过transformers库进行调用或下载。https://huggingface.co/hfl

<details>
<summary>历史新闻</summary>
2020/9/15 我们的论文["Revisiting Pre-Trained Models for Chinese Natural Language Processing"](https://arxiv.org/abs/2004.13922)被[Findings of EMNLP](https://2020.emnlp.org)录用为长文。

2020/8/27 哈工大讯飞联合实验室在通用自然语言理解评测GLUE中荣登榜首，查看[GLUE榜单](https://gluebenchmark.com/leaderboard)，[新闻](http://dwz.date/ckrD)。

2020/3/23 本目录发布的模型已接入[飞桨PaddleHub](https://github.com/PaddlePaddle/PaddleHub)，查看[快速加载](#快速加载)

2020/3/11 为了更好地了解需求，邀请您填写[调查问卷](https://wj.qq.com/s2/5637766/6281)，以便为大家提供更好的资源。

2020/2/26 哈工大讯飞联合实验室发布[知识蒸馏工具TextBrewer](https://github.com/airaria/TextBrewer)

2020/1/20 祝大家鼠年大吉，本次发布了RBT3、RBTL3（3层RoBERTa-wwm-ext-base/large），查看[小参数量模型](#小参数量模型)

2019/12/19 本目录发布的模型已接入[Huggingface-Transformers](https://github.com/huggingface/transformers)，查看[快速加载](#快速加载)

2019/10/14 发布萝卜塔RoBERTa-wwm-ext-large模型，查看[中文模型下载](#中文模型下载)

2019/9/10 发布萝卜塔RoBERTa-wwm-ext模型，查看[中文模型下载](#中文模型下载)

2019/7/30 提供了在更大通用语料（5.4B词数）上训练的中文`BERT-wwm-ext`模型，查看[中文模型下载](#中文模型下载)

2019/6/20 初始版本，模型已可通过谷歌下载，国内云盘也已上传完毕，查看[中文模型下载](#中文模型下载)
</details>

## 内容导引
| 章节 | 描述 |
|-|-|
| [简介](#简介) | 介绍BERT-wwm基本原理 |
| [中文模型下载](#中文模型下载) | 提供了BERT-wwm的下载地址 |
| [快速加载](#快速加载) | 介绍了如何使用[🤗Transformers](https://github.com/huggingface/transformers)、[PaddleHub](https://github.com/PaddlePaddle/PaddleHub)快速加载模型 |
| [模型对比](#模型对比) | 提供了本目录中模型的参数对比 |
| [中文基线系统效果](#中文基线系统效果) | 列举了部分中文基线系统效果 |
| [小参数量模型](#小参数量模型) | 列举了小参数量模型（3层Transformer）的效果 |
| [使用建议](#使用建议) | 提供了若干使用中文预训练模型的建议 |
| [英文模型下载](#英文模型下载) | 谷歌官方的英文BERT-wwm下载地址 |
| [FAQ](#FAQ) | 常见问题答疑 |
| [引用](#引用) | 本目录的技术报告 |


## 简介
**Whole Word Masking (wwm)**，暂翻译为`全词Mask`或`整词Mask`，是谷歌在2019年5月31日发布的一项BERT的升级版本，主要更改了原预训练阶段的训练样本生成策略。
简单来说，原有基于WordPiece的分词方式会把一个完整的词切分成若干个子词，在生成训练样本时，这些被分开的子词会随机被mask。
在`全词Mask`中，如果一个完整的词的部分WordPiece子词被mask，则同属该词的其他部分也会被mask，即`全词Mask`。

**需要注意的是，这里的mask指的是广义的mask（替换成[MASK]；保持原词汇；随机替换成另外一个词），并非只局限于单词替换成`[MASK]`标签的情况。
更详细的说明及样例请参考：[#4](https://github.com/ymcui/Chinese-BERT-wwm/issues/4)**

同理，由于谷歌官方发布的`BERT-base, Chinese`中，中文是以**字**为粒度进行切分，没有考虑到传统NLP中的中文分词（CWS）。
我们将全词Mask的方法应用在了中文中，使用了中文维基百科（包括简体和繁体）进行训练，并且使用了[哈工大LTP](http://ltp.ai)作为分词工具，即对组成同一个**词**的汉字全部进行Mask。

下述文本展示了`全词Mask`的生成样例。
**注意：为了方便理解，下述例子中只考虑替换成[MASK]标签的情况。**

| 说明 | 样例 |
| :------- | :--------- |
| 原始文本 | 使用语言模型来预测下一个词的probability。 |
| 分词文本 | 使用 语言 模型 来 预测 下 一个 词 的 probability 。 |
| 原始Mask输入 | 使 用 语 言 [MASK] 型 来 [MASK] 测 下 一 个 词 的 pro [MASK] ##lity 。 |
| 全词Mask输入 | 使 用 语 言 [MASK] [MASK] 来 [MASK] [MASK] 下 一 个 词 的 [MASK] [MASK] [MASK] 。 |


## 中文模型下载
本目录中主要包含base模型，故我们不在模型简称中标注`base`字样。对于其他大小的模型会标注对应的标记（例如large）。

* **`BERT-large模型`**：24-layer, 1024-hidden, 16-heads, 330M parameters  
* **`BERT-base模型`**：12-layer, 768-hidden, 12-heads, 110M parameters  

**注意：开源版本不包含MLM任务的权重；如需做MLM任务，请使用额外数据进行二次预训练（和其他下游任务一样）。**

| 模型简称 | 语料 | Google下载 | 百度网盘下载 |
| :------- | :--------- | :---------: | :---------: |
| **`RBT6, Chinese`** | **EXT数据<sup>[1]</sup>** | - | **[TensorFlow（密码hniy）](https://pan.baidu.com/s/1_MDAIYIGVgDovWkSs51NDA?pwd=hniy)** |
| **`RBT4, Chinese`** | **EXT数据<sup>[1]</sup>** | - | **[TensorFlow（密码sjpt）](https://pan.baidu.com/s/1MUrmuTULnMn3L1aw_dXxSA?pwd=sjpt)** |
| **`RBTL3, Chinese`** | **EXT数据<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1Jzn1hYwmv0kXkfTeIvNT61Rn1IbRc-o8)**<br/>**[PyTorch](https://drive.google.com/open?id=1qs5OasLXXjOnR2XuGUh12NanUl0pkjEv)** | **[TensorFlow（密码s6cu）](https://pan.baidu.com/s/1vV9ClBMbsSpt8wUpfQz62Q?pwd=s6cu)** |
| **`RBT3, Chinese`** | **EXT数据<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1-rvV0nBDvRCASbRz8M9Decc3_8Aw-2yi)**<br/>**[PyTorch](https://drive.google.com/open?id=1_LqmIxm8Nz1Abvlqb8QFZaxYo-TInOed)** | **[TensorFlow（密码5a57）](https://pan.baidu.com/s/1AnapwWj1YBZ_4E6AAtj2lg?pwd=5a57)** |
| **`RoBERTa-wwm-ext-large, Chinese`** | **EXT数据<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1dtad0FFzG11CBsawu8hvwwzU2R0FDI94)**<br/>**[PyTorch](https://drive.google.com/open?id=1-2vEZfIFCdM1-vJ3GD6DlSyKT4eVXMKq)** | **[TensorFlow（密码dqqe）](https://pan.baidu.com/s/1F68xzCLWEonTEVP7HQ0Ciw?pwd=dqqe)** |
| **`RoBERTa-wwm-ext, Chinese`** | **EXT数据<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1jMAKIJmPn7kADgD3yQZhpsqM-IRM1qZt)** <br/>**[PyTorch](https://drive.google.com/open?id=1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25)** | **[TensorFlow（密码vybq）](https://pan.baidu.com/s/1oR0cgSXE3Nz6dESxr98qVA?pwd=vybq)** |
| **`BERT-wwm-ext, Chinese`** | **EXT数据<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1buMLEjdtrXE2c4G1rpsNGWEx7lUQ0RHi)** <br/>**[PyTorch](https://drive.google.com/open?id=1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_)** | **[TensorFlow（密码wgnt）](https://pan.baidu.com/s/1x-jIw1X2yNYHGak2yiq4RQ?pwd=wgnt)** |
| **`BERT-wwm, Chinese`** | **中文维基** | **[TensorFlow](https://drive.google.com/open?id=1RoTQsXp2hkQ1gSRVylRIJfQxJUgkfJMW)** <br/>**[PyTorch](https://drive.google.com/open?id=1AQitrjbvCWc51SYiLN-cJq4e0WiNN4KY)** | **[TensorFlow（密码qfh8）](https://pan.baidu.com/s/1HDdDXiYxGT5ub5OeO7qdWw?pwd=qfh8)** |
| `BERT-base, Chinese`<sup>Google</sup> | 中文维基 | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) | - |
| `BERT-base, Multilingual Cased`<sup>Google</sup>  | 多语种维基 | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) | - |
| `BERT-base, Multilingual Uncased`<sup>Google</sup>  | 多语种维基 | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip) | - |

> [1] EXT数据包括：中文维基百科，其他百科、新闻、问答等数据，总词数达5.4B。

### PyTorch版本

如需PyTorch版本，

1）请自行通过[🤗Transformers](https://github.com/huggingface/transformers)提供的转换脚本进行转换。

2）或者通过huggingface官网直接下载PyTorch版权重：https://huggingface.co/hfl

下载方法：点击任意需要下载的模型 → 选择"Files and versions"选项卡 → 下载对应的模型文件。

### 使用说明

中国大陆境内建议使用百度网盘下载点，境外用户建议使用谷歌下载点，base模型文件大小约**400M**。 
以TensorFlow版`BERT-wwm, Chinese`为例，下载完毕后对zip文件进行解压得到：

```
chinese_wwm_L-12_H-768_A-12.zip
    |- bert_model.ckpt      # 模型权重
    |- bert_model.meta      # 模型meta信息
    |- bert_model.index     # 模型index信息
    |- bert_config.json     # 模型参数
    |- vocab.txt            # 词表
```
其中`bert_config.json`和`vocab.txt`与谷歌原版`BERT-base, Chinese`完全一致。
PyTorch版本则包含`pytorch_model.bin`, `bert_config.json`, `vocab.txt`文件。


## 快速加载
### 使用Huggingface-Transformers

依托于[🤗transformers库](https://github.com/huggingface/transformers)，可轻松调用以上模型。
```
tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BertModel.from_pretrained("MODEL_NAME")
```
**注意：本目录中的所有模型均使用BertTokenizer以及BertModel加载，请勿使用RobertaTokenizer/RobertaModel！**

其中`MODEL_NAME`对应列表如下：

| 模型名 | MODEL_NAME |
| - | - |
| RoBERTa-wwm-ext-large | hfl/chinese-roberta-wwm-ext-large |
| RoBERTa-wwm-ext | hfl/chinese-roberta-wwm-ext |
| BERT-wwm-ext | hfl/chinese-bert-wwm-ext |
| BERT-wwm | hfl/chinese-bert-wwm |
| RBT3 | hfl/rbt3 |
| RBTL3 | hfl/rbtl3 |

### 使用PaddleHub

依托[PaddleHub](https://github.com/PaddlePaddle/PaddleHub)，只需一行代码即可完成模型下载安装，十余行代码即可完成文本分类、序列标注、阅读理解等任务。

```
import paddlehub as hub
module = hub.Module(name=MODULE_NAME)
```

其中`MODULE_NAME`对应列表如下：

| 模型名 | MODULE_NAME |
| - | - |
| RoBERTa-wwm-ext-large | [chinese-roberta-wwm-ext-large](https://www.paddlepaddle.org.cn/hubdetail?name=chinese-roberta-wwm-ext-large&en_category=SemanticModel) |
| RoBERTa-wwm-ext       | [chinese-roberta-wwm-ext](https://www.paddlepaddle.org.cn/hubdetail?name=chinese-roberta-wwm-ext&en_category=SemanticModel) |
| BERT-wwm-ext          | [chinese-bert-wwm-ext](https://www.paddlepaddle.org.cn/hubdetail?name=chinese-bert-wwm-ext&en_category=SemanticModel) |
| BERT-wwm              | [chinese-bert-wwm](https://www.paddlepaddle.org.cn/hubdetail?name=chinese-bert-wwm&en_category=SemanticModel) |
| RBT3                  | [rbt3](https://www.paddlepaddle.org.cn/hubdetail?name=rbt3&en_category=SemanticModel) |
| RBTL3                 | [rbtl3](https://www.paddlepaddle.org.cn/hubdetail?name=rbtl3&en_category=SemanticModel) |


## 模型对比
针对大家比较关心的一些模型细节进行汇总如下。

| - | BERT<sup>Google</sup> | BERT-wwm | BERT-wwm-ext | RoBERTa-wwm-ext | RoBERTa-wwm-ext-large |
| :------- | :---------: | :---------: | :---------: | :---------: | :---------: |
| Masking | WordPiece | WWM<sup>[1]</sup> | WWM | WWM | WWM |
| Type | base | base | base | base | **large** |
| Data Source | wiki | wiki | wiki+ext<sup>[2]</sup> | wiki+ext | wiki+ext |
| Training Tokens # | 0.4B | 0.4B | 5.4B | 5.4B | 5.4B |
| Device | TPU Pod v2 | TPU v3 | TPU v3 | TPU v3 | **TPU Pod v3-32<sup>[3]</sup>** |
| Training Steps | ? | 100K<sup>MAX128</sup> <br/>+100K<sup>MAX512</sup> | 1M<sup>MAX128</sup> <br/>+400K<sup>MAX512</sup> | 1M<sup>MAX512</sup> | 2M<sup>MAX512</sup> |
| Batch Size | ? | 2,560 / 384 | 2,560 / 384 | 384 | 512 |
| Optimizer | AdamW | LAMB | LAMB | AdamW | AdamW |
| Vocabulary | 21,128 | ~BERT<sup>[4]</sup> | ~BERT | ~BERT | ~BERT |
| Init Checkpoint | Random Init | ~BERT | ~BERT | ~BERT | Random Init |

> [1] WWM = Whole Word Masking  
> [2] ext = extended data  
> [3] TPU Pod v3-32 (512G HBM)等价于4个TPU v3 (128G HBM)  
> [4] `~BERT`表示**继承**谷歌原版中文BERT的属性  


## 中文基线系统效果
为了对比基线效果，我们在以下几个中文数据集上进行了测试，包括`句子级`和`篇章级`任务。
对于`BERT-wwm-ext`、`RoBERTa-wwm-ext`、`RoBERTa-wwm-ext-large`，我们**没有进一步调整最佳学习率**，而是直接使用了`BERT-wwm`的最佳学习率。

最佳学习率：  

| 模型 | BERT | ERNIE | BERT-wwm* |
| :------- | :---------: | :---------: | :---------: |
| CMRC 2018 | 3e-5 | 8e-5 | 3e-5 |
| DRCD | 3e-5 | 8e-5 | 3e-5 |
| CJRC | 4e-5 | 8e-5 | 4e-5 |
| XNLI | 3e-5 | 5e-5 | 3e-5 |
| ChnSentiCorp | 2e-5 | 5e-5 | 2e-5 |
| LCQMC  | 2e-5 | 3e-5 | 2e-5 |
| BQ Corpus | 3e-5 | 5e-5 | 3e-5 |
| THUCNews | 2e-5 | 5e-5 | 2e-5 |

*代表所有wwm系列模型 (BERT-wwm, BERT-wwm-ext, RoBERTa-wwm-ext, RoBERTa-wwm-ext-large)


**下面仅列举部分结果，完整结果请查看我们的[技术报告](https://arxiv.org/abs/1906.08101)。**

- [**CMRC 2018**：篇章片段抽取型阅读理解（简体中文）](https://github.com/ymcui/cmrc2018)
- [**DRCD**：篇章片段抽取型阅读理解（繁体中文）](https://github.com/DRCSolutionService/DRCD)
- [**CJRC**: 法律阅读理解（简体中文）](http://cail.cipsc.org.cn)
- [**XNLI**：自然语言推断](https://github.com/google-research/bert/blob/master/multilingual.md)
- [**ChnSentiCorp**：情感分析](https://github.com/pengming617/bert_classification)
- [**LCQMC**：句对匹配](http://icrc.hitsz.edu.cn/info/1037/1146.htm)
- [**BQ Corpus**：句对匹配](http://icrc.hitsz.edu.cn/Article/show/175.html)
- [**THUCNews**：篇章级文本分类](http://thuctc.thunlp.org)

**注意：为了保证结果的可靠性，对于同一模型，我们运行10遍（不同随机种子），汇报模型性能的最大值和平均值（括号内为平均值）。不出意外，你运行的结果应该很大概率落在这个区间内。**

**评测指标中，括号内表示平均值，括号外表示最大值。**


### 简体中文阅读理解：CMRC 2018
[**CMRC 2018数据集**](https://github.com/ymcui/cmrc2018)是哈工大讯飞联合实验室发布的中文机器阅读理解数据。
根据给定问题，系统需要从篇章中抽取出片段作为答案，形式与SQuAD相同。
评测指标为：EM / F1

| 模型 | 开发集 | 测试集 | 挑战集 |
| :------- | :---------: | :---------: | :---------: |
| BERT | 65.5 (64.4) / 84.5 (84.0) | 70.0 (68.7) / 87.0 (86.3) | 18.6 (17.0) / 43.3 (41.3) |
| ERNIE | 65.4 (64.3) / 84.7 (84.2) | 69.4 (68.2) / 86.6 (86.1) | 19.6 (17.0) / 44.3 (42.8) |
| **BERT-wwm** | 66.3 (65.0) / 85.6 (84.7) | 70.5 (69.1) / 87.4 (86.7) | 21.0 (19.3) / 47.0 (43.9) |
| **BERT-wwm-ext** | 67.1 (65.6) / 85.7 (85.0) | 71.4 (70.0) / 87.7 (87.0) | 24.0 (20.0) / 47.3 (44.6) |
| **RoBERTa-wwm-ext** | 67.4 (66.5) / 87.2 (86.5) | 72.6 (71.4) / 89.4 (88.8) | 26.2 (24.6) / 51.0 (49.1) |
| **RoBERTa-wwm-ext-large** | **68.5 (67.6) / 88.4 (87.9)** | **74.2 (72.4) / 90.6 (90.0)** | **31.5 (30.1) / 60.1 (57.5)** |


### 繁体中文阅读理解：DRCD
[**DRCD数据集**](https://github.com/DRCKnowledgeTeam/DRCD)由中国台湾台达研究院发布，其形式与SQuAD相同，是基于繁体中文的抽取式阅读理解数据集。
**由于ERNIE中去除了繁体中文字符，故不建议在繁体中文数据上使用ERNIE（或转换成简体中文后再处理）。**
评测指标为：EM / F1

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 83.1 (82.7) / 89.9 (89.6) | 82.2 (81.6) / 89.2 (88.8) |
| ERNIE | 73.2 (73.0) / 83.9 (83.8) | 71.9 (71.4) / 82.5 (82.3) |
| **BERT-wwm** | 84.3 (83.4) / 90.5 (90.2) | 82.8 (81.8) / 89.7 (89.0) |
| **BERT-wwm-ext** | 85.0 (84.5) / 91.2 (90.9) | 83.6 (83.0) / 90.4 (89.9) |
| **RoBERTa-wwm-ext** | 86.6 (85.9) / 92.5 (92.2) | 85.6 (85.2) / 92.0 (91.7) |
| **RoBERTa-wwm-ext-large** | **89.6 (89.1) / 94.8 (94.4)** | **89.6 (88.9) / 94.5 (94.1)** |


### 司法阅读理解：CJRC
[**CJRC数据集**](http://cail.cipsc.org.cn)是哈工大讯飞联合实验室发布的面向**司法领域**的中文机器阅读理解数据。
需要注意的是实验中使用的数据并非官方发布的最终数据，结果仅供参考。
评测指标为：EM / F1

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 54.6 (54.0) / 75.4 (74.5) | 55.1 (54.1) / 75.2 (74.3) |
| ERNIE | 54.3 (53.9) / 75.3 (74.6) | 55.0 (53.9) / 75.0 (73.9) |
| **BERT-wwm** | 54.7 (54.0) / 75.2 (74.8) | 55.1 (54.1) / 75.4 (74.4) |
| **BERT-wwm-ext** | 55.6 (54.8) / 76.0 (75.3) | 55.6 (54.9) / 75.8 (75.0) |
| **RoBERTa-wwm-ext** | 58.7 (57.6) / 79.1 (78.3) | 59.0 (57.8) / 79.0 (78.0) |
| **RoBERTa-wwm-ext-large** | **62.1 (61.1) / 82.4 (81.6)** | **62.4 (61.4) / 82.2 (81.0)** |


### 自然语言推断：XNLI
在自然语言推断任务中，我们采用了[**XNLI**数据](https://github.com/google-research/bert/blob/master/multilingual.md)，需要将文本分成三个类别：`entailment`，`neutral`，`contradictory`。
评测指标为：Accuracy

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 77.8 (77.4) | 77.8 (77.5) |
| ERNIE | 79.7 (79.4) | 78.6 (78.2) |
| **BERT-wwm** | 79.0 (78.4) | 78.2 (78.0) |
| **BERT-wwm-ext** | 79.4 (78.6) | 78.7 (78.3) |
| **RoBERTa-wwm-ext** | 80.0 (79.2) | 78.8 (78.3) |
| **RoBERTa-wwm-ext-large** | **82.1 (81.3)** | **81.2 (80.6)** |


### 情感分析：ChnSentiCorp
在情感分析任务中，二分类的情感分类数据集ChnSentiCorp。
评测指标为：Accuracy

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 94.7 (94.3) | 95.0 (94.7) |
| ERNIE | 95.4 (94.8) | 95.4 **(95.3)** |
| **BERT-wwm** | 95.1 (94.5) | 95.4 (95.0) |
| **BERT-wwm-ext** | 95.4 (94.6) | 95.3 (94.7) |
| **RoBERTa-wwm-ext** | 95.0 (94.6) | 95.6 (94.8) |
| **RoBERTa-wwm-ext-large** | **95.8 (94.9)** | **95.8** (94.9) |


### 句对分类：LCQMC, BQ Corpus
以下两个数据集均需要将一个句对进行分类，判断两个句子的语义是否相同（二分类任务）。

#### LCQMC
[LCQMC](http://icrc.hitsz.edu.cn/info/1037/1146.htm)由哈工大深圳研究生院智能计算研究中心发布。 
评测指标为：Accuracy

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 89.4 (88.4) | 86.9 (86.4) |
| ERNIE | 89.8 (89.6) | **87.2 (87.0)** |
| **BERT-wwm** | 89.4 (89.2) | 87.0 (86.8) |
| **BERT-wwm-ext** | 89.6 (89.2) | 87.1 (86.6) |
| **RoBERTa-wwm-ext** | 89.0 (88.7) | 86.4 (86.1) |
| **RoBERTa-wwm-ext-large** | **90.4 (90.0)** | 87.0 (86.8) |


#### BQ Corpus 
[BQ Corpus](http://icrc.hitsz.edu.cn/Article/show/175.html)由哈工大深圳研究生院智能计算研究中心发布，是面向银行领域的数据集。
评测指标为：Accuracy

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 86.0 (85.5) | 84.8 (84.6) |
| ERNIE | 86.3 (85.5) | 85.0 (84.6) |
| **BERT-wwm** | 86.1 (85.6) | 85.2 **(84.9)** |
| **BERT-wwm-ext** | **86.4** (85.5) | 85.3 (84.8) |
| **RoBERTa-wwm-ext** | 86.0 (85.4) | 85.0 (84.6) |
| **RoBERTa-wwm-ext-large** | 86.3 **(85.7)** | **85.8 (84.9)** |


### 篇章级文本分类：THUCNews
篇章级文本分类任务我们选用了由清华大学自然语言处理实验室发布的新闻数据集**THUCNews**。
我们采用的是其中一个子集，需要将新闻分成10个类别中的一个。
评测指标为：Accuracy

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 97.7 (97.4) | 97.8 (97.6) |
| ERNIE | 97.6 (97.3) | 97.5 (97.3) |
| **BERT-wwm** | 98.0 (97.6) | 97.8 (97.6) |
| **BERT-wwm-ext** | 97.7 (97.5) | 97.7 (97.5) |
| **RoBERTa-wwm-ext** | 98.3 (97.9) | 97.7 (97.5) |
| **RoBERTa-wwm-ext-large** | 98.3 (97.7) | 97.8 (97.6) |


### 小参数量模型
以下是在若干NLP任务上的实验效果，表中只提供测试集结果对比。

| 模型 | CMRC 2018 | DRCD | XNLI | CSC | LCQMC | BQ | 平均 | 参数量 |
| :------- | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| RoBERTa-wwm-ext-large | 74.2 / 90.6 | 89.6 / 94.5 | 81.2 | 95.8 | 87.0 | 85.8 | 87.335 | 325M |
| RoBERTa-wwm-ext | 72.6 / 89.4 | 85.6 / 92.0 | 78.8 | 95.6 | 86.4 | 85.0 | 85.675 | 102M |
| RBTL3 | 63.3 / 83.4 | 77.2 / 85.6 | 74.0 | 94.2 | 85.1 | 83.6 | 80.800 | 61M (59.8%) |
| RBT3 | 62.2 / 81.8 | 75.0 / 83.9 | 72.3 | 92.8 | 85.1 | 83.3 | 79.550 | 38M (37.3%) |

效果相对值比较：

| 模型 | CMRC 2018 | DRCD | XNLI | CSC | LCQMC | BQ | 平均 | 分类平均 |
| :------- | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| RoBERTa-wwm-ext-large | 102.2% / 101.3% | 104.7% / 102.7% | 103.0% | 100.2% | 100.7% | 100.9% | 101.9% | 101.2% |
| RoBERTa-wwm-ext | 100% / 100% | 100% / 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| RBTL3 | 87.2% / 93.3% | 90.2% / 93.0% | 93.9% | 98.5% | 98.5% | 98.4% | 94.3% | 97.35% |
| RBT3 | 85.7% / 91.5% | 87.6% / 91.2% | 91.8% | 97.1% | 98.5% | 98.0% | 92.9% | 96.35% |

- 参数量是以XNLI分类任务为基准进行计算
- 括号内参数量百分比以原始base模型（即RoBERTa-wwm-ext）为基准
- RBT3：由RoBERTa-wwm-ext 3层进行初始化，继续训练了1M步
- RBTL3：由RoBERTa-wwm-ext-large 3层进行初始化，继续训练了1M步
- RBT的名字是RoBERTa三个音节首字母组成，L代表large模型
- 直接使用RoBERTa-wwm-ext-large前三层进行初始化并进行下游任务的训练将显著降低效果，例如在CMRC 2018上测试集仅能达到42.9/65.3，而RBTL3能达到63.3/83.4


## 使用建议
* 初始学习率是非常重要的一个参数（不论是`BERT`还是其他模型），需要根据目标任务进行调整。
* `ERNIE`的最佳学习率和`BERT`/`BERT-wwm`相差较大，所以使用`ERNIE`时请务必调整学习率（基于以上实验结果，`ERNIE`需要的初始学习率较高）。
* 由于`BERT`/`BERT-wwm`使用了维基百科数据进行训练，故它们对正式文本建模较好；而`ERNIE`使用了额外的百度贴吧、知道等网络数据，它对非正式文本（例如微博等）建模有优势。
* 在长文本建模任务上，例如阅读理解、文档分类，`BERT`和`BERT-wwm`的效果较好。
* 如果目标任务的数据和预训练模型的领域相差较大，请在自己的数据集上进一步做预训练。
* 如果要处理繁体中文数据，请使用`BERT`或者`BERT-wwm`。因为我们发现`ERNIE`的词表中几乎没有繁体中文。


## 英文模型下载
为了方便大家下载，顺便带上**谷歌官方发布**的英文`BERT-large (wwm)`模型：

*   **[`BERT-Large, Uncased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters

*   **[`BERT-Large, Cased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters

## FAQ
**Q: 这个模型怎么用？**  
A: 谷歌发布的中文BERT怎么用，这个就怎么用。
**文本不需要经过分词，wwm只影响预训练过程，不影响下游任务的输入。**

**Q: 请问有预训练代码提供吗？**  
A: 很遗憾，我不能提供相关代码，实现可以参考 [#10](https://github.com/ymcui/Chinese-BERT-wwm/issues/10) 和 [#13](https://github.com/ymcui/Chinese-BERT-wwm/issues/13)。

**Q: 某某数据集在哪里下载？**  
A: 请查看`data`目录，任务目录下的`README.md`标明了数据来源。对于有版权的内容，请自行搜索或与原作者联系获取数据。

**Q: 会有计划发布更大模型吗？比如BERT-large-wwm版本？**  
A: 如果我们从实验中得到更好效果，会考虑发布更大的版本。

**Q: 你骗人！无法复现结果😂**  
A: 在下游任务中，我们采用了最简单的模型。比如分类任务，我们直接使用的是`run_classifier.py`（谷歌提供）。
如果无法达到平均值，说明实验本身存在bug，请仔细排查。
最高值存在很多随机因素，我们无法保证能够达到最高值。
另外一个公认的因素：降低batch size会显著降低实验效果，具体可参考BERT，XLNet目录的相关Issue。

**Q: 我训出来比你更好的结果！**  
A: 恭喜你。

**Q: 训练花了多长时间，在什么设备上训练的？**  
A: 训练是在谷歌TPU v3版本（128G HBM）完成的，训练BERT-wwm花费约1.5天，BERT-wwm-ext则需要数周时间（使用了更多数据需要迭代更充分）。
需要注意的是，预训练阶段我们使用的是`LAMB Optimizer`（[TensorFlow版本实现](https://github.com/ymcui/LAMB_Optimizer_TF)）。该优化器对大的batch有良好的支持。
在微调下游任务时，我们采用的是BERT默认的`AdamWeightDecayOptimizer`。

**Q: ERNIE是谁？**  
A: 本项目中的ERNIE模型特指百度公司提出的[ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)，而非清华大学在ACL 2019上发表的[ERNIE](https://github.com/thunlp/ERNIE)。

**Q: BERT-wwm的效果不是在所有任务都很好**  
A: 本项目的目的是为研究者提供多元化的预训练模型，自由选择BERT，ERNIE，或者是BERT-wwm。
我们仅提供实验数据，具体效果如何还是得在自己的任务中不断尝试才能得出结论。
多一个模型，多一种选择。

**Q: 为什么有些数据集上没有试？**  
A: 很坦率的说：
1）没精力找更多的数据；
2）没有必要； 
3）没有钞票；

**Q: 简单评价一下这几个模型**  
A: 各有侧重，各有千秋。
中文自然语言处理的研究发展需要多方共同努力。

**Q: 你预测下一个预训练模型叫什么？**  
A: 可能叫ZOE吧，ZOE: Zero-shOt Embeddings from language model

**Q: 更多关于`RoBERTa-wwm-ext`模型的细节？**  
A: 我们集成了RoBERTa和BERT-wwm的优点，对两者进行了一个自然的结合。
和之前本目录中的模型之间的区别如下:  
1）预训练阶段采用wwm策略进行mask（但没有使用dynamic masking）  
2）简单取消Next Sentence Prediction（NSP）loss  
3）不再采用先max_len=128然后再max_len=512的训练模式，直接训练max_len=512  
4）训练步数适当延长  

需要注意的是，该模型并非原版RoBERTa模型，只是按照类似RoBERTa训练方式训练出的BERT模型，即RoBERTa-like BERT。
故在下游任务使用、模型转换时请按BERT的方式处理，而非RoBERTa。


## 引用
如果本项目中的资源或技术对你的研究工作有所帮助，欢迎在论文中引用下述论文。
- 首选（期刊扩充版）：https://ieeexplore.ieee.org/document/9599397
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

- 或者（会议版本）：https://www.aclweb.org/anthology/2020.findings-emnlp.58
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


## 致谢
第一作者部分受到[**谷歌TPU Research Cloud**](https://www.tensorflow.org/tfrc)计划资助。


## 免责声明
**本项目并非谷歌官方发布的Chinese BERT-wwm模型。同时，本项目不是哈工大或科大讯飞的官方产品。**
技术报告中所呈现的实验结果仅表明在特定数据集和超参组合下的表现，并不能代表各个模型的本质。
实验结果可能因随机数种子，计算设备而发生改变。
**该项目中的内容仅供技术研究参考，不作为任何结论性依据。使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。**


## 关注我们
欢迎关注哈工大讯飞联合实验室官方微信公众号，了解最新的技术动态。

![qrcode.png](https://github.com/ymcui/cmrc2019/raw/master/qrcode.jpg)


## 问题反馈
如有问题，请在GitHub Issue中提交。

