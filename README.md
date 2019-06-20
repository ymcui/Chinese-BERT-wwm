## 中文全词覆盖BERT
为了进一步促进中文自然语言处理的研究发展，我们提供了中文全词覆盖（Whole Word Masking）BERT的预训练模型。
同时在我们的技术报告中详细对比了当今流行的中文预训练模型：[BERT](https://github.com/google-research/bert)、[ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)、[BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)

**For English description, please read our technical report on arXiv: https://arxiv.org/abs/1906.08101**

**更多细节请参考我们的技术报告：https://arxiv.org/abs/1906.08101**

本项目基于谷歌官方的BERT：https://github.com/google-research/bert


## 内容
| 章节 | 描述 |
|-|-|
| [简介](#简介) | 介绍BERT-wwm |
| [中文模型下载](#中文模型下载) | 提供了BERT-wwm的下载地址 |
| [中文基线系统效果](#中文基线系统效果) | 列举了部分中文基线系统效果 |
| [英文模型下载](#英文模型下载) | 谷歌官方的英文BERT-wwm下载地址 |
| [引用](#引用) | 本目录的技术报告 |

## 简介
**Whole Word Masking (wwm)**，暂且翻译为`全词Mask`，是谷歌在2019年5月31日发布的一项BERT的升级版本，主要更改了原预训练阶段的训练样本生成策略。简单来说，原有基于WordPiece的分词方式会把一个完整的词切分成若干个词缀，在生成训练样本时，这些被分开的词缀会随机被`[MASK]`替换。在`全词Mask`中，如果一个完整的词的部分WordPiece被`[MASK]`替换，则同属该词的其他部分也会被`[MASK]`替换，即`全词Mask`。

同理，由于谷歌官方发布的`BERT-base , Chinese`中，中文是以**字**为粒度进行切分，没有考虑到传统NLP中的中文分词（CWS）。我们将全词Mask的方法应用在了中文中，使用了中文维基百科（包括简体和繁体）进行训练，并且使用了[哈工大LTP](http://ltp.ai)作为分词工具），即对组成同一个**词**的汉字全部进行[MASK]。

下述文本展示了`全词Mask`的生成样例。

| 说明 | 样例 |
| :------- | :--------- |
| 原始文本 | 使用语言模型来预测下一个词的probability。 |
| 分词文本 | 使用 语言 模型 来 预测 下 一个 词 的 probability 。 |
| 原始Mask输入 | 使 用 语 言 [MASK] 型 来 [MASK] 测 下 一 个 词 的 pro [MASK] ##lity 。 |
| 全词Mask输入 | 使 用 语 言 [MASK] [MASK] 来 [MASK] [MASK] 下 一 个 词 的 [MASK] [MASK] [MASK] 。 |


## 中文模型下载
*   **`BERT-base, Chinese (Whole Word Masking)`**:
    12-layer, 768-hidden, 12-heads, 110M parameters

#### TensorFlow版本（1.12、1.13、1.14测试通过）
- Google: [download_link_for_google_storage](#)
- 百度云: [download_link_for_baidu_pan](#)

#### PyTorch版本（请使用🤗 的[PyTorch-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) > 0.6，其他版本请自行转换）
- Google: [download_link_for_google_storage](#)
- 百度云: [download_link_for_baidu_pan](#)

中国大陆境内建议使用百度云下载点，境外用户建议使用谷歌下载点，文件大小约**400M**。
以TensorFlow版本为例，下载完毕后对zip文件进行解压得到：
```
chinese_wwm_L-12_H-768_A-12.zip
    |- bert_model.ckpt      # 模型权重
    |- bert_model.meta      # 模型meta信息
    |- bert_model.index     # 模型index信息
    |- bert_config.json     # 模型参数
    |- vocab.txt            # 词表
```
其中`bert_config.json`和`vocab.txt`与谷歌原版`**BERT-base, Chinese**`完全一致。


## 中文基线系统效果
为了对比基线效果，我们在以下几个中文数据集上进行了测试，包括`句子级`和`篇章级`任务。
**下面仅列举部分结果，完整结果请查看我们的[技术报告](https://arxiv.org/abs/1906.08101)。**

- [**CMRC 2018**：篇章抽取型阅读理解（简体中文）](https://github.com/ymcui/cmrc2018)
- [**DRCD**：篇章抽取型阅读理解（繁体中文）](https://github.com/DRCSolutionService/DRCD)
- [**NER**：中文命名实体抽取](http://sighan.cs.uchicago.edu/bakeoff2006/)
- [**THUCNews**：篇章级文本分类](http://thuctc.thunlp.org)

**注意：为了保证结果的可靠性，对于同一模型，我们运行10遍（不同随机种子），汇报模型性能的最大值和平均值。不出意外，你运行的结果应该很大概率落在这个区间内。**


### CMRC 2018
![./pics/cmrc2018.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/cmrc2018.png)

### DRCD
![./pics/drcd.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/drcd.png)

### NER
![./pics/ner.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/ner.png)

### THUCNews
![./pics/thucnews.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/thucnews.png)


## 英文模型下载
为了方便大家下载，顺便带上谷歌官方发布的**英文BERT-large（wwm）**模型：

*   **[`BERT-Large, Uncased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters

*   **[`BERT-Large, Cased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters

## 声明
**这不是谷歌官方发布的Chinese BERT-base (wwm)。**

技术报告中所呈现的实验结果仅表明在特定数据集和超参组合下的表现，并不能代表各个模型的本质。
实验结果可能因随机数种子，计算设备而发生改变。
由于我们没有直接在PaddlePaddle上使用ERNIE，所以在ERNIE上的实验结果仅供参考（虽然我们在多个数据集上复现了效果）。

## 引用
如果你觉得本目录中的内容对研究工作有所帮助，请在文献中引用下述技术报告：
https://arxiv.org/abs/1906.08101
```
@article{chinese-bert-wwm,
  title={Pre-Training with Whole Word Masking for Chinese BERT},
  author={Cui, Yiming and Che, Wanxiang and Liu, Ting and Qin, Bing and Yang, Ziqing and Wang, Shijin and Hu, Guoping},
  journal={arXiv preprint arXiv:1906.08101},
  year={2019}
 }
```

## 问题反馈
如有问题，请在GitHub Issue中提交。
