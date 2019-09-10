[**中文说明**](https://github.com/ymcui/Chinese-BERT-wwm/) | [**English**](https://github.com/ymcui/Chinese-BERT-wwm/blob/master/README_EN.md)

## 中文预训练BERT-wwm（Pre-Trained Chinese BERT with Whole Word Masking）

为了进一步促进中文自然语言处理的研究发展，我们提供了基于全词遮掩（Whole Word Masking）技术的中文预训练模型BERT-wwm。
同时在我们的技术报告中详细对比了当今流行的中文预训练模型：[BERT](https://github.com/google-research/bert)、[ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)、[BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)。
**更多细节请参考我们的技术报告：https://arxiv.org/abs/1906.08101**

![./pics/header.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/header.png)

**微信公众号文章介绍**

- 哈工大讯飞联合实验室：https://mp.weixin.qq.com/s/EE6dEhvpKxqnVW_bBAKrnA
- 机器之心：https://mp.weixin.qq.com/s/88OwaHqnrVMQ7vH98INA3w

本项目基于谷歌官方的BERT：https://github.com/google-research/bert


## 新闻
**2019/9/10 发布萝卜塔RoBERTa-wwm-ext模型，查看[中文模型下载](#中文模型下载)**

2019/7/30 提供了在更大通用语料（5.4B词数）上训练的中文`BERT-wwm-ext`模型，查看[中文模型下载](#中文模型下载)

2019/6/20 初始版本，模型已可通过谷歌下载，国内云盘也已上传完毕，查看[中文模型下载](#中文模型下载)


## 内容导引
| 章节 | 描述 |
|-|-|
| [简介](#简介) | 介绍BERT-wwm基本原理 |
| [中文模型下载](#中文模型下载) | 提供了BERT-wwm的下载地址 |
| [模型对比](#模型对比) | 提供了本目录中模型的参数对比 |
| [中文基线系统效果](#中文基线系统效果) | 列举了部分中文基线系统效果 |
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
由于目前只包含base模型，故我们不在模型简称中标注`base`字样。
* **`BERT-base模型`**：12-layer, 768-hidden, 12-heads, 110M parameters

| 模型简称 | 语料 | Google下载 | 讯飞云下载 |
| :------- | :--------- | :---------: | :---------: |
| **`RoBERTa-wwm-ext, Chinese`** | **中文维基+<br/>通用数据<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1jMAKIJmPn7kADgD3yQZhpsqM-IRM1qZt)** | **[TensorFlow（密码peMe）](https://pan.iflytek.com:443/link/A136858D5F529E7C385C73EEE336F27B)** |
| **`BERT-wwm-ext, Chinese`** | **中文维基+<br/>通用数据<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1buMLEjdtrXE2c4G1rpsNGWEx7lUQ0RHi)** <br/>**[PyTorch](https://drive.google.com/open?id=1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_)** | **[TensorFlow（密码thGd）](https://pan.iflytek.com:443/link/8AA4B23D9BCBCBA0187EE58234332B46)** <br/>**[PyTorch（密码bJns）](https://pan.iflytek.com:443/link/4AB35DEBECB79C578BEC9952F78FB6F2)** |
| **`BERT-wwm, Chinese`** | **中文维基** | **[TensorFlow](https://drive.google.com/open?id=1RoTQsXp2hkQ1gSRVylRIJfQxJUgkfJMW)** <br/>**[PyTorch](https://drive.google.com/open?id=1AQitrjbvCWc51SYiLN-cJq4e0WiNN4KY)** | **[TensorFlow（密码mva8）](https://pan.iflytek.com:443/link/4B172939D5748FB1A3881772BC97A898)** <br/>**[PyTorch（密码8fX5）](https://pan.iflytek.com:443/link/8D4E8680433E6AD0F33D521EA920348E)** |
| `BERT-base, Chinese`<sup>Google</sup> | 中文维基 | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) | - |
| `BERT-base, Multilingual Cased`<sup>Google</sup>  | 多语种维基 | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) | - |
| `BERT-base, Multilingual Uncased`<sup>Google</sup>  | 多语种维基 | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip) | - |

> [1] 通用数据包括：百科、新闻、问答等数据，总词数达5.4B

以上预训练模型以TensorFlow版本的权重为准。
对于PyTorch版本，我们使用的是由Huggingface出品的[PyTorch-Transformers 1.0](https://github.com/huggingface/pytorch-transformers)提供的转换脚本。
如果使用的是其他版本，请自行进行权重转换。

中国大陆境内建议使用讯飞云下载点，境外用户建议使用谷歌下载点，base模型文件大小约**400M**。 
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


### 测试任务数据
我们提供部分任务数据，请查看`data`目录了解。
压缩包内包含训练和测试数据，同一目录下的`README.md`标明数据来源。
由于一部分数据需要原作者授权，故我们无法提供下载链接，敬请谅解。

## 模型对比
针对大家比较关心的一些模型细节进行汇总如下。`~BERT`表示**继承**谷歌原版中文BERT的属性。

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


## 中文基线系统效果
为了对比基线效果，我们在以下几个中文数据集上进行了测试，包括`句子级`和`篇章级`任务。
对于`BERT-wwm-ext`以及`RoBERTa-wwm-ext`，我们**没有进一步调整最佳学习率**，而是直接使用了`BERT-wwm`的最佳学习率。

**下面仅列举部分结果，完整结果请查看我们的[技术报告](https://arxiv.org/abs/1906.08101)。**

- [**CMRC 2018**：篇章片段抽取型阅读理解（简体中文）](https://github.com/ymcui/cmrc2018)
- [**DRCD**：篇章片段抽取型阅读理解（繁体中文）](https://github.com/DRCSolutionService/DRCD)
- [**CJRC**: 法律阅读理解（简体中文）](http://cail.cipsc.org.cn)
- [**XNLI**：自然语言推断](https://github.com/google-research/bert/blob/master/multilingual.md)
- [**NER**：中文命名实体识别](http://sighan.cs.uchicago.edu/bakeoff2006/)
- [**THUCNews**：篇章级文本分类](http://thuctc.thunlp.org)

**注意：为了保证结果的可靠性，对于同一模型，我们运行10遍（不同随机种子），汇报模型性能的最大值和平均值。不出意外，你运行的结果应该很大概率落在这个区间内。**


### 简体中文阅读理解：CMRC 2018
[**CMRC 2018数据集**](https://github.com/ymcui/cmrc2018)是哈工大讯飞联合实验室发布的中文机器阅读理解数据。
根据给定问题，系统需要从篇章中抽取出片段作为答案，形式与SQuAD相同。

| 模型 | 开发集 | 测试集 | 挑战集 |
| :------- | :---------: | :---------: | :---------: |
| BERT | 65.5 (64.4) / 84.5 (84.0) | 70.0 (68.7) / 87.0 (86.3) | 18.6 (17.0) / 43.3 (41.3) | 
| ERNIE | 65.4 (64.3) / 84.7 (84.2) | 69.4 (68.2) / 86.6 (86.1) | 19.6 (17.0) / 44.3 (42.8) | 
| **BERT-wwm** | 66.3 (65.0) / 85.6 (84.7) | 70.5 (69.1) / 87.4 (86.7) | 21.0 (19.3) / 47.0 (43.9) | 
| **BERT-wwm-ext** | 67.1 (65.6) / 85.7 (85.0) | 71.4 (70.0) / 87.7 (87.0) | 24.0 (20.0) / 47.3 (44.6) |
| **RoBERTa-wwm-ext** | **67.4 (66.5) / 87.2 (86.5)** | **72.6 (71.4) / 89.4 (88.8)** | **26.2 (24.6) / 51.0 (49.1)** |


### 繁体中文阅读理解：DRCD
[**DRCD数据集**](https://github.com/DRCKnowledgeTeam/DRCD)由中国台湾台达研究院发布，其形式与SQuAD相同，是基于繁体中文的抽取式阅读理解数据集。
**由于ERNIE中去除了繁体中文字符，故不建议在繁体中文数据上使用ERNIE（或转换成简体中文后再处理）。**

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 83.1 (82.7) / 89.9 (89.6) | 82.2 (81.6) / 89.2 (88.8) | 
| ERNIE | 73.2 (73.0) / 83.9 (83.8) | 71.9 (71.4) / 82.5 (82.3) | 
| **BERT-wwm** | 84.3 (83.4) / 90.5 (90.2) | 82.8 (81.8) / 89.7 (89.0) | 
| **BERT-wwm-ext** | 85.0 (84.5) / 91.2 (90.9) | 83.6 (83.0) / 90.4 (89.9) |
| **RoBERTa-wwm-ext** | **86.6 (85.9) / 92.5 (92.2)** | **85.6 (85.2) / 92.0 (91.7)** | 


### 司法阅读理解：CJRC
[**CJRC数据集**](http://cail.cipsc.org.cn)是哈工大讯飞联合实验室发布的面向**司法领域**的中文机器阅读理解数据。
需要注意的是实验中使用的数据并非官方发布的最终数据，结果仅供参考。

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 54.6 (54.0) / 75.4 (74.5) | 55.1 (54.1) / 75.2 (74.3) | 
| ERNIE | 54.3 (53.9) / 75.3 (74.6) | 55.0 (53.9) / 75.0 (73.9) | 
| **BERT-wwm** | 54.7 (54.0) / 75.2 (74.8) | 55.1 (54.1) / 75.4 (74.4) | 
| **BERT-wwm-ext** | 55.6 (54.8) / 76.0 (75.3) | 55.6 (54.9) / 75.8 (75.0) | 
| **RoBERTa-wwm-ext** | **58.7 (57.6) / 79.1 (78.3)** | **59.0 (57.8) / 79.0 (78.0)** |


### 自然语言推断：XNLI
在自然语言推断任务中，我们采用了**XNLI**数据。

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 77.8 (77.4) | 77.8 (77.5) | 
| ERNIE | 79.7 **(79.4)** | 78.6 (78.2) | 
| **BERT-wwm** | 79.0 (78.4) | 78.2 (78.0) | 
| **BERT-wwm-ext** | 79.4 (78.6) | 78.7 (78.3) |
| **RoBERTa-wwm-ext** | **80.0** (79.2) | **78.8 (78.3)** |


### 命名实体识别：人民日报、MSRA-NER
中文命名实体识别（NER）任务中，我们采用了经典的**人民日报数据**以及**微软亚洲研究院发布的NER数据**。
在这里我们只列F值，其他数值请参看技术报告。
*(因为这两个数据集在网上流传的版本甚多，建议在自己的数据上比对相对提升，以下数据仅供参考。)*

| 模型 | 人民日报 | MSRA-NER |
| :------- | :---------: | :---------: |
| BERT | 95.2 (94.9) | 95.3 (94.9) |  
| ERNIE | **95.7 (94.5)** | **95.4 (95.1)** |
| **BERT-wwm** | 95.3 (95.1) | **95.4 (95.1)** |


### 篇章级文本分类：THUCNews
篇章级文本分类任务我们选用了由清华大学自然语言处理实验室发布的新闻数据集**THUCNews**。
我们采用的是其中一个子集，需要将新闻分成10个类别中的一个。

| 模型 | 开发集 | 测试集 | 
| :------- | :---------: | :---------: | 
| BERT | 97.7 (97.4) | **97.8 (97.6)** |
| ERNIE | 97.6 (97.3) | 97.5 (97.3) |
| **BERT-wwm** | **98.0 (97.6)** | **97.8 (97.6)** |


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
A: 请查看data目录。对于有版权的内容，请自行搜索或与原作者联系获取数据。

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


## 引用
如果本目录中的内容对你的研究工作有所帮助，请在文献中引用下述技术报告：
https://arxiv.org/abs/1906.08101
```
@article{chinese-bert-wwm,
  title={Pre-Training with Whole Word Masking for Chinese BERT},
  author={Cui, Yiming and Che, Wanxiang and Liu, Ting and Qin, Bing and Yang, Ziqing and Wang, Shijin and Hu, Guoping},
  journal={arXiv preprint arXiv:1906.08101},
  year={2019}
 }
```


## 致谢
第一作者部分受到[**谷歌TensorFlow Research Cloud**](https://www.tensorflow.org/tfrc)计划资助。


## 免责声明
**本项目并非谷歌官方发布的Chinese BERT-wwm模型。同时，本项目不是哈工大或科大讯飞的官方产品。**

技术报告中所呈现的实验结果仅表明在特定数据集和超参组合下的表现，并不能代表各个模型的本质。
实验结果可能因随机数种子，计算设备而发生改变。

**该项目中的内容仅供技术研究参考，不作为任何结论性依据。使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。**


## 关注我们
欢迎关注哈工大讯飞联合实验室官方微信公众号。

![qrcode.png](https://github.com/ymcui/cmrc2019/raw/master/qrcode.jpg)


## 问题反馈
如有问题，请在GitHub Issue中提交。

