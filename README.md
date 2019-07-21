## 中文全词覆盖BERT（Chinese BERT with Whole Word Masking）

**For English description, please read [README_EN.md](https://github.com/ymcui/Chinese-BERT-wwm/blob/master/README_EN.md) or our technical report on arXiv: https://arxiv.org/abs/1906.08101**

为了进一步促进中文自然语言处理的研究发展，我们提供了中文全词覆盖（Whole Word Masking）BERT的预训练模型。
同时在我们的技术报告中详细对比了当今流行的中文预训练模型：[BERT](https://github.com/google-research/bert)、[ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)、[BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)。
**更多细节请参考我们的技术报告：https://arxiv.org/abs/1906.08101**

![./pics/header.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/header.png)

**微信公众号文章介绍**

- 哈工大讯飞联合实验室：https://mp.weixin.qq.com/s/EE6dEhvpKxqnVW_bBAKrnA
- 机器之心：https://mp.weixin.qq.com/s/88OwaHqnrVMQ7vH98INA3w

本项目基于谷歌官方的BERT：https://github.com/google-research/bert

## 新闻
2019/6/20 初始版本，模型已可通过谷歌下载，国内云盘也已上传完毕，查看[中文模型下载](#中文模型下载)


## 内容导引
| 章节 | 描述 |
|-|-|
| [简介](#简介) | 介绍BERT-wwm |
| [中文模型下载](#中文模型下载) | 提供了BERT-wwm的下载地址 |
| [中文基线系统效果](#中文基线系统效果) | 列举了部分中文基线系统效果 |
| [使用建议](#使用建议) | 提供了若干使用中文预训练模型的建议 |
| [英文模型下载](#英文模型下载) | 谷歌官方的英文BERT-wwm下载地址 |
| [FAQ](#FAQ) | 常见问题答疑 |
| [引用](#引用) | 本目录的技术报告 |

## 简介
**Whole Word Masking (wwm)**，暂且翻译为`全词Mask`，是谷歌在2019年5月31日发布的一项BERT的升级版本，主要更改了原预训练阶段的训练样本生成策略。简单来说，原有基于WordPiece的分词方式会把一个完整的词切分成若干个词缀，在生成训练样本时，这些被分开的词缀会随机被mask。在`全词Mask`中，如果一个完整的词的部分WordPiece被mask，则同属该词的其他部分也会被mask，即`全词Mask`。

**需要注意的是，这里的mask指的是广义的mask（替换成[MASK]；保持原词汇；随机替换成另外一个词），并非只局限于单词替换成`[MASK]`标签的情况。更详细的说明及样例请参考：[issue-4](https://github.com/ymcui/Chinese-BERT-wwm/issues/4)**

同理，由于谷歌官方发布的`BERT-base , Chinese`中，中文是以**字**为粒度进行切分，没有考虑到传统NLP中的中文分词（CWS）。我们将全词Mask的方法应用在了中文中，使用了中文维基百科（包括简体和繁体）进行训练，并且使用了[哈工大LTP](http://ltp.ai)作为分词工具），即对组成同一个**词**的汉字全部进行Mask。

下述文本展示了`全词Mask`的生成样例（注意：为了方便理解，下述例子中只考虑替换成[MASK]标签的情况。）。

| 说明 | 样例 |
| :------- | :--------- |
| 原始文本 | 使用语言模型来预测下一个词的probability。 |
| 分词文本 | 使用 语言 模型 来 预测 下 一个 词 的 probability 。 |
| 原始Mask输入 | 使 用 语 言 [MASK] 型 来 [MASK] 测 下 一 个 词 的 pro [MASK] ##lity 。 |
| 全词Mask输入 | 使 用 语 言 [MASK] [MASK] 来 [MASK] [MASK] 下 一 个 词 的 [MASK] [MASK] [MASK] 。 |


## 中文模型下载
*   [**`BERT-base, Chinese (Whole Word Masking)`**](https://drive.google.com/open?id=1RoTQsXp2hkQ1gSRVylRIJfQxJUgkfJMW): 
    12-layer, 768-hidden, 12-heads, 110M parameters

#### TensorFlow版本（1.12、1.13、1.14测试通过）
- Google: [download_link_for_google_drive](https://drive.google.com/open?id=1RoTQsXp2hkQ1gSRVylRIJfQxJUgkfJMW)
- 讯飞云: [download_link_密码mva8](https://pan.iflytek.com:443/link/4B172939D5748FB1A3881772BC97A898)

#### PyTorch版本（请使用🤗 的[PyTorch-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) > 0.6，其他版本请自行转换）
- Google: [download_link_for_google_drive](https://drive.google.com/open?id=1NlMd5GRG97N5BYJHDQR79EU41fEfzMCv)
- 讯飞云: [download_link_密码m1CE](https://pan.iflytek.com:443/link/F23B12B39A3077CF1ED7A08DDAD081E3)

中国大陆境内建议使用讯飞云下载，境外用户建议使用谷歌下载点，文件大小约**400M**。
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


### 测试任务数据
我们提供部分任务数据，请查看`data`目录了解。
压缩包内包含训练和测试数据，同一目录下的`README.md`标明数据来源。
由于一部分数据需要原作者授权，故我们无法提供下载链接，敬请谅解。


## 中文基线系统效果
为了对比基线效果，我们在以下几个中文数据集上进行了测试，包括`句子级`和`篇章级`任务。

**下面仅列举部分结果，完整结果请查看我们的[技术报告](https://arxiv.org/abs/1906.08101)。**

- [**CMRC 2018**：篇章抽取型阅读理解（简体中文）](https://github.com/ymcui/cmrc2018)
- [**DRCD**：篇章抽取型阅读理解（繁体中文）](https://github.com/DRCSolutionService/DRCD)
- [**NER**：中文命名实体抽取](http://sighan.cs.uchicago.edu/bakeoff2006/)
- [**THUCNews**：篇章级文本分类](http://thuctc.thunlp.org)

**注意：为了保证结果的可靠性，对于同一模型，我们运行10遍（不同随机种子），汇报模型性能的最大值和平均值。不出意外，你运行的结果应该很大概率落在这个区间内。**


### CMRC 2018
[CMRC 2018数据集](https://github.com/ymcui/cmrc2018)是哈工大讯飞联合实验室发布的中文机器阅读理解数据。根据给定问题，系统需要从篇章中抽取出片段作为答案，形式与SQuAD相同。

![./pics/cmrc2018.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/cmrc2018.png)

### DRCD
[DRCD数据集](https://github.com/DRCKnowledgeTeam/DRCD)由中国台湾台达研究院发布，其形式与SQuAD相同，是基于繁体中文的抽取式阅读理解数据集。

![./pics/drcd.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/drcd.png)

### NER
中文命名实体识别（NER）任务中，我们采用了经典的人民日报数据以及微软亚洲研究院发布的NER数据。

![./pics/ner.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/ner.png)

### THUCNews
由清华大学自然语言处理实验室发布的新闻数据集，需要将新闻分成10个类别中的一个。

![./pics/thucnews.png](https://github.com/ymcui/Chinese-BERT-wwm/raw/master/pics/thucnews.png)

## 使用建议
* 初始学习率是非常重要的一个参数（不论是BERT还是其他模型），需要根据目标任务进行调整。
* ERNIE的最佳学习率和BERT/BERT-wwm相差较大，所以使用ERNIE时请务必调整学习率（基于以上实验结果，ERNIE需要的初始学习率较高）。
* 由于BERT/BERT-wwm使用了维基百科数据进行训练，故它们对正式文本建模较好；而ERNIE使用了额外的百度百科、贴吧、知道等网络数据，它对非正式文本（例如微博等）建模有优势。
* 在长文本建模任务上，例如阅读理解、文档分类，BERT和BERT-wwm的效果较好。
* 如果目标任务的数据和预训练模型的领域相差较大，请在自己的数据集上进一步做预训练。
* 如果要处理繁体中文数据，请使用BERT或者BERT-wwm。因为我们发现ERNIE的词表中几乎没有繁体中文。

## 英文模型下载
为了方便大家下载，顺便带上**谷歌官方发布的英文`BERT-large（wwm）`**模型：

*   **[`BERT-Large, Uncased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters

*   **[`BERT-Large, Cased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters


## FAQ
**Q: 这个模型怎么用？**
A: BERT怎么用，这个就怎么用。文本不需要经过分词，wwm只影响预训练过程，不影响下游任务的输入。

**Q: 请问有预训练代码提供吗？**
A: 很遗憾，我不能提供相关代码，实现可以参考 #10 #13。

**Q: XXXXX数据集在哪里下载？**
A: 请查看data目录。对于有版权的内容，请自行搜索或与原作者联系获取数据。

**Q: 会有计划发布更大模型吗？比如BERT-large-wwm版本？**</br>
A: 如果我们从实验中得到更好效果，会考虑发布更大的版本。

**Q: 你骗人！无法复现结果😂**</br>
A: 在下游任务中，我们采用了最简单的模型。比如分类任务，我们直接使用的是`run_classifier.py`（谷歌提供）。如果无法达到平均值，说明实验本身存在bug，请仔细排查。最高值存在很多随机因素，我们无法保证能够达到最高值。

**Q: 我训出来比你更好的结果！**</br>
A: 恭喜你。

**Q: 训练花了多长时间，在什么设备上训练的？**</br>
A: 训练是在谷歌TPU v3版本（128G HBM）完成的，大约需要1.5天左右。需要注意的是，预训练阶段我们使用的是`LAMB Optimizer`（[TensorFlow版本实现](https://github.com/ymcui/LAMB_Optimizer_TF)）。该优化器对大的batch有良好的支持。在微调下游任务时，我们采用的是BERT默认的`AdamWeightDecayOptimizer`。

**Q: ERNIE是谁？**</br>
A: 本项目中的ERNIE模型特指百度公司提出的[ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)，而非清华大学在ACL 2019上发表的[ERNIE](https://github.com/thunlp/ERNIE)。

**Q: 你们这个和百度的ERNIE有什么区别？**</br>
A: 因为百度ERNIE的提出先于谷歌提出whole word masking（仅以公开相关工作的时间为基准），基于全词mask的方法应该是百度的相关工作在先。从数据上看，ERNIE采用了更多的网络数据（百科，贴吧，新闻），而本项目中只使用了中文维基百科数据。

**Q: 你们在实验中使用了ERNIE，是怎么用的呢？**</br>
A: 我们将ERNIE从PaddlePaddle格式转换为TensorFlow格式，并加载到下游任务的代码中。很遗憾，目前我们不能提供PP转TF/PT的代码，但GitHub中有一些开源的实现，可以搜索关注一下。同时，因为版权原因，我们不会提供TensorFlow/PyTorch版本的ERNIE权重供大家下载。关于ERNIE在PaddlePaddle中使用的相关问题，请咨询[ERNIE官方](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)。

**Q: BERT-wwm的效果不是在所有任务都很好**</br>
A: 本项目的目的是为研究者提供多元化的预训练模型，自由选择BERT，ERNIE，或者是BERT-wwm。我们仅提供实验数据，具体效果如何还是得在自己的任务中不断尝试才能得出结论。多一个模型，多一种选择。

**Q: 为什么有些数据集上没有试？**</br>
A: 很坦率的说：1）没精力找更多的数据；2）没有必要； 3）没有钞票；

**Q: 简单评价一下这几个模型**</br>
A: 各有侧重，各有千秋。中文自然语言处理的研究发展需要多方共同努力。

**Q: 你预测下一个预训练模型叫什么？**</br>
A: 可能叫ZOE吧，ZOE: Zero-shOt Embeddings from language model


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


## 免责声明
**本项目并非谷歌官方发布的Chinese BERT-base (wwm)。同时，本项目不是哈工大或科大讯飞的官方产品。**

技术报告中所呈现的实验结果仅表明在特定数据集和超参组合下的表现，并不能代表各个模型的本质。
实验结果可能因随机数种子，计算设备而发生改变。
由于我们没有直接在PaddlePaddle上使用ERNIE，所以在ERNIE上的实验结果仅供参考（虽然我们在多个数据集上复现了效果）。

**该项目中的内容仅供技术研究参考，不作为任何结论性依据。使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。**


## 致谢
第一作者部分受到[谷歌TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc)计划的支持。


## 关注我们
欢迎关注哈工大讯飞联合实验室官方微信公众号。

![qrcode.png](https://github.com/ymcui/cmrc2019/blob/master/qrcode.jpg)


## 问题反馈
如有问题，请在GitHub Issue中提交。

