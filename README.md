### Sentiment-in-Market
Create on 2022-1-16

bert_model.py & bert_train.py 是BERT+CNN/Linear的模型构建与训练文件

darnn_model.py & darnn_train.py是LSTM部分的代码

#### BERT

对于BERT部分，在transformers提供的框架下，使用熵简科技的预训练FinBERT模型，

​	https://github.com/valuesimplex/FinBERT

​	使用：将下载的FinBERT_L-12_H-768_A-12_pytorch目录置于项目文件夹下

​				is_train决定了训练/测试选项，没有命令行参数直接在代码中修改相应部分

2022-3-05 金融新闻训练集，三分类，0/1/2（负面-中性-正面）

​					bert分词长度 200；50epoch，acc=0.97；

​					对于评论的正确识别率较低				

​	训练集文件 merge.csv(test set + 0.8 train set)，无预处理	   

​	生成模型文件 model-e50.model

2022-03-22 bert模型修改

​	bert 不使用max_length padding，改用变长

​	CNN 支持变长操作，一个batch中长度相同

​	Linear 使用最后一个隐藏层的[CLS]，引入dropout=0.2

2022-04-29 换数据集，3700 sentence

​	accuracy 大约0.65

2022-06-27 细节调整

​	修改了model代码中的一处错误

​	完善了fine-tune的模型结构：textCNN增加卷积核种类，Linear从一层变为2层

​	acc 0.67
