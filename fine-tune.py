import pandas as pd
import numpy as np
import sklearn

import torch
import torch.nn as nn
import torch.utils.data as Data
from transformers import BertModel, BertTokenizer
from model import textCNN, BertCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file = 'data/weibo_senti_100k_shuffle.csv'      # 119988 records
data = pd.read_csv(file, sep=',', encoding='utf-8')
length = len(data)
# data = sklearn.utils.shuffle(data)
# data.to_csv('data/weibo_senti_100k_shuffle.csv', encoding='utf-8', index=0, sep=',')

sentences = list(data['review'])
labels = list(data['label'])

split = int(0.8*length)
train_sentences = sentences[:split]
train_labels = labels[:split]
test_sentences = sentences[split:]
test_labels = labels[split:]


# training hyper
BATCH_SIZE = 4
EPOCH = 300
learning_rate = 1e-3
weight_decay = 1e-2

bert_model = r'FinBERT_L-12_H-768_A-12_pytorch/'
tokenizer = BertTokenizer.from_pretrained(bert_model)
# bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)


class SaDataset(Data.Dataset):
    def __init__(self, tokenizer, sentences, labels=None, with_labels=True):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.labels = labels
        self.with_labels = with_labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        sentence_tokenized = self.tokenizer(sentence, padding='max_length',
                                            truncation=True, max_length=256, return_tensors="pt")

        token_ids = sentence_tokenized['input_ids'].squeeze(0)
        attn_masks = sentence_tokenized['attention_mask'].squeeze(0)
        token_type_ids = sentence_tokenized['token_type_ids'].squeeze(0)

        if self.with_labels:
            label = self.labels[idx]
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids


token_data = SaDataset(tokenizer, train_sentences, train_labels)
train_data = Data.DataLoader(token_data, batch_size=BATCH_SIZE, shuffle=True)

# test_token_data = SaDataset(tokenizer, test_sentences, test_labels)
# test_data_input = [test_token_data[:][i] for i in range(3)]
# test_data_res = test_token_data[:][3]
# print(test_token_label.shape)

# for batch in train_data:
#     print(type(batch))
#     print(batch[0].shape)   # batch_size*max_length
#     print(type(batch[0]))
#     print(batch[1].shape)
#     print(batch[2].shape)
#     print(batch[3].shape)   # batch_size, to be unsqueezed


# training
model = BertCNN(bert_model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_func = nn.CrossEntropyLoss()
training_loss = []

# text = '我今天很高兴'
# token_text = SaDataset(tokenizer, text, with_labels=False)
# x = token_text[0]
# x = [p.unsqueeze(0) for p in x]
# predict = model(x)
# print(predict)

for epoch in range(EPOCH):
    for i, batch in enumerate(train_data):
        out = model(batch[:3])

        cla = batch[3]
        loss = loss_func(out, cla)
        training_loss.append(loss)

        if i % 2 == 0:
            print("Epoch: {}, batch: {}, loss: {:.4f}".format(epoch, i, loss))

        # if epoch % 100 == 0:
        #     test_out = model(test_data)
        #     test_predict = torch.max(test_out, 1)[1].data.numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # input_ids, attention_mask, token_type_ids = batch[0], batch[1], batch[2]
        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # h_states = outputs.hidden_states
        # last_h_state = outputs.last_hidden_state
        # print(outputs.last_hidden_state.shape)    # (batch_size, max_length, hidden_size)

        # cls_embedding = h_states[1][:, 0, :].unsqueeze(1)   # (batch_size, hidden_size)->(batch_size, 1, hidden_size)
        # for layer in range(2, 13):
        #     cls_embedding = torch.cat((cls_embedding, h_states[layer][:, 0, :].unsqueeze(1)), dim=1)
        # out = model(cls_embedding)

        # print(cls_embedding.shape)    # (batch_size, num_hidden_layers, hidden_size)


