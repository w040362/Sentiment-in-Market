import pandas as pd
import numpy as np
import sklearn

import torch
import torch.nn as nn
import torch.utils.data as Data
from transformers import BertModel, BertTokenizer
from model import textCNN, BertCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training hyper
BATCH_SIZE = 64
EPOCH = 20
learning_rate = 1e-3
weight_decay = 1e-2


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
                                            truncation=True, max_length=160, return_tensors="pt")

        token_ids = sentence_tokenized['input_ids'].squeeze(0)
        attn_masks = sentence_tokenized['attention_mask'].squeeze(0)
        token_type_ids = sentence_tokenized['token_type_ids'].squeeze(0)

        if self.with_labels:
            label = self.labels[idx]
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids


def data_prepare(train_file, test_ratio, train=True):
    data = pd.read_csv(train_file, sep=',', encoding='utf-8')
    length = len(data)
    # data = sklearn.utils.shuffle(data)
    # data.to_csv('data/weibo_senti_100k_shuffle.csv', encoding='utf-8', index=0, sep=',')

    sentences = list(data['review'])
    labels = list(data['label'])

    split = int(test_ratio * length)
    if train:
        return sentences[:split], labels[:split]
    else:
        return sentences[split:], labels[split:]


class Trainer:
    def __init__(self, bert_model, test_ratio, train_file):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.test_ratio = test_ratio
        self.model = BertCNN(bert_model, device).to(device)
        self.optimizer = torch.optim.Adam(self.model.textCNN.parameters(),
                                          lr=learning_rate, weight_decay=weight_decay)  # ??
        self.loss_func = nn.CrossEntropyLoss()
        self.training_loss = []
        self.train_file = train_file

    def train_batch(self):
        train_sentences, train_labels = data_prepare(self.train_file, self.test_ratio, train=True)

        token_data = SaDataset(self.tokenizer, train_sentences, train_labels)
        train_data = Data.DataLoader(token_data, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in range(EPOCH):
            for i, batch in enumerate(train_data):
                out = self.model(torch.cat((batch[0].unsqueeze(0), batch[1].unsqueeze(0), batch[2].unsqueeze(0)),
                                           dim=0).to(device))

                cla = batch[3]
                loss = self.loss_func(out, cla.to(device))
                self.training_loss.append(loss)

                if i % 40 == 0:
                    print("Epoch: {}, batch: {}, loss: {:.4f}".format(epoch, i, loss))

                if (epoch + 1) % 5 == 0:
                    torch.save(self.model.textCNN.state_dict(), 'models/textCNN-e' + str(epoch + 1) + '.model')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                del out, batch

    def load_model(self, model_path):
        self.model.textCNN.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    def test(self):
        test_sentences, test_labels = data_prepare(self.train_file, self.test_ratio, train=False)
        print(test_labels[0])
        test_token_data = SaDataset(self.tokenizer, test_sentences, test_labels)

        total_count = 0
        match_count = 0

        print('Computing...')
        for token in test_token_data:
            out = self.model(torch.cat((token[0].unsqueeze(0), token[1].unsqueeze(0), token[2].unsqueeze(0)),
                                       dim=0).unsqueeze(1).to(device))  # 1-dim data, dim 1 for batch_size: 1
            pred = torch.max(out, dim=1)[1].item()

            total_count += 1
            if pred == token[3]:
                match_count += 1
            if total_count % 1000 == 0:
                print("{} finished".format(total_count))
        accuracy = match_count / total_count
        print('Accuracy: {}'.format(accuracy))

    def predict(self, text):
        token_text = SaDataset(self.tokenizer, text, with_labels=False)
        x = token_text[0]
        out = self.model(torch.cat((x[0].unsqueeze(0), x[1].unsqueeze(0), x[2].unsqueeze(0)),
                                   dim=0).unsqueeze(1).to(device))
        pred = torch.max(out, dim=1)[1].item()
        return pred


# # for batch in train_data:
# #     print(type(batch))
# #     print(batch[0].shape)   # batch_size*max_length
# #     print(type(batch[0]))
# #     print(batch[1].shape)
# #     print(batch[2].shape)
# #     print(batch[3].shape)   # batch_size, to be unsqueezed
#
# for epoch in range(EPOCH):
#     for i, batch in enumerate(train_data):
#         out = model(torch.cat((batch[0].unsqueeze(0), batch[1].unsqueeze(0), batch[2].unsqueeze(0)),
#                               dim=0))
#
#         cla = batch[3]
#         loss = loss_func(out, cla)
#         training_loss.append(loss)
#
#         if i % 20 == 0:
#             print("Epoch: {}, batch: {}, loss: {:.4f}".format(epoch, i, loss))
#
#         if (epoch + 1) % 50 == 0:
#             torch.save(model.textCNN.state_dict(), 'models/textCNN-e' + str(epoch + 1) + '.model')
#
#         # if epoch % 100 == 0:
#         #     test_out = model(test_data)
#         #     test_predict = torch.max(test_out, 1)[1].data.numpy()
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # input_ids, attention_mask, token_type_ids = batch[0], batch[1], batch[2]
#         # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         # h_states = outputs.hidden_states
#         # last_h_state = outputs.last_hidden_state
#         # print(outputs.last_hidden_state.shape)    # (batch_size, max_length, hidden_size)
#
#         # cls_embedding = h_states[1][:, 0, :].unsqueeze(1)   # (batch_size, hidden_size)->(batch_size, 1, hidden_size)
#         # for layer in range(2, 13):
#         #     cls_embedding = torch.cat((cls_embedding, h_states[layer][:, 0, :].unsqueeze(1)), dim=1)
#         # out = model(cls_embedding)
#
#         # print(cls_embedding.shape)    # (batch_size, num_hidden_layers, hidden_size)


if __name__ == '__main__':
    model = r'FinBERT_L-12_H-768_A-12_pytorch/'
    # bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
    file = 'data/weibo_senti_100k_shuffle.csv'
    trainer = Trainer(bert_model=model, test_ratio=0.7, train_file=file)

    is_train = False
    if is_train:
        trainer.train_batch()
    else:
        CNN_model_name = 'models/textCNN-e20.model'
        trainer.load_model(CNN_model_name)
        trainer.test()
        # res = trainer.predict('今天好'.encode('utf-8'))
        # print(res)
