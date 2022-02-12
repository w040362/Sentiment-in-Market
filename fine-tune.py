import pandas as pd
import numpy as np

import torch
import torch.utils.data as Data
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from model import textCNN

file = 'data/weibo_senti_100k.csv'
data = pd.read_csv(file, sep=',', encoding='utf-8')
sentences = list(data['review'])
labels = list(data['label'])


BATCH_SIZE = 1

model = r'FinBERT_L-12_H-768_A-12_pytorch/'
tokenizer = BertTokenizer.from_pretrained(model)
bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)


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


token_data = SaDataset(tokenizer, sentences, labels)
train_data = Data.DataLoader(token_data, batch_size=BATCH_SIZE, shuffle=True)

# for batch in train_data:
#     print(batch[0].shape) batch_size*max_length
#     print(batch[1].shape)
#     print(batch[2].shape)
#     print(batch[3].shape) batch_size, to be unsqueezed

model = textCNN()

for i, batch in enumerate(train_data):
    input_ids, attention_mask, token_type_ids = batch[0], batch[1], batch[2]
    outputs = bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    h_states = outputs.hidden_states
    last_h_state = outputs.last_hidden_state
    # print(outputs.last_hidden_state.shape)    # (batch_size, max_length, hidden_size)

    # cls_embedding = h_states[1][:, 0, :].unsqueeze(1)   # (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
    # for layer in range(2, 13):
    #     cls_embedding = torch.cat((cls_embedding, h_states[layer][:, 0, :].unsqueeze(1)), dim=1)
    # out = model(cls_embedding)

    # print(cls_embedding.shape)    # (batch_size, num_hidden_layers, hidden_size)
    out = model(last_h_state)

