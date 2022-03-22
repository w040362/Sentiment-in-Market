import pandas as pd
import sklearn
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as Data
from transformers import BertTokenizer
from bert_model import BertCNN


class SaDataset(Data.Dataset):
    def __init__(self, sentences, labels=None, with_labels=True):
        self.sentences = sentences
        self.labels = labels
        self.with_labels = with_labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        if self.with_labels:
            return sentence, label
        else:
            return sentence


def data_prepare(train_file, test_ratio, validate_ratio, mode='Train'):
    data = pd.read_csv(train_file, sep=',', encoding='utf-8')   # tsv: sep='\t'
    length = len(data)
    # data = sklearn.utils.shuffle(data)
    # data.to_csv('data/weibo_senti_100k_shuffle.csv', encoding='utf-8', index=0, sep=',')

    sentences = list(data['review'])
    labels = list(data['label'])

    split1 = int(test_ratio * length)
    split2 = int((test_ratio+validate_ratio) * length)
    if mode == 'Train':
        print('Training...')
        split = int(test_ratio * length)
        return sentences[:split], labels[:split]
    elif mode == 'Validation':
        print('Validating...')
        split1 = int(test_ratio * length)
        split2 = int((test_ratio + validate_ratio) * length)
        return sentences[split1:split2], labels[split1:split2]
    elif mode == 'Test':
        print('Testing...')
        split = int((test_ratio + validate_ratio) * length)
        return sentences[split:], labels[split:]


class Trainer:
    def __init__(self, bert_model, train_ratio, validate_ratio, train_file, freeze_bert):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.train_ratio = train_ratio
        self.validate_ratio = validate_ratio
        self.model = BertCNN(bert_model, freeze_bert, BERT_LENGTH).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate, weight_decay=weight_decay)  # ??
        self.loss_func = nn.CrossEntropyLoss()
        self.training_loss = []
        self.train_file = train_file

    def coffate_fn(self, examples):
        inputs, targets = [], []
        for sentence, label in examples:
            inputs.append(sentence)
            targets.append(int(label))
        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt",
                                max_length=BERT_LENGTH)
        targets = torch.tensor(targets)
        return inputs, targets

    def train_batch(self):
        train_sentences, train_labels \
            = data_prepare(self.train_file, self.train_ratio, self.validate_ratio, mode='Train')

        train_data = SaDataset(train_sentences, train_labels)
        train_dataloader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=self.coffate_fn)

        for epoch in range(EPOCH):
            for i, batch in tqdm(enumerate(train_dataloader),
                                 total=len(train_dataloader), desc=f"Training Epoch {epoch}", leave=True):
                ###
                # inputs, targets = [x for x in batch]
                # inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt",
                #                         max_length=BERT_LENGTH)
                # inputs.to(device)
                # targets = torch.tensor(targets).to(device)
                ###
                inputs, targets = [x.to(device) for x in batch]

                self.optimizer.zero_grad()
                out = self.model(inputs)

                loss = self.loss_func(out, targets)
                self.training_loss.append(loss.item())

                if i % 100 == 0:
                    print("Epoch: {}, batch: {}, loss: {:.4f}".format(epoch, i, loss))

                loss.backward()
                self.optimizer.step()

                del out, batch

            if (epoch + 1) % 5 == 0:
                self.test('Validation')
                torch.save(self.model.state_dict(), 'models/model-e' + str(epoch + 1) + '.model')

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    def test(self, mode):
        test_sentences, test_labels = data_prepare(self.train_file,
                                                   self.train_ratio, self.validate_ratio, mode=mode)
        test_data = SaDataset(test_sentences, test_labels)
        test_dataloader = Data.DataLoader(test_data, batch_size=1, collate_fn=self.coffate_fn)

        accuracy = 0.
        # for token in tqdm(test_dataloader, desc=f"Testing: ", leave=False):
        for token in test_dataloader:
            inputs, targets = [x.to(device) for x in token]
            with torch.no_grad():
                out = self.model(inputs)
            accuracy += (torch.max(out, dim=1)[1].item() == targets)

        accuracy = accuracy / len(test_dataloader)
        print('Accuracy: {}'.format(accuracy))

    def predict(self, text):
        token = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt",
                               max_length=BERT_LENGTH)
        with torch.no_grad():
            out = self.model(token.to(device))
        pred = torch.max(out, dim=1)[1].item()
        return pred


device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training hyper
BATCH_SIZE = 4
EPOCH = 20
learning_rate = 1e-5
weight_decay = 1e-2

BERT_LENGTH = 160

is_train = True
freeze_bert = False


if __name__ == '__main__':
    model = r'FinBERT_L-12_H-768_A-12_pytorch/'
    # bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
    # file = 'data/weibo_senti_100k_shuffle.csv'
    file = 'data/weibo_senti_100k_shuffle.csv'
    trainer = Trainer(bert_model=model, train_ratio=0.7, validate_ratio=0.15,
                      train_file=file, freeze_bert=freeze_bert)

    if is_train:
        trainer.train_batch()
    else:
        model_name = 'models/model-e50.model'
        trainer.load_model(model_name)
        # trainer.test('Test')
        res = trainer.predict('')
        print(res)
