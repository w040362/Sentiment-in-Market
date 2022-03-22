import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

# textCNN hyper_parameters
embed_size = 768
num_classes = 3     # positive or negative (1/0)
output_channel = 3  # for textCNN


class BertCNN(nn.Module):
    def __init__(self, model, freeze_bert, seq_len):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.freeze_bert = freeze_bert

        # testCNN classifier
        self.conv = nn.Sequential(
            nn.Conv2d(1, output_channel, kernel_size=(2, embed_size)),
            nn.ReLU()
        )
        self.fc = nn.Linear(output_channel, num_classes)        # fc after convolution
        # Linear classifier on [cls]
        self.classifier = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x['input_ids'], x['token_type_ids'], x['attention_mask']
        # print(input_ids.shape)
        if self.freeze_bert:
            with torch.no_grad():
                bert_outputs = self.bert(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)
        else:
            bert_outputs = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)
        # CNN layer
        last_h_state = bert_outputs.last_hidden_state   # (bs, seq_len, hidden_size)
        x = last_h_state.unsqueeze(1)   # channel
        batch_size = x.shape[0]

        conv_x = self.conv(x)
        pool_x = torch.max(conv_x, dim=2)[0]    # maxpool2D
        flat_x = pool_x.view(batch_size, -1)
        out = self.fc(flat_x)

        # Linear layer
        # cls = bert_outputs[0][:, 0, :]
        # out = self.classifier(self.dropout(cls))
        # print(out)
        return out
