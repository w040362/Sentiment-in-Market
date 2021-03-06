import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

# textCNN hyper_parameters
embed_size = 768
num_classes = 3     # positive or negative (1/0)
seq_len = 160
output_channel = 3  # for textCNN


class BertCNN(nn.Module):
    def __init__(self, model, freeze_bert):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.freeze_bert = freeze_bert

        self.conv2 = nn.Sequential(     # size 2 kernel
            nn.Conv2d(1, output_channel, kernel_size=(2, embed_size)),
            nn.ReLU(),
            nn.MaxPool2d((seq_len - 2 + 1, 1))
        )
        self.conv3 = nn.Sequential(     # size 3 kernel
            nn.Conv2d(1, output_channel, kernel_size=(3, embed_size)),
            nn.ReLU(),
            nn.MaxPool2d((seq_len - 3 + 1, 1))
        )
        self.fc = nn.Linear(2*output_channel, num_classes)        # fc after convolution

        self.classifier = nn.Linear(embed_size, num_classes)    # Linear classifier on [cls]
        self.classifier2 = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        input_ids, token_type_ids, attention_mask = x['input_ids'], x['token_type_ids'], x['attention_mask']
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

        conv_x2 = self.conv2(x)
        conv_x3 = self.conv3(x)
        conv_x = torch.cat([conv_x2, conv_x3], dim=1)
        flat_x = conv_x.view(batch_size, -1)
        out = self.fc(flat_x)

        # Linear layer
        # cls = bert_outputs.pooler_output
        # out = self.classifier(cls)
        # print(out)
        return out
