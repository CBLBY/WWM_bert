import torch
import torch.nn as nn
import  numpy as np
import torch.nn.functional as F

class BertBase(nn.Module):
    def __init__(self, bert, opt):
        super(BertBase, self).__init__()
        self.bert = bert
        self.max_length = opt.max_length
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim*2, opt.polarity_dim)

    def forward(self, inputs):
        title_bert_indices, content_bert_indices = inputs[0], inputs[1]

        #bert_segement_ids = np.asarray([0] * self.max_length)#要转化为tensor

        _, title_pooled_output = self.bert(title_bert_indices)
        _, content_pooled_output = self.bert(content_bert_indices)
        pooled_output = torch.cat((title_pooled_output, content_pooled_output), dim=-1)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits
