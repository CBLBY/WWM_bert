from layers.dynamic_rnn import DynamicRNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class BertHAN(nn.Module):
    def __int__(self, bert, opt):
        super(BertHAN, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.rnn = DynamicRNN(opt.bert_dim, opt.hidden_size, 'GRU')
        self.dense = nn.Linear(opt.hidden_size * 2, opt.polarity_dim)
        self.hidden_size = opt.hidden_size
        self.batch_size = opt.batch_size

    def forward(self, inputs):
        title, content = inputs[0], inputs[1]
        title_len = torch.sum(title != 0, dim=1)
        content_len = torch.sum(content != 0, dim=1)
        content_length_max = torch.max(content_len)
        # embedding
        content, _ = self.bert(content, output_all_encoded_layers=False)
        title, _ = self.bert(title, output_all_encoded_layers=False)
        # GRU layer
        content_output, _ = self.rnn(content, content_len)
        _, h_title = self.rnn(title, title_len) # h_title:(batch_size, hidden_size)
        # 层次注意力
        '''参数问题'''
        weight_m = nn.Parameter(torch.Tensor(self.batch_size, self.hidden_size * 2, self.hidden_size * 2))
        weight_r = nn.Parameter(torch.Tensor(self.batch_size, 1, self.hidden_size *2))

        h_title = torch.unsqueeze(h_title, dim=1).expand(-1, content_length_max, -1)
        H = torch.cat(content_output, h_title)
        m = F.tanh(torch.bmm(weight_m, H))
        att = F.softmax(torch.bmm(weight_r, m))
        r = torch.bmm(att, H)
        r = torch.transpose(r, 0, 1)




