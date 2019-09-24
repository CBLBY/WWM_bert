import torch
import torch.nn as nn
import numpy as np

class DynamicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type, num_layers=1, bias=True, batch_first=True, dropout=0):
        '''

        :param input_size:
        :param hidden_size:
        :param rnn_type: it can be 'LSTM','GRU' and 'RNN'
        :param num_layers:
        :param bias:
        :param batch_first:
        :param dropout:
        '''
        super(DynamicRNN, self).__init__()
        self.inputsize = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bias=bias,
                               batch_first=batch_first, dropout=dropout)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bias=bias,
                               batch_first=batch_first, dropout=dropout)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bias=bias,
                               batch_first=batch_first, dropout=dropout)

    def forward(self, x, x_len): # x->(batch_size, sequence_length, embeeding_size) s_len is the lengrh of each sentence
        '''sort
        example:
            x : [3, 2, 4] -> x_sort : [4, 3, 2] descending_sort
            sort_idx : [2, 0, 1] -> sort_idx_sort : [0, 1, 2] acceding sort
            unsort_idx : [1, 2, 0]
            s_sort[s_unsort] == x
        '''
        x_sort_idx = torch.sort(-x_len)[1].long() # descending sort
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]

        # pack
        x_emb_p = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            # out_pack(batch_size,sequence_length, hidden_size) , ht(num_layers * num_directions, batch, hidden_size)
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None

        # unsort
        ht = torch.transpose(ht, 0, 1)[x_unsort_idx]
        ht = torch.transpose(ht, 0, 1)

        # unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
        out = out[x_unsort_idx]

        if self.rnn_type == 'LSTM':
            ct = torch.transpose(ct, 0, 1)[x_unsort_idx]
            ct = torch.transpose(ct, 0, 1)

        return out, (ht, ct)




