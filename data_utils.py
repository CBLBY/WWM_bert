import torch
import numpy as np
import pandas as pd
import os
import pickle
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer


class Tokenizer4Bert:
    def __init__(self, max_length, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_length = max_length

    def text_to_sequence(self, text):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        return sequence_process(sequence, self.max_length) # return pad and turncated sequence

def sequence_process(sequence, max_length):
    '''
    To pad and turncate sequence
    :param sequence:
    :param max_length:
    :return:
    '''
    if(len(sequence) <= max_length):
        x = (np.ones(max_length) * 0).astype('int64')
        trunc = sequence[:max_length]
        trunc = np.asarray(trunc, dtype='int64')
        x[:len(trunc)] = trunc
    else:
        x = (np.ones(max_length) * 0).astype('int64')
        pre = sequence[:int(max_length/2)]
        post = sequence[-(max_length - int(max_length/2)): ]
        pre = np.asarray(pre, dtype='int64')
        post = np.asarray(post, dtype='int64')
        x[:int(max_length/2)] = pre
        x[-(max_length - int(max_length/2)): ] = post
    return x

class MyDataset(Dataset):
    def __init__(self, fname, max_length, pretrained_bert_name):
        self.max_length = max_length
        self.pretrained_bert_name = pretrained_bert_name
        self.fname = fname

        # guid=[];content=[];title=[];polarity=[]
        df = pd.read_csv(self.fname)
        tokenizer = Tokenizer4Bert(self.max_length, self.pretrained_bert_name)
        all_data = []

        if 'train' in self.fname:
            for val in df[['id', 'content', 'title', 'label']].values:
                guid = val[0];content = val[1];title = val[2];polarity = val[3]
    ############################################################################################################
                text_bert_indices = tokenizer.text_to_sequence('[CLS]' + title + '[SEP]' + content + '[SEP]')
                bert_segement_ids = np.asarray([0] * (len(title) + 2) + [1] * (len(content) + 1))
                bert_segement_ids = sequence_process(bert_segement_ids, max_length)
    ############################################################################################################
                title_bert_indices = tokenizer.text_to_sequence('[CLS]' + title + '[SEP]')
                content_bert_indices = tokenizer.text_to_sequence('[CLS]' + content + '[SEP]')

                data = {'text_bert_indices': text_bert_indices,
                        'bert_segemrnt_ids' : bert_segement_ids,
                        'guid' : guid,
                        'polarity' : polarity,
                        'title_bert_indices' : title_bert_indices,
                        'content_bert_indices' : content_bert_indices
                        }
                all_data.append(data)
        else:
            for val in df[['id', 'content', 'title']].values:
                guid = val[0]
                content = val[1]
                title = val[2]

                ############################################################################################################
                text_bert_indices = tokenizer.text_to_sequence('[CLS]' + title + '[SEP]' + content + '[SEP]')
                bert_segement_ids = np.asarray([0] * (len(title) + 2) + [1] * (len(content) + 1))
                bert_segement_ids = sequence_process(bert_segement_ids, max_length)
                ############################################################################################################
                title_bert_indices = tokenizer.text_to_sequence('[CLS]' + title + '[SEP]')
                content_bert_indices = tokenizer.text_to_sequence('[CLS]' + content + '[SEP]')

                data = {'text_bert_indices': text_bert_indices,
                        'bert_segemrnt_ids': bert_segement_ids,
                        'guid': guid,
                        'title_bert_indices': title_bert_indices,
                        'content_bert_indices': content_bert_indices
                        }
                all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
















