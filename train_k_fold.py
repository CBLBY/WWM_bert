import logging
import torch
import argparse
import os
import sys
import math
import numpy as np
import pandas as pd
from time import strftime, localtime

from torch.utils.data import DataLoader, random_split, ConcatDataset
from pytorch_transformers import BertModel
from sklearn import metrics
import torch.nn as nn

from data_utils import MyDataset, Tokenizer4Bert
from models.bert_base import BertBase

# 获取logger, 设置返回level, 增加handler
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        # 模型实例化
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)
        self.pretrained_bert_state_dict = bert.state_dict()
        # 数据集实例化
        self.trainset = MyDataset(opt.dataset_file['train'], opt.max_length, opt.pretrained_bert_name)
        self.testset = MyDataset(opt.dataset_file['test'], opt.max_length, opt.pretrained_bert_name)

        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info('n_trainable_params: {0}, n_nontrainable_parmas: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('>>>>training_arguments')
        for arg in vars(self.opt):
            logger.info('>>>>>>> {0} : {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_parms(self):
        # 在整体模型下看子模型
        for child in self.model.children():
            if type(child) != BertModel:
                for p in child.parameters():
                    if p.requires_grad:
                        # 多维参数用 torch 的 initiate
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        # 一维参数用高斯分布初始化
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            nn.init.uniform_(p, -stdv, stdv)
        else:
            self.model.bert.load_state_dict(self.pretrained_bert_state_dict)


    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_f1 = 0
        global_step = 0
        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('>'*50 + 'start to train' + '>'*50)
            logger.info('epoch:{}'.format(epoch))
            n_total, loss_total = 0, 0
            n_correct = 0
            self.model.train()
            for i_batch, sample_batch in enumerate(train_data_loader):
                global_step += 1

                inputs = [sample_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batch['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss = loss / self.opt.accumulation_steps  # 梯度累计
                loss.backward()
                if ((i_batch + 1) % self.opt.accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total = loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_f1 = self._evaluate_f1(val_data_loader)
            logger.info('val_f1: {:.4f}'.format(val_f1))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_val_f1:{1}'.format(self.opt.model_name, round(val_f1, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>>> saved:{}'.format(path))

        return path, max_val_f1

    def _evaluate_f1(self, data_loader):
        n_total = 0
        self.model.eval()
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for i_batch, sample_batch in enumerate(data_loader):
                t_inputs = [sample_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = sample_batch['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_outputs_all = t_outputs
                    t_targets_all = t_targets
                else:
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)

        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return f1

    def _do_test(self, dataloader, fid):
        self.model.eval()
        t_outputs_all, t_guid_all = None, None
        with torch.no_grad():
            for i_batch, sample_batch in enumerate(dataloader):
                t_inputs = [sample_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_guid = sample_batch['guid']
                t_outputs = self.model(t_inputs)

                if t_outputs_all is None:
                    t_outputs_all = t_outputs
                    t_guid_all = t_guid
                else:
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
                    t_guid_all = np.concatenate((t_guid_all, t_guid), axis=0)
        guid = t_guid_all.reshape((-1, 1))
        target = t_outputs_all.cpu().numpy()
        target = target.reshape((-1, 3))
        result = np.concatenate((guid, target), axis=1)
        df = pd.DataFrame(result, columns=['id', 'label0','label1', 'label2'])
        df.to_csv('./WWM_bert/sub_temp/sub{}.csv'.format(fid))

    def run(self):
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        singset_len = len(self.trainset) // self.opt.cross_val_fold
        splitedsets = random_split(self.trainset, tuple([singset_len] * (self.opt.cross_val_fold - 1) +
                                                        [len(self.trainset) - singset_len * (self.opt.cross_val_fold - 1)]))
        all_val_f1 = []
        for fid in range(self.opt.cross_val_fold):
            logger.info('fold:{}'.format(fid))
            logger.info('>' * 100)
            trainset = ConcatDataset([x for i, x in enumerate(splitedsets) if i != fid])
            valset = splitedsets[fid]
            train_data_loader = DataLoader(dataset=trainset, batch_size=self.opt.batch_size, shuffle=True)
            val_data_loader = DataLoader(dataset=valset, batch_size=self.opt.batch_size, shuffle=False)

            self._reset_parms()
            best_model_path, max_val_f1 = self._train(criterion, optimizer, train_data_loader, val_data_loader)
            all_val_f1.append(max_val_f1)

            self.model.load_state_dict(torch.load(best_model_path))
            self.model.eval()
            self._do_test(test_data_loader, fid)
        mean_val_f1 = np.mean(all_val_f1)
        logger.info('>' * 100)
        logger.info('>>> mean_val_f1: {:.4f}'.format(mean_val_f1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_base', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float)    #5e-5, 2e-5
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.002, type=float)  # 尝试更小
    parser.add_argument('--num_epoch', default=8, type=int)
    parser.add_argument('--batch_size', default=8, type=int, help='尝试8的倍数')
    parser.add_argument('--accumulation_steps', default=16, type=int, help='real_batch_size = batch_size * accumulation_steps')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-chinese', type=str)
    parser.add_argument('--max_length', default=256, type=int, help='尝试适合的值减少使用内存')
    parser.add_argument('--polarity_dim', default=3, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--cross_val_fold', default=5, type=int)
    opt = parser.parse_args()

    model_classes = {
        'bert_base':BertBase
    }
    dataset_file = {
        'train':'./WWM_bert/datas/preprocessed_train_data.csv',
        'test':'./WWM_bert/datas/preprocessed_test_data.csv'
    }
    input_coles = {
        'bert_base':[ 'title_bert_indices', 'content_bert_indices']

    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.dataset_file = dataset_file
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_coles[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}.log'.format(opt.model_name, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()

if __name__ == '__main__':
    main()





















