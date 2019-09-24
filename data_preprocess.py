import pandas as pd
import os
import re

def filter(text):
    html_tags = re.compile('<.*?>')
    text = re.sub(html_tags, '', text)

    return text

def data_prprocess(train_data_path, label_data_path, test_data_path):
    train_data = pd.read_csv(train_data_path)
    label_data = pd.read_csv(label_data_path)
    # 拼接
    train_data = train_data.merge(label_data, on='id', how='left')
    # 处理空白内容
    # train_data['label'] = train_data['label'].fillna(-1)
    # train_data = train_data[train_data['label'!= -1]]
    train_data['content'] = train_data['content'].fillna('无')
    train_data['title'] = train_data['title'].fillna('无')
    train_data = train_data.dropna()
    train_data['label'] = train_data['label'].astype(int)
    # 数据清洗
    train_data['content'] = train_data['content'].apply(lambda x: filter(x))
    train_data['title'] = train_data['title'].apply(lambda x: filter(x))

    test_data = pd.read_csv(test_data_path)
    test_data['content'] = test_data['content'].fillna('无')
    test_data['title'] = test_data['title'].fillna('无')
    # 数据清洗
    test_data['content'] = test_data['content'].apply(lambda x: filter(x))
    test_data['title'] = test_data['title'].apply(lambda x: filter(x))

    train_data.to_csv('./WWM_bert/datas/preprocessed_train_data.csv')
    test_data.to_csv('./WWM_bert/datas/preprocessed_test_data.csv')

if not os.path.exists('./WWM_bert/datas/preprocessed_train_data.csv'):
    data_prprocess('./WWM_bert/datas/Train_DataSet.csv', './WWM_bert/datas/Train_DataSet_Label.csv',
                   './WWM_bert/datas/Test_DataSet.csv')