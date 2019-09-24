import pandas as pd
import numpy as np

k = 5
df = pd.read_csv('./WWM_bert/datas/submit_example.csv')
df['0']=0
df['1']=0
df['2']=0
for i in range(k):
    temp = pd.read_csv('./WWM_bert/sub_temp/sub{}.csv'.format(i))
    df['0'] += temp['label0']/k
    df['1'] += temp['label1']/k
    df['2'] += temp['label2']/k

df['label'] = np.argmax(df[['0', '1', '2']].values, -1)
df[['id','label']].to_csv('./WWM_bert/final_sub.csv', index=False)
