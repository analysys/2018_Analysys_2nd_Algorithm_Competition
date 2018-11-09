from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import pandas as pd
import os
path='../data/'
def read_csv(fname):
        file1=open(fname)
        data_=[]
        for line in file1:
                data_.append(line.replace("\n","").split(","))
        data = pd.DataFrame(data_[1:])
        data.columns=[data_[0]]
        return data

tr_P1=read_csv(path+'train/P1.csv')
tr_P2=read_csv(path+'train/P2.csv')
te_P1=read_csv(path+'test/P1.csv')
te_P2=read_csv(path+'test/P2.csv')

data_P1=pd.concat([tr_P1,te_P1])
data_P2=pd.concat([tr_P2,te_P2])

data = pd.DataFrame()
data['P1']=data_P1['P1']
data['P2']=data_P2['P2']
one_hot_feature=['P1','P2']

for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

print(data.shape)
train=data.iloc[[i for i in range(50000)]]
test=data.iloc[[i for i in range(50000,72727)]]
enc = OneHotEncoder()

enc.fit(data['P1'].values.reshape(-1, 1))
train_P1=enc.transform(train['P1'].values.reshape(-1, 1))
test_P1 = enc.transform(test['P1'].values.reshape(-1, 1))
sparse.save_npz('../data/train/train_P1.npz',train_P1)
sparse.save_npz('../data/test/test_P1.npz',test_P1)

enc.fit(data['P2'].values.reshape(-1, 1))
train_P2=enc.transform(train['P2'].values.reshape(-1, 1))
test_P2 = enc.transform(test['P2'].values.reshape(-1, 1))
sparse.save_npz('../data/train/train_P2.npz',train_P2)
sparse.save_npz('../data/test/test_P2.npz',test_P2)

print('one-hot prepared !')