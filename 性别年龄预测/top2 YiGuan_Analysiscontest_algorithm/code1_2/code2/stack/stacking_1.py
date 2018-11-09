# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import gc
import logging
import os
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from scipy import sparse
from tqdm import tqdm

path='../data/'
tr_label=open(path+'train/label.csv')
train_label=[]
for line in tr_label:
	train_label.append(line.replace("\n","").split(","))
del tr_label
gc.collect()
train_label = pd.DataFrame(train_label[1:])
train_label.columns=['label']
print(train_label.shape)
labels=train_label.label.values
columns_=['1-0','1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9','1-10','2-0','2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9','2-10']

def read_csv(filename):
	file_=open(filename)
	file_text=[]
	for line in file_:
		file_text.append(line.replace("\n","").split(","))
	file_ = pd.DataFrame(file_text[1:])
	file_.columns=file_text[0]
	return file_

train_meta1=read_csv('./features/GBM272_train.csv')[columns_]
test_meta1=read_csv('./features/GBM272_test.csv')[columns_]
train_meta2=read_csv('./features/GBM2724_train.csv')[columns_]
test_meta2=read_csv('./features/GBM2724_test.csv')[columns_]
train_meta3=read_csv('./features/GBM273_train.csv')[columns_]
test_meta3=read_csv('./features/GBM273_test.csv')[columns_]
train_meta4=read_csv('./features/GBM2735_train.csv')[columns_]
test_meta4=read_csv('./features/GBM2735_test.csv')[columns_]
train_meta5=read_csv('./features/meta_lgbv1_train.csv')[columns_]
test_meta5=read_csv('./features/meta_lgbv1_test.csv')[columns_]

train_meta6=read_csv('./features/meta_nnv1_train.csv')[columns_]
test_meta6=read_csv('./features/meta_nn1_test.csv')[columns_]

train_meta9=np.load('./features/applist_train.npy')
test_meta9=np.load('./features/applist_test.npy')
train_meta10=np.load('./features/b1_train.npy')
test_meta10=np.load('./features/b1_test.npy')
train_meta11=np.load('./features/h1_train.npy')
test_meta11=np.load('./features/h1_test.npy')
train_meta14=np.load('./features/label_train.npy')
test_meta14=np.load('./features/label_test.npy')
train_meta18=np.load('./features/lgbtfidf_train.npy')
test_meta18=np.load('./features/lgbtfidf_test.npy')
train_meta20=np.load('stack_train.npy')
test_meta20=np.load('stack_test.npy')

train_meta21=read_csv('./features/meta_nnv2_train.csv')[columns_]
test_meta21=read_csv('./features/meta_nn2_test.csv')[columns_]

train_meta23=read_csv('./meta_nnv5_train.csv')[columns_]
test_meta23=read_csv('./meta_nn5_test.csv')[columns_]

train_x = np.hstack([train_meta1,train_meta2,train_meta3,train_meta4,train_meta5,train_meta6,train_meta7,train_meta9,train_meta10,n_meta14,train_meta18,train_meta20,train_meta21,train_meta23]) 
test_x = np.hstack([test_meta1,test_meta2,test_meta3,test_meta4,test_meta5,test_meta6,test_meta7,test_meta9,test_meta10,test_meta11,test_meta14,test_meta18,test_meta20,test_meta21,test_meta23]) 

testIDs =read_csv('./features/GBM272_train.csv')['DeviceID']

gc.collect()
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 22,
    'metric': {'multi_logloss'},
    'learning_rate': 0.01,
    'num_leaves': 12,
    'max_depth':  4,
    'min_child_samples': 20,
    'max_bin': 255,
    'subsample': 1,
    'colsample_bytree': 0.6,
    'min_child_weight': 5,
    'subsample_for_bin': 200000,
    }
num_folds = 10
all_loss=[]
predict = np.zeros((train_x.shape[0],6))
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=1996)
num=0
score=0
for train_index, test_index in skf.split(train_x,labels):
	num+=1
	print('----train_index----',train_index)
	print('----test_index----',test_index)
	predict_val = np.zeros((test_index.shape[0],6))
	kfold_y_train,kfold_y_test = labels[train_index], labels[test_index]
	kfold_X_train = train_x[train_index]
	kfold_X_valid = train_x[test_index]
	lgb_train = lgb.Dataset(kfold_X_train, label=kfold_y_train)
	lgb_eval = lgb.Dataset(kfold_X_valid, label=kfold_y_test) 
	model = lgb.train(params, train_set=lgb_train, num_boost_round=10000,valid_sets=[lgb_eval], valid_names=['eval'], verbose_eval=20,early_stopping_rounds=200)
	print(model.best_score)
	score+=model.best_score['eval']['multi_logloss']
	predict =  np.array(model.predict(test_x))
	resCols = ['DeviceID','1-0','1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9','1-10','2-0','2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9','2-10']
	res = pd.DataFrame(predict,columns=['1-0','1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9','1-10','2-0','2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9','2-10'])
	res['DeviceID'] = testIDs
	res[resCols].to_csv('../submit/submit_testv1_'+str(num)+'.csv',index=None) 
print(score/10)
all_loss.append(score/10)
print(all_loss)