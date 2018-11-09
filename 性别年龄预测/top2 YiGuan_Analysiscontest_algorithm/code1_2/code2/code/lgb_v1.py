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

train_apps_stat=sparse.load_npz(path+'train/train_app_stat.npz')
test_apps_stat=sparse.load_npz(path+'test/test_app_stat.npz')

train_P1=sparse.load_npz(path+'train/train_P1.npz')
test_P1=sparse.load_npz(path+'test/test_P1.npz')

train_app1=sparse.load_npz(path+'train/x_app1.npz')
test_app1=sparse.load_npz(path+'test/x_app1.npz')

train_app2=sparse.load_npz(path+'train/x_app2.npz')
test_app2=sparse.load_npz(path+'test/x_app2.npz')

train_app1_stat=sparse.load_npz(path+'train/use_apps_1.npz')
test_app1_stat=sparse.load_npz(path+'test/use_apps_1.npz')

train_install_app1_stat=sparse.load_npz(path+'train/install_apps_1.npz')
test_install_app1_stat=sparse.load_npz(path+'test/install_apps_1.npz')

train_base = pd.read_csv(path+"train/apps_base.csv")
test_base = pd.read_csv(path+"test/apps_base.csv")


train_lda = pd.read_csv(path+"train/lda_fea.csv")
test_lda = pd.read_csv(path+"test/lda_fea.csv")

train_week_start = pd.read_csv(path+"train/start_time_fea.csv")
test_week_start = pd.read_csv(path+"test/start_time_fea.csv")
train_week_close = pd.read_csv(path+"train/close_time_fea.csv")
test_week_close = pd.read_csv(path+"test/close_time_fea.csv")

print(train_base.shape,test_base.shape,train_apps_stat.shape,test_apps_stat.shape)

train_x = sparse.hstack([train_app1,train_week_start,train_week_close,train_base,train_apps_stat,train_lda,train_P1,train_app1_stat,train_install_app1_stat,train_app2], format='csr') 
test_x = sparse.hstack([test_app1,test_week_start,test_week_close,test_base,test_apps_stat,test_lda,test_P1,test_app1_stat,test_install_app1_stat,test_app2], format='csr') 

print(train_x.shape,test_x.shape)

gc.collect()
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 22,
    'metric': {'multi_logloss','acc'},
    'learning_rate': 0.03,
    'num_leaves': 45,  # we should let it be smaller than 2^(max_depth)
    'max_depth': -1,  # -1 means no limit
    'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 255,  # Number of bucketed bin for feature values
    'subsample': 0.8,  # Subsample ratio of the training instance.
    'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    }

num_folds = 10
predict = np.zeros((test_x.shape[0],22))
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=1996)
num=0
score=0
meta_train = np.zeros((train_x.shape[0],22))
test_preds=[]
for train_index, test_index in skf.split(train_x,labels):
	
	kfold_y_train,kfold_y_test = labels[train_index], labels[test_index]
	kfold_X_train = train_x[train_index]
	kfold_X_valid = train_x[test_index]
	lgb_train = lgb.Dataset(kfold_X_train, label=kfold_y_train)
	lgb_eval = lgb.Dataset(kfold_X_valid, label=kfold_y_test) 
	model = lgb.train(params, train_set=lgb_train, num_boost_round=10000,valid_sets=[lgb_eval], valid_names=['eval'], verbose_eval=20,early_stopping_rounds=50)
	score+=model.best_score['eval']['multi_logloss']
	val_pred = model.predict(kfold_X_valid)
	meta_train[test_index] = pd.DataFrame(val_pred).values

	predict =  np.array(model.predict(test_x))
	test_preds.append(predict)

columns_lists=['1-0','1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9','1-10','2-0','2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9','2-10']
meta_train = pd.DataFrame(meta_train)
meta_train.columns = columns_lists
meta_train.to_csv('../stack/meta_lgbv1_train.csv', index = None)
	
resCols = ['DeviceID','1-0','1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9','1-10','2-0','2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9','2-10']
test_df=pd.DataFrame()
test_df['DeviceID'] = [x for x in range(test_x.shape[0])]
preds=(test_preds[0]+test_preds[1]+test_preds[2]+test_preds[3]+test_preds[4]+test_preds[5]+test_preds[6]+test_preds[7]+test_preds[8]+test_preds[9])/10
for col in range(len(columns_lists)):
	test_df[columns_lists[col]]=preds[:,col].tolist()
test_df.to_csv('../stack/meta_lgbv1_test.csv',index=None)
print(score/10)
