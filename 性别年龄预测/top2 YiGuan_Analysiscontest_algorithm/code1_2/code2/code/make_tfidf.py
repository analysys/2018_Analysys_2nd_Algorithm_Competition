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

path='../data/'

tr_app=open(path+'train/app_list.csv')
te_app=open(path+'test/app_list.csv')

train_app=[]
test_app=[]
for line in tr_app:
	train_app.append(line.replace("\n",""))
for line in te_app:
	test_app.append(line.replace("\n",""))
train_app = pd.DataFrame(train_app)
train_app.columns=['app']
test_app = pd.DataFrame(test_app)
test_app.columns=['app']
print(train_app.shape, test_app.shape)

from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
vec_app1 = TFIDF(ngram_range=(1,1),analyzer='word')
vec_app2 = TFIDF(ngram_range=(2,2),analyzer='word')

train_x_app = vec_app1.fit_transform(train_app['app'])
test_x_app = vec_app1.transform(test_app['app'])

sparse.save_npz('../data/train/x_app1.npz',train_x_app)
sparse.save_npz('../data/test/x_app1.npz',test_x_app)

print('api2 done')

train_x_app = vec_app2.fit_transform(train_app['app'])
test_x_app = vec_app2.transform(test_app['app'])
sparse.save_npz('../data/train/x_app2.npz',train_x_app)
sparse.save_npz('../data/test/x_app2.npz',test_x_app)

print('api2 done')
