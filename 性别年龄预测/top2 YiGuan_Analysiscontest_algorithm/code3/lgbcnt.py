
# coding: utf-8

# In[ ]:


import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
import warnings
from help_function import LoadData
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


applist = pd.read_csv('features/applist_cnt.csv')
labelcnt = pd.read_csv('features/label_cnt.csv')
brand = pd.read_csv('features/brand100.csv')
h1 = pd.read_csv('features/h1.csv')
h2 = pd.read_csv('features/h2_cnt300.csv')


# In[ ]:


# trian data , test data
# train test data
train_datapath =  '../Demo/deviceid_train.tsv' 
test_datapath =  '../Demo/deviceid_test.tsv' 
train_data, test_data = LoadData(train_datapath, test_datapath)


# Merge data

# In[ ]:


train_data = train_data.merge(applist, on='device_id',how='left')
train_data = train_data.merge(labelcnt, on='device_id',how='left')
train_data = train_data.merge(brand, on='device_id',how='left')
train_data = train_data.merge(h1, on='device_id', how='left')
train_data = train_data.merge(h2, on='device_id', how='left')
 

test_data = test_data.merge(applist, on='device_id',how='left')
test_data =  test_data.merge(labelcnt, on='device_id',how='left')
test_data = test_data.merge(brand, on='device_id', how='left')
test_data = test_data.merge(h1, on='device_id', how='left')
test_data = test_data.merge(h2, on='device_id', how='left')


# Feature select

# In[ ]:


def FeatureSelect(train_data, label='label', num_class=22, obj='multiclass',
                  metric='multi_logloss'):
    # binary   , binary_logloss
    model = lgb.LGBMClassifier(boosting_type='gbdt',n_estimators=1000, colsample_bytree=1,
                               objective = obj,max_depth=3,learning_rate = 0.1,
                               num_leaves =31, num_class=num_class,reg_lambda = 1.,
                               reg_alpha = 1, n_jobs = -1,random_state = 8082)
    # split train valid data
    y = train_data[[label]]
    data = train_data.drop(['device_id','sex','age','label'],axis=1)
    x_train, x_valid, y_train, y_valid = train_test_split(data, y, test_size=0.1,random_state=666)
    # fit
    model.fit(x_train, y_train, eval_metric = metric,
              eval_set = [(x_train, y_train),(x_valid, y_valid)],
              eval_names = ['train','valid'],
              early_stopping_rounds = 10, verbose = 0) 
    feature_importance = pd.DataFrame()
    feature_importance['feature'] = x_train.columns.values
    feature_importance['importrance'] = model.feature_importances_
    useless_feature = feature_importance[feature_importance.importrance==0].feature.tolist()
    return useless_feature


# ## CV train

# In[ ]:


def model(train_data, test_data,label, num_class, n_folds = 10,
         obj='multiclass',metric='multi_logloss'):
    
    #binary ; log_loss
    labels = train_data[[label]]
    train_data = train_data.drop(['device_id','sex','age','label'],axis=1)
    test_data = test_data.drop(['device_id'],axis=1)
    # 10 folds cross validation
    SKF = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 2018)
    # test predictions
    test_predictions = np.zeros((test_data.shape[0],num_class))
    # validation predictions
    out_of_fold = np.zeros((train_data.shape[0],num_class))
    # record scores : logloss
    train_logloss = []
    valid_logloss = []
    # Iterate through each fold
    for train_indices, valid_indices in SKF.split(train_data,labels):
        # Training data for the fold
        train_features = train_data.loc[train_indices, :]
        train_labels = labels.loc[train_indices, :]
        # Validation data for the fold
        valid_features = train_data.loc[valid_indices, :]
        valid_labels = labels.loc[valid_indices, :]
        # Create the model
        model = lgb.LGBMClassifier(boosting_type='gbdt',n_estimators=1000, 
                                   objective = obj ,max_depth=3,
                                   learning_rate = 0.1,  num_leaves =31,num_class=num_class,
                                   reg_lambda = 1.,reg_alpha = 1,
                                   subsample = 1., n_jobs = -1, random_state = 8082)

        # Train the model
        model.fit(train_features, train_labels, eval_metric = metric,
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], #categorical_feature =['brand','type','btype'],
                  early_stopping_rounds = 10, verbose = 0)
        # Record the best iteration
        best_iteration = model.best_iteration_
        # test result
        test_predictions+= model.predict_proba(test_data, num_iteration = best_iteration)/n_folds
        # valid result
        pred_valid = model.predict_proba(valid_features, num_iteration = best_iteration)
        # Record the best multi logloss
        valid_score = model.best_score_['valid'][metric]
        train_score = model.best_score_['train'][metric]
        valid_logloss.append(valid_score)
        train_logloss.append(train_score)
        # validation set result
        out_of_fold[valid_indices] = pred_valid
        print('train loss is : %.5f  |  valid loss is : %.5f'%(train_score,valid_score))
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
    # overall valida
    valid_logloss.append(np.mean(valid_logloss))
    train_logloss.append(np.mean(train_logloss))
    # dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train_logloss':train_logloss,
                            'valid_logloss':valid_logloss})
    return metrics,out_of_fold,test_predictions


# In[ ]:


useless_feature = FeatureSelect(train_data)


# In[ ]:


train = train_data.drop(useless_feature, axis=1)
test = test_data.drop(useless_feature, axis=1)


# In[ ]:


get_ipython().run_line_magic('time', "metric, train_proba, test_proba = model(train, test, 'label', 22, 10)")


# In[ ]:


np.save('new_feature/lgbcnt_train.npy',train_proba)
np.save('new_feature/lgbcnt_test.npy',test_proba)

