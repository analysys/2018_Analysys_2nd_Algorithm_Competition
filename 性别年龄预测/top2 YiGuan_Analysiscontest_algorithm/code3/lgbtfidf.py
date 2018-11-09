
# coding: utf-8

# In[2]:


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


# In[3]:


applist = pd.read_csv('features/applist_tfidf.csv')
labelcnt = pd.read_csv('features/label_tfidf.csv')
brand = pd.read_csv('features/brand100.csv')
h1 = pd.read_csv('features/h1.csv')
h2 = pd.read_csv('features/h2_tfidf300.csv')
h3 = pd.read_csv('features/h3.csv')


# In[4]:


# trian data , test data
# train test data
train_datapath =  '../Demo/deviceid_train.tsv' 
test_datapath =  '../Demo/deviceid_test.tsv' 
train_data, test_data = LoadData(train_datapath, test_datapath)


# # Merge data

# In[4]:


data = applist.merge(labelcnt, on='device_id', how='left')
data = data.merge(brand, on='device_id', how='left')
data = data.merge(h1, on='device_id', how='left')
data = data.merge(h2, on='device_id', how='left')
data = data.merge(h3, on='device_id', how='left')


# ---------------------------------------------
# # Feature select

# In[5]:


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

# In[7]:


def model(train_data, test_data,label, num_class, n_folds = 10,
         obj='multiclass',metric='multi_logloss'):
    
    #binary ; log_loss
    labels = train_data[[label]]
    train_data = train_data.drop(['device_id','sex','age','label'],axis=1)
    test_data = test_data.drop(['device_id'],axis=1)
    # 10 folds cross validation
    KF = KFold(n_splits = n_folds, shuffle = True, random_state = 2018)
    # test predictions
    nclass = num_class
    if num_class == 1:
        nclass = 2
    test_predictions = np.zeros((test_data.shape[0],nclass))
    # validation predictions
    out_of_fold = np.zeros((train_data.shape[0],nclass))
    # record scores : logloss
    train_logloss = []
    valid_logloss = []
    # Iterate through each fold
    for train_indices, valid_indices in KF.split(train_data):
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
        #print('train loss is : %.5f  |  valid loss is : %.5f'%(train_score,valid_score))
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

def TrainCode(data, label, num_class, nfolds=10,obj='multiclass',
              metric='multi_logloss'):
    train = train_data.merge(data, on='device_id',how='left')
    test = test_data.merge(data, on='device_id', how='left')
    useless_feature = FeatureSelect(train, label, num_class, obj, metric)
    train.drop(useless_feature, axis=1, inplace=True)
    test.drop(useless_feature, axis=1, inplace=True)
    metric, train_prob,test_prob = model(train, test,label,num_class,nfolds,obj,metric)
    return metric, train_prob, test_prob


# -------------------
# 

# In[13]:


metric , lgbtfidf_train , lgbtfidf_test = TrainCode(data, 'label', 22)
np.save('new_feature/lgbtfidf_train.npy',lgbtfidf_train)
np.save('new_feature/lgbtfidf_test.npy',lgbtfidf_test)


# In[ ]:


metric, lgbbsex_train, lgbbsex_test = TrainCode(data, 'sex', 1, 10, 'binary','binary_logloss')
np.save('new_feature/lgbbsex_train.npy',lgbbsex_train)
np.save('new_feature/lgbbsex_test.npy',lgbbsex_test)


# In[ ]:


metric, lgbbage_train, lgbbage_test = TrainCode(data, 'age', 11, 10)
np.save('new_feature/lgbbage_train.npy',lgbbage_train)
np.save('new_feature/lgbbage_test.npy',lgbbage_test)

