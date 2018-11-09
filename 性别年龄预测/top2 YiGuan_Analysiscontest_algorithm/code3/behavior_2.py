
# coding: utf-8

# In[1]:


import gc
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from help_function import LoadData
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# 行为数据
behavior = pd.read_csv('../Demo/deviceid_package_start_close.tsv',sep='\t',
                       names = ['device_id','app_id','start','close'])


# In[3]:


# trian data , test data
# train test data
train_datapath =  '../Demo/deviceid_train.tsv' 
test_datapath =  '../Demo/deviceid_test.tsv' 
train_data, test_data = LoadData(train_datapath, test_datapath)


# ## start , close

# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
vectorizer=CountVectorizer()


# In[5]:


def TransTt2Hour(x):
    timeArray = time.localtime(float(x)/1000)
    otherStyleTime = time.strftime('%H', timeArray)
    return str(otherStyleTime)
def ret_list(arr):
    return list(arr)


# In[6]:


behavior['s_hour'] = behavior.start.apply(TransTt2Hour)
behavior['c_hour'] = behavior.close.apply(TransTt2Hour)


# In[7]:


# group-obj
group_obj = behavior.groupby(by='device_id')
features = pd.DataFrame({'device_id':behavior.device_id.unique()})


# ### start : s_hour

# In[8]:


groupfeature = group_obj.s_hour.agg(ret_list).reset_index()

groupfeature.rename(index=str,columns={0:'s_hour'},inplace=True)

s_hours = groupfeature.s_hour.apply(lambda x:' '.join(x)).tolist()

sh_vector = vectorizer.fit_transform(s_hours)

# cntvector
f_names = ['s'+str(x) for x in range(24)]
sh_vector = pd.DataFrame(sh_vector.toarray(),columns=f_names)

sh_vector['device_id'] = groupfeature.device_id.values


# ### close : c_hour

# In[9]:


groupfeature = group_obj.c_hour.agg(ret_list).reset_index()

groupfeature.rename(index=str,columns={0:'c_hour'},inplace=True)

c_hours = groupfeature.c_hour.apply(lambda x:' '.join(x)).tolist()

ch_vector = vectorizer.fit_transform(c_hours)

f_names = ['c'+str(x) for x in range(24)]
ch_vector = pd.DataFrame(ch_vector.toarray(),columns=f_names)

ch_vector['device_id'] = groupfeature.device_id.values


# In[10]:


# s_hour + c_hour
sc_vector = sh_vector.merge(ch_vector, on='device_id', how='left')
sc_vector.to_csv('features/h3.csv',index=False)


# In[11]:


train_set = train_data.merge(sc_vector, on='device_id', how='left')
test_set = test_data.merge(sc_vector, on='device_id', how='left')


# train code

# In[12]:


def xgbc_code(train_data, test_data,label, num_class, n_folds=5,
              obj='multi:softprob', metric='mlogloss'):
    labels = train_data[[label]]
    train_data = train_data.drop(['device_id','sex','age','label'],axis=1)
    test_data = test_data.drop(['device_id'],axis=1)
    train_predvec = np.zeros((train_data.shape[0], num_class))
    test_predvec = np.zeros((test_data.shape[0], num_class))
    SKF = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 2018)
    for train_indices, valid_indices in SKF.split(train_data,labels):
        # Training data for the fold
        x_train = train_data.loc[train_indices, :]
        y_train = labels.loc[train_indices, :]
        # Validation data for the fold
        x_valid = train_data.loc[valid_indices, :]
        y_valid = labels.loc[valid_indices, :]
        # XGboost
        xgbc = xgb.XGBClassifier(max_depth=3, learning_rate=0.09, n_estimators=1000,
                         silent=True, objective = obj,
                         booster='gbtree', n_jobs=-1,
                         gamma=0, subsample=1,
                         colsample_bytree=0.6, colsample_bylevel=1.,
                         reg_alpha=0, reg_lambda=1,
                         scale_pos_weight=1,
                         base_score=0.5,
                         max_delta_step = 0,
                         random_state=666)
        xgbc.fit(x_train, y_train,
                 eval_set=[(x_train, y_train),(x_valid, y_valid)],
                 eval_metric = metric,
                 early_stopping_rounds=10,
                 verbose=0)
        # record logloss
        train_predvec[valid_indices] = xgbc.predict_proba(x_valid)
        test_predvec += xgbc.predict_proba(test_data)/n_folds
        # Clean up memory
        gc.enable()
        del xgbc, x_train, y_train, x_valid, y_valid
        gc.collect()
    return train_predvec, test_predvec


# In[14]:


# sex+age   num_class = 
train_1, test_1 = xgbc_code(train_set, test_set, 'label', 22, 10)

np.save('new_feature/h3_train.npy',train_1)
np.save('new_feature/h3_test.npy',test_1)


# In[15]:


# age num_class = 11
age_train, age_test = xgbc_code(train_set, test_set, 'age', 11, 10)

np.save('new_feature/age_train.npy',age_train)
np.save('new_feature/age_test.npy',age_test)


# In[16]:


# sex num_class = 2
sex_train, sex_test = xgbc_code(train_set, test_set, 'sex', 2, 10,
                                        'binary:logistic','logloss')

np.save('new_feature/sex_train.npy',sex_train)
np.save('new_feature/sex_test.npy',sex_test)

