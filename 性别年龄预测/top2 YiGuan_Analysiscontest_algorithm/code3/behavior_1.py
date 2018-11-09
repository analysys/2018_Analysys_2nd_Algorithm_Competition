
# coding: utf-8

# ### 使用应用行为数据和APP数据构造特征进行与训练
# 
# 应用数据：每个设备上各个应用的打开、关闭行为数据【deviceid_package_start_close.tsv】  
# 
# APP数据：每个应用的类别信息【package_label.tsv】
# 
# 训练数据：每个设备对应的性别、年龄段【deviceid_train.tsv】
# 
# 测试数据：提供设备ID供参赛者进行模型验证【deviceid_test.tsv】

# In[1]:


import time
import pandas as pd
from help_function import LoadData


# In[2]:


# 行为数据
behavior = pd.read_csv('../Demo/deviceid_package_start_close.tsv',sep='\t',
                       names = ['device_id','app_id','start','close'])

# 应用label数据
app_label = pd.read_csv('../Demo/package_label.tsv',sep='\t',
                        names=['app_id','label_1','label_2'])
app_label.label_1 = app_label.label_1.apply(lambda x:x.split('(')[0])
app_label.label_2 = app_label.label_2.apply(lambda x:x.split('/')[0])


# In[3]:


def TransTimestamp(x):
    timeArray = time.localtime(float(x)/1000)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime
def TransTt2Date(x):
    timeArray = time.localtime(float(x)/1000)
    otherStyleTime = time.strftime("%m-%d", timeArray)
    return otherStyleTime[:10]
def TransTt2Hour(x):
    timeArray = time.localtime(float(x)/1000)
    otherStyleTime = time.strftime('%H', timeArray)
    return int(otherStyleTime)


# In[4]:


# 转换时间戳为日期和小时
behavior['diff'] = behavior.close - behavior.start
behavior['date'] = behavior.start.apply(TransTt2Date)
behavior['s_hour'] = behavior.start.apply(TransTt2Hour)
behavior['c_hour'] = behavior.close.apply(TransTt2Hour)


# ---------------------------
# 按照 device_id 分组构造特征

# In[5]:


# group-obj
group_obj = behavior.groupby(by='device_id')

features = pd.DataFrame({'device_id':behavior.device_id.unique()})


# In[6]:


# 次数
groupfeature = group_obj.app_id.agg('count').reset_index()
groupfeature.rename(index=str,columns={'app_id':'times'},inplace=True)
features = features.merge(groupfeature,on='device_id',how='left')


# In[7]:


# days
groupfeature = group_obj.date.nunique().reset_index()
groupfeature.rename(index=str,columns={'date':'days'},inplace=True)
features = features.merge(groupfeature,on='device_id',how='left')


# In[8]:


# 使用频繁的app
from collections import Counter

def findMfua(arr):
    arr = list(arr)
    dic = Counter(arr)
    return dic.most_common(1)[0][0]


# In[9]:


# Most frequently used app
groupfeature = group_obj.app_id.agg(findMfua).reset_index()
groupfeature.rename(index=str,columns={'app_id':'mfua'},inplace=True)
features = features.merge(groupfeature, on='device_id',how='left')


# ## Statistical Features

# In[10]:


for col in ['start','close','s_hour','c_hour']:
    for func in ['min','max','mean','median']:
        groupfeature = group_obj[col].agg(func).reset_index()
        groupfeature.rename(index=str,columns={col:col+'_'+func},inplace=True)
        features = features.merge(groupfeature, on='device_id',how='left')


# In[11]:


for func in ['min','max','mean','median','std','sum']:
    groupfeature = group_obj['diff'].agg(func).reset_index()
    groupfeature.rename(index=str,columns={'diff':'diff'+'_'+func},inplace=True)
    features = features.merge(groupfeature, on='device_id',how='left')


# In[12]:


features['hdiff'] = features.c_hour_max - features.s_hour_min
features['m_times'] = features.times / features.days


# ### 把mfua,使用最频繁的APP转换为对应的label
# 
# - mfua : 对应的label1,label2

# In[13]:


app_label.set_index('app_id',inplace=True)
label1_dict = app_label['label_1'].to_dict()
label2_dict = app_label['label_2'].to_dict()


# In[14]:


def replace_label1(app_id):
    if app_id in set(label1_dict.keys()):
        return label1_dict[app_id]
    else :
        return 'unknow'
        
def replace_label2(app_id):
    if app_id in set(label2_dict.keys()):
        return label2_dict[app_id]
    else :
        return 'unknow'


# In[15]:


features['label_1'] = features.mfua.agg(replace_label1)
features['label_2'] = features.mfua.agg(replace_label2)


# ## add device brand features

# In[16]:


device_brand = pd.read_csv('new_feature/device_brand.csv')
features = features.merge(device_brand, on='device_id',how='left')
# encode
features['brand'] = pd.Categorical(features.brand).codes
features['model'] = pd.Categorical(features.model).codes
features['btype'] = pd.Categorical(features.btype).codes

features['mfua'] = pd.Categorical(features.mfua).codes
features['label_1'] = pd.Categorical(features.label_1).codes
features['label_2'] = pd.Categorical(features.label_2).codes

features.to_csv('features/h1.csv',index=False)


# ## Load train_data, test_data

# In[17]:


# train test data
train_datapath =  '../Demo/deviceid_train.tsv' 
test_datapath =  '../Demo/deviceid_test.tsv' 
train_data, test_data = LoadData(train_datapath, test_datapath)


# ### 第一组特征

# In[18]:


h1_train = train_data.merge(features, on='device_id', how='left')
h1_test = test_data.merge(features, on='device_id', how='left')


# ## Xgboost

# In[19]:


import gc
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


def xgbc_code(train_data, test_data,label, num_class, n_folds=5):
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
        xgbc = xgb.XGBClassifier(max_depth=3, learning_rate=0.111, n_estimators=1000,
                         silent=True, objective='multi:softprob',
                         booster='gbtree', n_jobs=-1,
                         gamma=0, subsample=1,
                         colsample_bytree=1, colsample_bylevel=1.,
                         reg_alpha=0, reg_lambda=1,
                         scale_pos_weight=1,
                         base_score=0.5,
                         max_delta_step = 0,
                         random_state=666)
        xgbc.fit(x_train, y_train,
                 eval_set=[(x_train, y_train),(x_valid, y_valid)],
                 eval_metric = 'mlogloss',
                 early_stopping_rounds=10,
                 verbose=False)
        # record logloss
        train_predvec[valid_indices] = xgbc.predict_proba(x_valid)
        test_predvec += xgbc.predict_proba(test_data)/n_folds
        # Clean up memory
        gc.enable()
        del xgbc, x_train, y_train, x_valid, y_valid
        gc.collect()
    return train_predvec, test_predvec


# In[21]:


h1_train, h1_test = xgbc_code(h1_train, h1_test, 'label', 22, 10)

np.save('new_feature/h1_train.npy',h1_train)
np.save('new_feature/h1_test.npy',h1_test)


#  ----------------------------------------------------------------------------------
#  第二组特征
# 
#  ## 设备app使用情况

# In[22]:


## CountVector

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD


# In[23]:


def ret_list(arr):
    return list(arr)


# In[24]:


groupfeature = group_obj.app_id.agg(ret_list).reset_index()

groupfeature.rename(index=str,columns={0:'app_ids'},inplace=True)

app_ids = groupfeature.app_ids.apply(lambda x:' '.join(x)).tolist()

vectorizer=CountVectorizer()
transformer=TfidfTransformer()

# 应用使用情况统计
cnt_vector = vectorizer.fit_transform(app_ids)
# tfidf权重
tfidf = transformer.fit_transform(cnt_vector)


# ## 不同维度的降维

# In[25]:


svd300 = TruncatedSVD(n_components=300, n_iter=15, random_state=666)

h2_cnt = svd300.fit_transform(cnt_vector)
f_names = ['h2_'+str(x) for x in range(300)]
h2_cnt = pd.DataFrame(h2_cnt,columns=f_names)
h2_cnt['device_id'] = groupfeature.device_id.values
# TF-IDF
h2_tfidf = svd300.fit_transform(tfidf)
f_names = ['h2t_'+str(x) for x in range(300)]
h2_tfidf = pd.DataFrame(h2_tfidf,columns=f_names)
h2_tfidf['device_id'] = groupfeature.device_id.values


# In[26]:


# save h2 features

h2_cnt.to_csv('features/h2_cnt300.csv',index=False)
h2_tfidf.to_csv('features/h2_tfidf300.csv',index=False)


# In[27]:


svd = TruncatedSVD(n_components=550, n_iter=15, random_state=666)

svd_cntvec = svd.fit_transform(tfidf)
f_names = ['besvd_'+str(x) for x in range(550)]
svd_cntvec = pd.DataFrame(svd_cntvec,columns=f_names)
# add tfidf_sum columns
svd_cntvec['tfidf_sum'] = tfidf.sum(axis=1)
svd_cntvec['device_id'] = groupfeature.device_id.values


# --------------------

# In[28]:


h2_train = train_data.merge(svd_cntvec, on='device_id',how='left')
h2_test = test_data.merge(svd_cntvec, on='device_id', how='left')


# In[29]:


from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')


# In[30]:


def train_code(train_data, test_data,label, num_class, n_folds=5):
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
        # MLPC
        mlpc = MLPClassifier(hidden_layer_sizes=(640,640,640),
                             alpha=0.0033,
                             batch_size=128,
                             learning_rate='adaptive',
                             learning_rate_init=0.00054321,
                             random_state=666, verbose=0, early_stopping=True,
                             validation_fraction=0.1)
        mlpc.fit(x_train, y_train)
        
        train_predvec[valid_indices] = mlpc.predict_proba(x_valid)
        test_predvec += mlpc.predict_proba(test_data)/n_folds
        # Clean up memory
        gc.enable()
        del mlpc, x_train, y_train, x_valid, y_valid
        gc.collect()
    return train_predvec, test_predvec


# In[31]:


h2_train, h2_test = train_code(h2_train, h2_test, 'label', 22, 10)


# In[32]:


np.save('new_feature/h2_train.npy',h2_train)
np.save('new_feature/h2_test.npy',h2_test)

