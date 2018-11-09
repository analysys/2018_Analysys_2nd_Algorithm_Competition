
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss
from help_function import LoadData
get_ipython().run_line_magic('matplotlib', 'inline')


#  设备数据：每个设备上的应用安装列表，设备应用名都进行了hash处理【deviceid_packages.tsv】

# In[2]:


device_applist = pd.read_csv('../Demo/deviceid_packages.tsv',sep='\t',
                             names=['device_id','app_names'])

device_applist['app_names']=device_applist['app_names'].apply(lambda x:x.split(','))
device_applist['app_count']=device_applist['app_names'].apply(lambda x:len(x))


# In[3]:


vectorizer=CountVectorizer()
transformer=TfidfTransformer()
# 所有设备的应用安装列表
apps = device_applist['app_names'].apply(lambda x:' '.join(x)).tolist()
# 设备安装应用稀疏矩阵
cntTf = vectorizer.fit_transform(apps)
# tfidf权重
tfidf=transformer.fit_transform(cntTf)
# TruncateSVD
svd = TruncatedSVD(n_components=550, n_iter=15, random_state=666)
# countvector
app_svd = svd.fit_transform(cntTf)
f_names = ['svd-'+str(x) for x in range(550)]
app_svd = pd.DataFrame(app_svd,columns=f_names)
# add tfidf_sum columns
app_svd['tfidf_sum'] = tfidf.sum(axis=1)
app_svd['device_id'] = device_applist.device_id.values


# In[ ]:


app_tfidf = svd.fit_transform(tfidf)
f_names = ['tfidf'+str(x) for x in range(550)]
app_tfidf = pd.DataFrame(app_tfidf, columns=f_names)

app_tfidf['device_id'] = device_applist.device_id.values

app_svd.to_csv('features/applist_cnt.csv',index=False)
app_tfidf.to_csv('features/applist_tfidf.csv',index=False)


# In[5]:


device_applist =  device_applist.merge(app_svd, on='device_id', how='left')


# ---------------

# ## Train/test

# In[6]:


train_path = '../Demo/deviceid_train.tsv'
test_path = '../Demo/deviceid_test.tsv'
train_data, test_data = LoadData(train_path, test_path)


# ### Merge(applist)

# In[7]:


train_data = train_data.merge(device_applist, on='device_id',how='left')
test_data = test_data.merge(device_applist, on='device_id',how='left')


# # MLPC

# In[8]:


from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import gc
import warnings
warnings.filterwarnings('ignore')


# In[9]:


def train_code(train_data, test_data,label, num_class, n_folds=5):
    labels = train_data[[label]]
    train_data = train_data.drop(['device_id','sex','age','label','app_names'],axis=1)
    test_data = test_data.drop(['device_id','app_names'],axis=1)
    train_predvec = np.zeros((train_data.shape[0], num_class))
    test_predvec = np.zeros((test_data.shape[0], num_class))
    SKF = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 2018)
     
    for train_indices, valid_indices in SKF.split(train_data, labels):
        # Training data for the fold
        x_train = train_data.loc[train_indices, :]
        y_train = labels.loc[train_indices, :]
        # Validation data for the fold
        x_valid = train_data.loc[valid_indices, :]
        y_valid = labels.loc[valid_indices, :]
        # MLPC
        mlpc = MLPClassifier(hidden_layer_sizes=(640,640,640),  #300,300,300
                     alpha=0.0033,            
                     batch_size=256,          # 256
                     learning_rate='adaptive',
                     learning_rate_init=0.00054321,
                     random_state=666,
                     verbose=False,
                     early_stopping=True,
                     validation_fraction=0.1)       
        mlpc.fit(x_train, y_train)
         
        train_predvec[valid_indices] = mlpc.predict_proba(x_valid)
        test_predvec += mlpc.predict_proba(test_data)/n_folds
        # Clean up memory
        gc.enable()
        del mlpc, x_train, y_train, x_valid, y_valid
        gc.collect()
    return train_predvec, test_predvec


# In[10]:


train_set, test_set = train_code(train_data, test_data,'label', 22, 10)


# In[12]:


np.save('new_feature/applist_train.npy',train_set)
np.save('new_feature/applist_test.npy',test_set)

