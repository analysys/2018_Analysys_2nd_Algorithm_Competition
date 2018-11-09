
# coding: utf-8

# In[1]:


import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from help_function import LoadData
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# 机型数据：每个设备的品牌和型号【deviceid_brand.tsv】

# In[2]:


device_brand = pd.read_csv('new_feature/device_brand.csv')


# label =  sex+age, one-hot encode

# In[3]:


# trian data , test data
train_datapath =  '../Demo/deviceid_train.tsv' 
test_datapath =  '../Demo/deviceid_test.tsv' 
train_data, test_data = LoadData(train_datapath, test_datapath)


# In[4]:


train_data.drop(['sex','age'],axis=1,inplace=True)

test_data['label'] = 'unknow'

data = train_data.append(test_data)


# Merge device_brand

# In[5]:


data = data.merge(device_brand,on='device_id',how='left')

data.fillna('unknow',inplace=True)


# In[6]:


brand = data[['device_id','brand']].set_index('device_id')
model = data[['device_id','model']].set_index('device_id')
btype = data[['device_id','btype']].set_index('device_id')

# one-hot encode
# 1 : brand
# 2 : model
# 3 : btype
brand = pd.get_dummies(brand).reset_index()
model = pd.get_dummies(model).reset_index()
btype = pd.get_dummies(btype).reset_index()


# ### brand+model+btype

# In[7]:


data = brand.merge(model, on='device_id', how='left')
data = data.merge(btype, on='device_id', how='left')


# ### 不同尺度的降维

# In[8]:


svd100 = TruncatedSVD(n_components=100, n_iter=15, random_state=666)

brand_100 = pd.DataFrame(svd100.fit_transform(data.iloc[:,1:]))
brand_100['device_id'] = data.device_id.values


# In[9]:


svd550 = TruncatedSVD(n_components=550, n_iter=15, random_state=666)
brand_550 = pd.DataFrame(svd550.fit_transform(data.iloc[:,1:]))
brand_550['device_id'] = data.device_id.values
train = train_data.merge(brand_550, on='device_id', how='left')
test = test_data.merge(brand_550, on='device_id', how='left')


# In[14]:


def train_code(train_data, test_data,label, num_class, n_folds=5):
    labels = train_data[[label]]
    train_data = train_data.drop(['device_id','label'],axis=1)
    test_data = test_data.drop(['device_id','label'],axis=1)
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
                             alpha=0.0003,
                             batch_size=128,
                             tol = 0.0003,
                             learning_rate='adaptive',
                             learning_rate_init=0.00054321,
                             random_state=666, verbose=False, early_stopping=True,
                             validation_fraction=0.1)
        mlpc.fit(x_train, y_train)
        train_predvec[valid_indices] = mlpc.predict_proba(x_valid)
        test_predvec += mlpc.predict_proba(test_data)/n_folds
        # Clean up memory
        gc.enable()
        del mlpc, x_train, y_train, x_valid, y_valid
        gc.collect()
    return train_predvec, test_predvec


# In[15]:


brand_train, brand_test = train_code(train, test, 'label', 22, 10)


# In[16]:


np.save('new_feature/brand_train.npy',brand_train)
np.save('new_feature/brand_test.npy',brand_test)
brand_100.to_csv('features/brand100.csv',index=False)

