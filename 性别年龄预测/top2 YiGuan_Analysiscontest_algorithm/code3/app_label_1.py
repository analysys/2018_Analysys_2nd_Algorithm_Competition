
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from help_function import LoadData
get_ipython().run_line_magic('matplotlib', 'inline')


# - deviceid_packages.tsv
# - package_label.tsv
# - deviceid_train.tsv
# ----------------------------------

# In[2]:


app_label = pd.read_csv('../Demo/package_label.tsv',sep='\t',names=['app_id','label_1','label_2'])
device_applist = pd.read_csv('../Demo/deviceid_packages.tsv',sep='\t',names=['device_id','app_names'])


# In[3]:


device_applist['app_names']=device_applist['app_names'].apply(lambda x:x.split(','))


# ### 去掉冗余label

# In[4]:


app_label.label_1 = app_label.label_1.apply(lambda x:x.split('(')[0])
app_label.label_2 = app_label.label_2.apply(lambda x:x.split('/')[0])
app_label.label_2 = app_label.label_2.apply(lambda x:x.split('(')[0])


# ### app_id : label

# In[5]:


app_label = app_label.set_index('app_id')
# app_id : label_1
label1_dict = app_label.label_1.to_dict()
# app_id : label_2
label2_dict = app_label.label_2.to_dict()


# ### 设备安装应用列表label数据

# In[6]:


# app_id 转换为 label1, label2
device_applist['label_1'] = device_applist.app_names.apply(lambda x:','.join([label1_dict[key] if key in label1_dict else 'unknow' for key in x])) 

device_applist['label_2'] = device_applist.app_names.apply(lambda x:','.join([label2_dict[key] if key in label2_dict else 'unknow' for key in x]))

# conversion to list
device_applist['label_1'] = device_applist.label_1.apply(lambda x:x.split(','))

device_applist['label_2'] = device_applist.label_2.apply(lambda x:x.split(','))

label1s = device_applist.label_1.apply(lambda x:' '.join(x)).tolist()

label2s = device_applist.label_2.apply(lambda x:' '.join(x)).tolist()


# In[7]:


vectorizer = CountVectorizer()
transformer=TfidfTransformer()


# ---------------------------
# ## label_1

# In[8]:


# label1 count vector
label1_cntvector = vectorizer.fit_transform(label1s)

label1_tfidf = transformer.fit_transform(label1_cntvector)

label1_vocabulary =  pd.DataFrame.from_dict(vectorizer.vocabulary_,orient='index')

label1_names = list(label1_vocabulary.sort_values(by=0).index)

# label1 CountVector
label1_cnt =pd.DataFrame(label1_cntvector.toarray(),columns=label1_names)

label1_cnt['device_id'] = device_applist.device_id.values

# label1  Tfidf
label1_tfidf = pd.DataFrame(label1_tfidf.toarray(),columns=label1_names)

label1_tfidf['device_id'] = device_applist.device_id.values


# --------------
# ### label2 

# In[9]:


label2_vectorizer = CountVectorizer()

label2_cntvector = label2_vectorizer.fit_transform(label2s)
label2_tfidf = transformer.fit_transform(label2_cntvector)

label2_vocabulary =  pd.DataFrame.from_dict(label2_vectorizer.vocabulary_,orient='index')

label2_names = list(label2_vocabulary.sort_values(by=0).index)

label2_cnt = pd.DataFrame(label2_cntvector.toarray(),columns=label2_names)
label2_cnt['device_id'] = device_applist.device_id.values

label2_tfidf = pd.DataFrame(label2_tfidf.toarray(),columns=label2_names)
label2_tfidf['device_id'] = device_applist.device_id.values


# ## Merge label1 label2

# In[10]:


label_cnt = label1_cnt.merge(label2_cnt, on='device_id', how='left')

label_tfidf= label1_tfidf.merge(label2_tfidf, on='device_id',how='left')


# In[11]:


# load trian test data
train_datapath =  '../Demo/deviceid_train.tsv' 
test_datapath =  '../Demo/deviceid_test.tsv' 
train_data, test_data = LoadData(train_datapath, test_datapath)


# ------------------------
# ## Merge data

# In[12]:


train_data = train_data.merge(label_tfidf,on='device_id',how='left')

test_data = test_data.merge(label_tfidf, on='device_id',how='left')


# ------------------------------------
# # Train code

# In[17]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import log_loss
import warnings
import gc
warnings.filterwarnings('ignore')


# In[18]:


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
        # Model
        model = MLPClassifier(hidden_layer_sizes=(200,200,200),  #300,300,300
                     alpha=0.0001,            # 0.00013
                     batch_size=128,
                     learning_rate='adaptive',
                     learning_rate_init=0.00054321,
                     random_state=666,
                     tol = 0.005,
                     verbose=False,
                     early_stopping=True,
                     validation_fraction=0.1)       
        model.fit(x_train, y_train)
        train_predvec[valid_indices] = model.predict_proba(x_valid)
        test_predvec += model.predict_proba(test_data)/n_folds
        # Clean up memory
        gc.enable()
        del model, x_train, y_train, x_valid, y_valid
        gc.collect()
    return train_predvec, test_predvec


# In[19]:


train_set, test_set = train_code(train_data, test_data,'label', 22, 10)


# In[20]:


np.save('new_feature/label_train.npy', train_set)
np.save('new_feature/label_test.npy', test_set)


# In[21]:


label_cnt.to_csv('features/label_cnt.csv',index=False)
label_tfidf.to_csv('features/label_tfidf.csv',index=False)

