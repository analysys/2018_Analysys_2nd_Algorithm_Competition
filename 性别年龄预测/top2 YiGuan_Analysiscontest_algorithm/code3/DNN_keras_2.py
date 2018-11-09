
# coding: utf-8

# In[ ]:


from __future__ import print_function
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1671) # for reproducibility
from help_function import LoadData


# ### Load Train/test data

# In[ ]:


train_datapath =  '../Demo/deviceid_train.tsv' 
test_datapath =  '../Demo/deviceid_test.tsv' 
train_data, test_data = LoadData(train_datapath, test_datapath)


# ### load new features

# In[ ]:


applist_train = np.load('new_feature/applist_train.npy')
applist_test = np.load('new_feature/applist_test.npy')
label_train = np.load('new_feature/label_train.npy')
label_test = np.load('new_feature/label_test.npy')
h1_train = np.load('new_feature/h1_train.npy')
h1_test = np.load('new_feature/h1_test.npy')
h2_train = np.load('new_feature/h2_train.npy')
h2_test = np.load('new_feature/h2_test.npy')
h3_train = np.load('new_feature/h3_train.npy')
h3_test = np.load('new_feature/h3_test.npy')
b1_train = np.load('new_feature/brand_train.npy')
b1_test = np.load('new_feature/brand_test.npy')
age_train = np.load('new_feature/age_train.npy')
age_test = np.load('new_feature/age_test.npy')
sex_train = np.load('new_feature/sex_train.npy')
sex_test = np.load('new_feature/sex_test.npy')


# In[ ]:


lgbtrain = np.load('new_feature/lgbcnt_train.npy')
lgbtest = np.load('new_feature/lgbcnt_test.npy')

lgbtfidf_train = np.load('new_feature/lgbtfidf_train.npy')
lgbtfidf_test = np.load('new_feature/lgbtfidf_test.npy')

lgbbagetrain = np.load('new_feature/lgbbage_train.npy')
lgbbagetest = np.load('new_feature/lgbbage_test.npy')
lgbbsextrain = np.load('new_feature/lgbbsex_train.npy')
lgbbsextest = np.load('new_feature/lgbbsex_test.npy')


# In[ ]:


train_set = np.concatenate([applist_train, h1_train,h2_train,b1_train,
                            lgbtrain,lgbtfidf_train,lgbbagetrain,lgbbsextrain,
                            h3_train,age_train,sex_train,
                            label_train],axis=1)
test_set = np.concatenate([applist_test, h1_test,h2_test,b1_test,
                           lgbtest,lgbtfidf_test,lgbbagetest,lgbbsextest ,
                           h3_test,age_test, sex_test,
                           label_test],axis=1)


# In[ ]:


train_set.shape, test_set.shape


# In[ ]:


def NNmodel(N_HIDDEN,Input_shape, DROPOUT, NB_CLASSES):
    model = Sequential()
    model.add(Dense(N_HIDDEN, input_shape=(Input_shape,)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(N_HIDDEN))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(N_HIDDEN))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(N_HIDDEN))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(N_HIDDEN))
    model.add(Activation('relu'))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0003, decay=0),
              metrics=['accuracy'])
    return model


# In[ ]:


def CVtrain(train_data, test_data,labels, num_class, n_folds=5):
    train_predvec = np.zeros((train_data.shape[0], num_class))
    test_predvec = np.zeros((test_data.shape[0], num_class))
    SKF = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 2018)
    train_logloss = []
    valid_logloss = []
    for train_indices, valid_indices in SKF.split(train_data,labels):
        # Training data for the fold
        x_train = train_data[train_indices, :]
        y_train = labels.loc[train_indices, :]
        # Validation data for the fold
        x_valid = train_data[valid_indices, :]
        y_valid = labels.loc[valid_indices, :]
        # NNmodel
        y_train = np_utils.to_categorical(y_train, num_class)
        y_valid = np_utils.to_categorical(y_valid, num_class)
        model = NNmodel(700, 202, 0.25, num_class)
        learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_loss', 
                                            patience = 3,mode='min',
                                            min_delta = 0.001,
                                            verbose = 2, factor=0.3, min_lr = 0.00001) 
        model.fit(x_train, y_train,batch_size=300,epochs=40,verbose=2,
                  validation_data=(x_valid, y_valid),
                  callbacks=[learning_rate_reduction])
        # record logloss
        train_logloss.append(log_loss(y_train, model.predict_proba(x_train)))
        valid_logloss.append(log_loss(y_valid, model.predict_proba(x_valid)))
        train_predvec[valid_indices] = model.predict_proba(x_valid)
        test_predvec += model.predict_proba(test_data)/n_folds
        # Clean up memory
        gc.enable()
        del model, x_train, y_train, x_valid, y_valid
        gc.collect()
        print('############## one flod is over ##############')
    train_logloss.append(np.mean(train_logloss))
    valid_logloss.append(log_loss(labels, train_predvec))
    # dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train_logloss':train_logloss,
                            'valid_logloss':valid_logloss})
    return metrics, train_predvec, test_predvec


# In[ ]:


labels = train_data[['label']]


# In[ ]:


get_ipython().run_line_magic('time', 'metric, train, test = CVtrain(train_set, test_set, labels, 22, 10)')


# In[ ]:


def make_submit(result,test_id):
    result = pd.DataFrame(result,
                          columns=['1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7',
                                   '1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3','2-4', '2-5', '2-6', '2-7', '2-8', 
                                   '2-9', '2-10'])
    result['DeviceID'] = test_id.values
    result = result[['DeviceID']+list(result.columns[:-1])]
    return result


# In[ ]:


result = make_submit(test, test_data.device_id)


# In[ ]:


result.to_csv('submit/sub4.csv',index=False)

