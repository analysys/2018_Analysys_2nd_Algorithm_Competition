import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
import xgboost as xgb
import lightgbm as lgb
from keras.callbacks import EarlyStopping
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from numpy import random,mat
from scipy.sparse import issparse
from scipy import sparse
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import model_set
import platform

nb_class = 22
nb_input=38755
nb_output=22
features_location='./features/'
train_features_file=features_location+'feature_train_v1.npz'
test_features_file=features_location+'feature_test_v1.npz'
layer1_train_output_location = './layer1_train_output/'
layer1_test_output_location = './layer1_test_output/'


models=[
("lgb_1",model_set.lgb_model_set.simple_lgb(0.017,850,10,0.5,6,0.2,0.9,0.9,777)),
("nn_2layers_relu",model_set.nn_model_set.nn_2layers_relu(nb_input,nb_output,**{
'hn1' : 150,
'hn2' : 50,
'dp'  : 0.32, 
'epochs' : 13,
'batch_size': 250})),
("nn_2layers_leaky",model_set.nn_model_set.nn_2layers_leaky(nb_input,nb_output,**{
                'hn1': 150,
                'hn2': 50,
                'dp': 0.3,
                'alpha': 0.25,
                'epochs': 13,
                'batch_size': 250,
                })),
("nn_2layers_relu2_1",model_set.nn_model_set.nn_2layers_relu2(nb_input,nb_output,**{
                'hn1' : 180,
                'hn2' : 60,
                'dp'  : 0.23,
                'epochs' : 5,
                'batch_size': 32,
                })),
("nn_2layers_relu2_2",model_set.nn_model_set.nn_2layers_relu2(nb_input,nb_output,**{
                'hn1' : 16,
                'hn2' : 64,
                'dp'  : 0.2,
                'epochs' : 12,
                'batch_size': 32,
                })),
("nn_2layers_relu2_3",model_set.nn_model_set.nn_2layers_relu2(nb_input,nb_output,**{
                'hn1' : 24,
                'hn2' : 36,
                'dp'  : 0.2,
                'epochs' : 13,
                'batch_size': 32,
                })),
("nn_3layers",model_set.nn_model_set.nn_3layers(nb_input,nb_output,**{
                'hn1' : 210,
                'hn2' : 100,
                'hn3' : 50,
                'dp1'  : 0.4,
                'dp2'  : 0.2,
                'epochs': 10, 
                'batch_size': 250
                }))]

Xtrain = []
Xtest = []
y = []



def load_target():
    y = pd.read_csv(features_location+'y_train.csv',header=None)

    y = y.iloc[:,0]
    return y

def load_features(filename):
    tmp = np.load(filename)
    return csr_matrix((tmp['data'], tmp['indices'], tmp['indptr']), shape= tmp['shape'])


def learning(model,Xtrain,y,Xtest, number_of_folds= 10, seed = 42):
    #print( 'Model: ' ,model,model[1].summary())
    
    if model[0].startswith('nn_'):
    		Xtrain=Xtrain.todense()
    		Xtest=Xtest.todense()
    		nn_y= to_categorical(y, 22)
    
    train_predict_y = np.zeros((len(y), nb_class))
    test_predict_y = np.zeros((Xtest.shape[0], nb_class))
    
    ll = 0.
    skf = StratifiedKFold(n_splits = number_of_folds ,shuffle=True, random_state= seed )
    print(Xtrain.shape,y.shape)
    for i, (train_idx, val_idx) in enumerate(skf.split(Xtrain,y)):
    	
        if model[0].startswith('lgb_'):
        		model[1].fit(Xtrain[train_idx], y[train_idx],  eval_set=[(Xtrain[train_idx], y[train_idx]),(Xtrain[val_idx], y[val_idx])], early_stopping_rounds=50, verbose=True)

        if model[0].startswith('nn_'):
        		print(type(Xtrain),type(nn_y))
        		model[1].fit(Xtrain[train_idx], nn_y[train_idx],  validation_data =(Xtrain[train_idx], nn_y[train_idx]))        

        scoring = model[1].predict_proba(Xtrain[val_idx])
        
        train_predict_y[val_idx] = scoring
        l_score = log_loss(y[val_idx], scoring)
        ll += l_score
        
        print( 'Fold ',i,'log_loss: ',l_score)    
            
        test_predict_y=test_predict_y+model[1].predict_proba(Xtest)  
          
    test_predict_y=test_predict_y/number_of_folds    
    
    print( 'average val log_loss:' ,(ll / number_of_folds))
    
    
    return train_predict_y,test_predict_y


if __name__ == "__main__":
	
    nb_class =22
    seed=777
    np.random.seed(seed)
    n_folds=5

    Xtrain = load_features(train_features_file)
    Xtest =  load_features(test_features_file)
    y = load_target()
    
    #Xtrain=Xtrain[0:10000,:]
    #Xtest=Xtest[0:1000,:]
    #y=y[0:10000]

    for model in models:
        train_predict_y,test_predict_y=learning(model,Xtrain,y,Xtest,number_of_folds= n_folds, seed=seed)
        filename = model[0]+ '_' + str(seed) + '_' + str(n_folds) + 'fold'
        np.save(layer1_train_output_location + filename + '_train' , train_predict_y)
        np.save(layer1_test_output_location + filename + '_test', test_predict_y)
        
