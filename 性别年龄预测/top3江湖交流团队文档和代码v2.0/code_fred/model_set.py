from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, ThresholdedReLU, LeakyReLU
from keras.optimizers import Adadelta, Adam, rmsprop, SGD
from keras.utils import np_utils
import numpy as np
import pandas as pd
import lightgbm as lgb


class lgb_model_set():
    @staticmethod
    def simple_lgb(learning_rate,num_leaves,min_data_in_leaf,bagging_fraction,bagging_freq,colsample_bytree,reg_alpha,reg_lambda,random_state=777, **kwargs):

		    params = {
		                'boosting_type':'gbdt',
		                'n_estimators':4000,
		                'nthread':8,
		                'learning_rate': learning_rate,
		                'max_depth': -1,
		                'metric': 'multi_logloss',
		                'objective': 'multiclass',
		                'num_class': 22,
		                'random_state':random_state,
		                'silent' : True,
		                'num_leaves':num_leaves,
		                'min_data_in_leaf':min_data_in_leaf,
		                'is_unbalance' : [True],
		                'bagging_fraction' : bagging_fraction,
		                'bagging_freq' : bagging_freq,
		                'colsample_bytree': colsample_bytree,
		                'reg_alpha' :reg_alpha,
		                'reg_lambda' :reg_lambda                	
		    }
		
		    model=lgb.LGBMClassifier(boosting_type=params['boosting_type'],learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],max_depth=params['max_depth'],
		                                     num_leaves=params['num_leaves'],metric='multi_logloss',num_class=params['num_class'],objective='multiclass',
		                                     random_state=params['random_state'],
		                                     min_data_in_leaf=params['min_data_in_leaf'],
		                                     bagging_fraction=params['bagging_fraction'],
		                                     bagging_freq=params['bagging_freq'],
		                                     colsample_bytree=params['colsample_bytree'],
		                                    reg_alpha=params['reg_alpha'],reg_lambda=params['reg_lambda'],early_stopping_rounds=50,
		                                     nthread=params['nthread']
		                                    )
		    return model

class nn_model_set():

    @staticmethod
    def nn_2layers_relu(nb_input, nb_output, hn1 = 16, hn2 = 64, dp=0.2, **kwargs):
        model = Sequential()
        model.add(Dense(hn1, input_dim= nb_input, init='glorot_uniform', activation='relu'))
        model.add(Dropout(dp))
        model.add(Dense(hn2, init='glorot_uniform', activation='relu'))
        model.add(Dropout(dp))
        model.add(Dense(nb_output, init='glorot_uniform', activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adadelta')  #logloss
        return model

    @staticmethod
    def nn_2layers_prelu(nb_input, nb_output, hn1 = 16, hn2 = 64, dp = 0.2, **kwargs):
        # create model
        model = Sequential()
        model.add(Dense(hn1, input_dim= nb_input, init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(dp))
        model.add(Dense(hn2, init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(dp))
        model.add(Dense(nb_output, init='glorot_uniform', activation='softmax'))
        # Compile model
        sgd = SGD(lr=0.03, decay =0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)  #logloss
        return model
    @staticmethod
    def nn_2layers_leaky(nb_input, nb_output, hn1 = 16, hn2 = 64, dp = 0.2, alpha = 0.3, **kwargs):
        # create model
        model = Sequential()
        model.add(Dense(hn1, input_dim= nb_input, init='normal'))
        model.add(LeakyReLU(alpha = alpha))
        model.add(Dropout(dp))
        model.add(Dense(hn2, init='normal'))
        model.add(LeakyReLU(alpha = alpha))
        model.add(Dense(nb_output, init='normal', activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
        return model
        
    @staticmethod        
    def nn_2layers_relu2(nb_input, nb_output, hn1 = 16, hn2 = 64, dp = 0.2, **kwargs):
        """ Provide customable on number of hidden units in layer 1/2
            As well as drop out size in connection between layer 1 and 2
        """
        # create model
        model = Sequential()
        model.add(Dense(hn1, input_dim= nb_input, init='glorot_uniform', activation='relu'))
        model.add(Dropout(dp))
        model.add(Dense(hn2, init='glorot_uniform', activation='tanh'))
        model.add(Dropout(0.1))
        model.add(Dense(nb_output, init='glorot_uniform', activation='softmax'))
        # Compile model
        sgd = SGD(lr=0.03, decay =0.0001)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta')  #logloss
        return model

    @staticmethod
    def nn_3layers(nb_input, nb_output,hn1=210,hn2=100,hn3=50,dp1=0.4,dp2=0.2, **kwargs):
        """ Provide customable on number of hidden units in layer 1/2
            As well as drop out size in connection between layer 1 and 2
        """
        # create model
        model = Sequential()
        model.add(Dense(hn1, input_dim= nb_input, init='glorot_uniform', activation='relu'))
        model.add(Dropout(dp1))
        model.add(Dense(hn2, init='normal', activation='tanh'))
        model.add(Dropout(dp2))
        model.add(Dense(hn3, init='glorot_uniform', activation='linear'))
        model.add(Dense(nb_output, init='normal', activation='softmax'))
        # Compile model
        sgd = SGD(lr=0.08, decay =0.0001)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta')  #logloss
        return model
        
    @staticmethod        
    def nn_3layer_for_stacking(nb_input, nb_output, hn1=100,hn2=50):
        model = Sequential()
        model.add(Dense(hn1, input_dim= nb_input, init='glorot_uniform', activation='relu'))
        model.add(Dense(h2, input_dim= input_size, init='glorot_uniform', activation='tanh'))
        model.add(Dense(nb_output, init='glorot_uniform', activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Adam')
        return model
