import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split 

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import PReLU
import os

import model_set



features_location='./features/'
layer1_train_output_location = './layer1_train_output/'
layer1_test_output_location = './layer1_test_output/'
stacking_output_location = '../output/'
data='../input/'


def build_model(input_size):
    model = Sequential()
    model.add(Dense(88, input_dim= input_size, init='glorot_uniform', activation='relu'))
    
    model.add(Dense(44, input_dim= input_size, init='glorot_uniform', activation='tanh'))
    model.add(Dense(22, init='glorot_uniform', activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='Adam')

    return model



stack = []
for file in os.listdir(layer1_train_output_location):
    if file.endswith(".npy"):
        temp = np.load(layer1_train_output_location + file)
        #temp = np.loadtxt(layer1_train_output_location + file, delimiter=',')
        stack.append(temp)

test_stack = []
for file in os.listdir(layer1_test_output_location):
        if file.endswith(".npy"):
            temp = np.load(layer1_test_output_location + file)
            test_stack.append(temp)


train_stacked = np.hstack(stack)
test_stacked = np.hstack(test_stack)
nb_input = train_stacked.shape[1]

del stack, test_stack

number_of_folds = 5
number_of_bagging = 1

y = pd.read_csv(features_location+'y_train.csv',header=None)
y = y.iloc[:,0]
#y = y['sex-age'][0:10000]
bag_of_predictions = np.zeros((test_stacked.shape[0], 22))


for j in range(number_of_bagging):
#for j in range(1):

    print '------------- bagging round %d ------------' % j
    skf = StratifiedKFold(n_splits= number_of_folds ,shuffle=True)

    y_dummy = to_categorical(y.tolist())

    train_predict_y = np.zeros((len(y), 22))
    test_predict_y = np.zeros((test_stacked.shape[0], 22))

    test_predict_list = []
    for i, (train_idx, val_idx) in enumerate(skf.split(train_stacked,y)):
    	
        print '------------- fold round %d ------------' % i
        
        #model = build_model(features)
        model=model_set.model_set.nn_3layer_for_stacking(nb_input,22,88,44)

        model.fit(train_stacked[train_idx], y_dummy[train_idx],batch_size=32, epochs=6, verbose=1 ,**{'validation_data': (train_stacked[val_idx], y_dummy[val_idx])})
#		        model.fit(train_stacked[train_idx], y_dummy[train_idx],batch_size=32, epochs=14, verbose=1 ,**{'validation_data': (train_stacked[val_idx], y_dummy[val_idx])})
        #model.fit(train_stacked[train_idx], y_dummy[train_idx],batch_size=32, epochs=1, verbose=1 ,**{'validation_data': (train_stacked[val_idx], y_dummy[val_idx])})

        scoring = model.predict_proba(train_stacked[val_idx])
        train_predict_y[val_idx] = scoring
        l_score = log_loss(y[val_idx], scoring)
        print '    Fold %d loss: %f' % (i, l_score)

        tresult = model.predict_proba(test_stacked)
        test_predict_y = test_predict_y + tresult

    l_score = log_loss(y, train_predict_y)
    print 'Final Fold loss: %f' % (l_score)

    test_predict_y = test_predict_y / number_of_folds
    bag_of_predictions = bag_of_predictions + test_predict_y

bag_of_predictions = bag_of_predictions / number_of_bagging

filename = 'fred_predict_v11.csv'
filename=stacking_output_location + filename
classes_=['1-0','1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9','1-10','2-0','2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9','2-10']
test = pd.read_csv(data+"deviceid_test.tsv", index_col=0,encoding='utf8', sep='\t',header=None)
output = pd.DataFrame(bag_of_predictions, index = test.index[0:bag_of_predictions.shape[0]], columns=classes_)
output.index.name='DeviceID'
output.to_csv(filename,encoding='utf-8',header=True,index=True)

