import pandas as pd
import numpy as np
import gc
from gensim.models.word2vec import Word2Vec,Text8Corpus
import logging
import os
from keras import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,log_loss
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D
from keras.layers import *
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import Callback
from keras import backend as K
import keras
import fasttext
from sklearn.model_selection import KFold
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation,  GlobalAveragePooling1D, LSTM, GRU
from keras.layers import merge, Bidirectional, add, Conv1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K
import tensorflow as tf
from gensim.models.word2vec import Word2Vec,Text8Corpus
from keras.utils import multi_gpu_model
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"]="3"

path = '../data/'
tr_label=open(path+'train/label.csv')
train_label=[]
for line in tr_label:
	train_label.append(line.replace("\n","").split(","))
del tr_label
train_label = pd.DataFrame(train_label[1:])
train_label.columns=['label']
print(train_label.shape)

f=open(path+'train/app_list.csv')
t=open(path+'test/app_list.csv')
train=[]
test=[]
for line in f:
	train.append(line.replace("\n","").split(","))
for line in t:
	test.append(line.replace("\n","").split(","))

del f,t
gc.collect()

train = pd.DataFrame(train)
train.columns=['text']
test = pd.DataFrame(test)
test.columns=['text']
print(train.shape, test.shape)

f=open(path+'train/install_apps.csv')
t=open(path+'test/install_apps.csv')
train_=[]
test_=[]
for line in f:
	train_.append(line.replace("\n","").split(","))
for line in t:
	test_.append(line.replace("\n","").split(","))

del f,t
gc.collect()

train_install = pd.DataFrame(train_[1:])
train_install.columns=['install_text']
test_install = pd.DataFrame(test_[1:])
test_install.columns=['install_text']
print(train_install.shape, test_install.shape)


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

texts_train,labels = train.text.values,train_label.label.values
install_train = train_install.install_text.values
install_test = test_install.install_text.values
texts_test,test_ids = test.text.values,test.index
y_train = keras.utils.to_categorical(labels, num_classes=22)


del test,train
gc.collect()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(texts_train) + list(texts_test))

sequences_train = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)
sequences_install_train = tokenizer.texts_to_sequences(install_train)
sequences_install_test = tokenizer.texts_to_sequences(install_train)
word_index = tokenizer.word_index
maxlen=3000

train_1 = pad_sequences(sequences_train,maxlen=maxlen,padding='pre', truncating='pre')
test_1 = pad_sequences(sequences_test,maxlen=maxlen,padding='pre', truncating='pre')

print('Shape of data tensor:', train_1.shape)
print('Shape of data tensor:', labels.shape)
print('Shape of label tensor:', test_1.shape)
del texts_train,texts_test,sequences_train,sequences_test
gc.collect()

fast_model = fasttext.load_model('../w2v/fast_300_model.bin')
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
	embedding_matrix[i]=fast_model[word]

class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


def model_cnn(nb_words, embedding_dim,embedding_matrix, max_sequence_length,out_size):
	input1 = Input(shape=(max_sequence_length,))
	recurrent_units=64
	embedding_layer = Embedding(nb_words,embedding_dim,weights=[embedding_matrix],input_length=max_sequence_length,trainable=False)
	emb_1= SpatialDropout1D(0.3)(embedding_layer(input1))
	
	rnn_1 = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(emb_1)
	rnn_2 = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(rnn_1)
	x = concatenate([rnn_1, rnn_2], axis=2)

	last = Lambda(lambda t: t[:, -1], name='last')(x)
	maxpool = GlobalMaxPooling1D()(x)
	attn = AttentionWeightedAverage()(x)
	average = GlobalAveragePooling1D()(x)

	all_views = concatenate([last, maxpool, average, attn], axis=1)
	x = Dropout(0.3)(all_views)
	x = BatchNormalization()(x)
	x = Dense(512, activation="relu",name='dense1')(x)
	x = BatchNormalization()(x)
	x = Dense(128, activation="relu",name='dense2')(Dropout(0.1)(x))
	output_layer = Dense(out_size, activation="softmax")(x)
	model = Model(inputs=input1, outputs=output_layer)
	adam_optimizer = optimizers.Adam(lr=0.001, clipvalue=5, decay=1e-5)
	model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
	model.summary()
	return model

batch_size = 200
epochs = 100
num_folds = 10 
early_stopping = EarlyStopping(monitor='val_loss', patience=6, mode='min')

from sklearn.model_selection import StratifiedKFold
num=0
skf = StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=2018)
meta_train = np.zeros((train_1.shape[0],22))
test_preds=[]
for train_index, test_index in skf.split(train_1,labels):
	num+=1
	predict = np.zeros((test_1.shape[0], 6))
	kfold_y_train,kfold_y_test = y_train[train_index], y_train[test_index]
	kfold_X_train1 = train_1[train_index]
	kfold_X_valid1 = train_1[test_index]
	
	model = model_cnn(nb_words=len(word_index)+1, embedding_dim=300,embedding_matrix=embedding_matrix, max_sequence_length=maxlen,out_size=22)
	model.fit(kfold_X_train1,kfold_y_train, validation_data=(kfold_X_valid1, kfold_y_test),batch_size=batch_size, epochs=epochs, verbose=1,callbacks=[early_stopping])
	model.save('../model_v5'+str(num)+'base.h5')
	val_pred = model.predict(kfold_X_valid1)
	meta_train[test_index] = pd.DataFrame(val_pred).values
	predict =  np.array(model.predict(test_1))
	test_preds.append(predict)
	del model
	gc.collect()
columns_lists=['1-0','1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9','1-10','2-0','2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9','2-10']
meta_train = pd.DataFrame(meta_train)
meta_train.columns = columns_lists
meta_train.to_csv('../stack/meta_nnv5_train.csv', index = None)
	
resCols = ['DeviceID','1-0','1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9','1-10','2-0','2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9','2-10']	
test_df=pd.DataFrame()
test_df['DeviceID'] = pd.read_csv('../data/Demo/deviceid_test.tsv',sep='\t',names=['device_id'])['device_id']
preds=(test_preds[0]+test_preds[1]+test_preds[2]+test_preds[3]+test_preds[4]+test_preds[5]+test_preds[6]+test_preds[7]+test_preds[8]+test_preds[9])/10
for col in range(len(columns_lists)):
	test_df[columns_lists[col]]=preds[:,col].tolist()
test_df.to_csv('../stack/meta_nn5_test.csv',index=None)
