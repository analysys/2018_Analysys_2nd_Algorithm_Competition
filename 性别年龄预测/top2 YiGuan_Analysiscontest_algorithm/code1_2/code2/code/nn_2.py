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
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
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
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

path = '../data/'
tr_label=open(path+'train/label.csv')
train_label=[]
for line in tr_label:
	train_label.append(line.replace("\n","").split(","))
del tr_label
train_label = pd.DataFrame(train_label[1:])
train_label.columns=['label']
print(train_label.shape)

train_lda = pd.read_csv(path+"train/lda_fea.csv")
test_lda = pd.read_csv(path+"test/lda_fea.csv")




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

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

texts_train,labels = train.text.values,train_label.label.values
texts_test,test_ids = test.text.values,test.index
y_train = keras.utils.to_categorical(labels, num_classes=22)

del test,train
gc.collect()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(texts_train) + list(texts_test))

sequences_train = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)

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
	lda_input=Input(shape=(50,))
	embedding_layer = Embedding(nb_words,embedding_dim,weights=[embedding_matrix],input_length=max_sequence_length,trainable=False)
	emb_1= SpatialDropout1D(0.2)(embedding_layer(input1))

	conv3 = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu',name="cnn_3")
	conv3_ = Conv1D(filters=200, kernel_size=1, padding='same', activation='relu',name="cnn_3_3")
	conv4 = Conv1D(filters=256, kernel_size=4, padding='same', activation='relu',name="cnn_4")
	conv4_ = Conv1D(filters=200, kernel_size=3, padding='same', activation='relu',name="cnn_4_4")
	conv5 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu',name="cnn_5")
	conv5_ = Conv1D(filters=200, kernel_size=3, padding='same', activation='relu',name="cnn_5_5")
	conv6 = Conv1D(filters=256, kernel_size=6, padding='same', activation='relu',name="cnn_6")
	conv6_ = Conv1D(filters=200, kernel_size=3, padding='same', activation='relu',name="cnn_6_6")

	conv_3 = conv3(emb_1)
	conv_3 = conv3_(conv_3)
	maxpool_3 = GlobalMaxPooling1D()(conv_3)
	attn_3 = AttentionWeightedAverage()(conv_3)
	average_3 = GlobalAveragePooling1D()(conv_3)
	
	conv_4 = conv4(emb_1)
	conv_4 = conv4_(conv_4)
	maxpool_4 = GlobalMaxPooling1D()(conv_4)
	attn_4 = AttentionWeightedAverage()(conv_4)
	average_4 = GlobalAveragePooling1D()(conv_4)
	
	conv_5 = conv5(emb_1)
	conv_5 = conv5_(conv_5)
	maxpool_5 = GlobalMaxPooling1D()(conv_5)
	attn_5 = AttentionWeightedAverage()(conv_5)
	average_5 = GlobalAveragePooling1D()(conv_5)
	
	conv_6 = conv6(emb_1)
	conv_6 = conv6_(conv_6)
	maxpool_6 = GlobalMaxPooling1D()(conv_6)
	attn_6 = AttentionWeightedAverage()(conv_6)
	average_6 = GlobalAveragePooling1D()(conv_6)
	
	concatenated = concatenate([maxpool_3, attn_3, average_3,maxpool_4,attn_4,average_4,maxpool_5, attn_5, average_5,maxpool_6, attn_6, average_6], axis=1)

	lda_1= BatchNormalization()(lda_input)
	lda_1= Dense(20, activation="relu")(lda_1)
	x = Dropout(0.3)(concatenated)
	x = Dense(512, activation="relu",name='dense1')(concatenate([x, lda_1], axis=1))
	x = Dense(128, activation="relu",name='dense2')(Dropout(0.1)(x))
	output_layer = Dense(out_size, activation="softmax")(x)
	model = Model(inputs=[input1,lda_input], outputs=output_layer)
	adam_optimizer = optimizers.Adam(lr=0.001, clipvalue=5, decay=1e-5)
	model = multi_gpu_model(model, 4)
	model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
	model.summary()
	return model
batch_size = 400
epochs = 100
num_folds = 10 
early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='min')

from sklearn.model_selection import StratifiedKFold
num=0
skf = StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=2018)
meta_train = np.zeros((train_1.shape[0],22))
test_preds=[]
for train_index, test_index in skf.split(train_1,labels):
	num+=1
	predict = np.zeros((test_1.shape[0], 6))
	kfold_y_train,kfold_y_test = y_train[train_index], y_train[test_index]
	kfold_X1_train1 = train_1[train_index]
	kfold_X1_valid1 = train_1[test_index]
	kfold_X2_train1 = train_lda.values[train_index]
	kfold_X2_valid1 = train_lda.values[test_index]
	
	
	model = model_cnn(nb_words=len(word_index)+1, embedding_dim=300,embedding_matrix=embedding_matrix, max_sequence_length=maxlen,out_size=22)
	model.fit([kfold_X1_train1,kfold_X2_train1],kfold_y_train, validation_data=([kfold_X1_valid1,kfold_X2_valid1], kfold_y_test),batch_size=batch_size, epochs=epochs, verbose=1,callbacks=[early_stopping])
	model.save_weights('../model/v2__'+str(num)+'base.h5')
	val_pred = model.predict([kfold_X1_valid1,kfold_X2_valid1])
	meta_train[test_index] = pd.DataFrame(val_pred).values
	predict =  np.array(model.predict([test_1,test_lda.values]))
	test_preds.append(predict)
	del model
	gc.collect()
columns_lists=['1-0','1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9','1-10','2-0','2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9','2-10']
meta_train = pd.DataFrame(meta_train)
meta_train.columns = columns_lists
meta_train.to_csv('../stack/meta_nnv2_train.csv', index = None)
	
resCols = ['DeviceID','1-0','1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9','1-10','2-0','2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9','2-10']	
test_df=pd.DataFrame()
test_df['DeviceID'] = [x for x in range(test_1.shape[0])]
preds=(test_preds[0]+test_preds[1]+test_preds[2]+test_preds[3]+test_preds[4]+test_preds[5]+test_preds[6]+test_preds[7]+test_preds[8]+test_preds[9])/10
for col in range(len(columns_lists)):
	test_df[columns_lists[col]]=preds[:,col].tolist()
test_df.to_csv('../stack/meta_nn2_test.csv',index=None)
