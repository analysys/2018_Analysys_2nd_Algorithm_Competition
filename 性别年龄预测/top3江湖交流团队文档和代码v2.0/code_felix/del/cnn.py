# -*- coding: utf-8 -*-
import collections

import nltk
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils

from code_felix.tiny.util import reduce_low_frequency
from code_felix.utils_.util_cache_file import *

MAX_FEATURES = 2000
def convert2num(category):
    import numpy as np
    import pandas as pd
    category = np.array(category)
    a_enc = pd.factorize(category)
    print(f'{len(a_enc[1])}:{sorted(a_enc[1])}')
    return a_enc[0]


@timed()
@file_cache(type='h5')
def get_lstm_feature():
    # MAX_SENTENCE_LENGTH, EMBEDDING_SIZE, HIDDEN_LAYER_SIZE, BATCH_SIZE, NUM_EPOCHS, \
    # global vocab_size
    DATA_DIR = "./cache"
    #file_name = 'get_device_app_sequence_mini.csv'
    file_name = 'get_device_app_sequence_[]_{}.csv'

    global MAX_FEATURES
    # Read training data and generate vocabulary
    maxlen = 0
    word_freqs = collections.Counter()
    num_recs = 0
    ftrain = open(os.path.join(DATA_DIR, file_name), 'r')
    print('Begin to count')
    for line in ftrain:
        label, sentence = line.strip().split(",")
        if label == '' or label == 'sex_age':
            continue
        words = nltk.word_tokenize(sentence.lower())
        words = reduce_low_frequency(words)
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
        if num_recs % 1000 == 0:
            print(f'{"="*20}_{num_recs}')
    ftrain.close()
    print(f'count work, row:{num_recs}')
    ## Get some information about our corpus
    # print maxlen            # 42
    # print len(word_freqs)   # 2313
    # 1 is UNK, 0 is PAD
    # We take MAX_FEATURES-1 featurs to accound for PAD
    vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
    word2index = {x[0]: i + 2 for i, x in
                  enumerate(word_freqs.most_common(MAX_FEATURES))}
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    index2word = {v: k for k, v in word2index.items()}
    # convert sentences to sequences
    X = np.empty((num_recs,), dtype=list)
    y = np.zeros((num_recs,))
    i = 0
    ftrain = open(os.path.join(DATA_DIR, file_name), 'r')
    label_list = []
    for line in ftrain:
        label, sentence = line.strip().split(",")
        if label == '' or label == 'sex_age':
            continue
        words = nltk.word_tokenize(sentence.lower())
        words = reduce_low_frequency(words)
        seqs = []
        for word in words:
            # print(type(word2index))
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs
        label_list.append(label)
        # y[i] = int(label)
        i += 1
    ftrain.close()
    y = convert2num(label_list)
    return pd.DataFrame({'X':X, 'y':y, 'para':vocab_size})


@timed()
def train_lstm():
    args = locals()

    dropout = 0.95

    global MAX_FEATURES
    MAX_SENTENCE_LENGTH = 4000
    EMBEDDING_SIZE = 128
    HIDDEN_LAYER_SIZE = 64
    BATCH_SIZE = 32
    NUM_EPOCHS = 1000
    df = get_lstm_feature()
    X = df.X
    y = df.y
    vocab_size = df['para'].max()
    print(f'Test(raw) feature is ready{X.shape}, {y.shape},vocab_size:{vocab_size}')
    print(y.shape)
    y = np_utils.to_categorical(y, 22)
    # Pad the sequences (left padded with zeros)
    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
    print('Test feature is ready')
    # Split input into training and test
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)
    print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)
    # Build model

    ##TODO
    filters = 25

    kernel_size = 30

    print(kernel_size)
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
    model.add(Dropout(dropout))

    model.add(Conv1D(filters,
                     kernel_size,
                     padding='same',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(MaxPooling1D(2))
    model.add(Dropout(dropout))


    # We add a vanilla hidden layer:
    model.add(Flatten())
    model.add(Dense(250))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))


    model.add(Dense(22))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  # loss="binary_crossentropy", optimizer="adam",
                  # metrics=["accuracy"]
                  )
    # check_best = ModelCheckpoint(filepath=replace_invalid_filename_char(file_path),
    #                              monitor='val_loss', verbose=1,
    #                              save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', verbose=1,
                               patience=20,
                               )
    print('Begin training')
    history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        callbacks=[early_stop],
                        validation_data=(Xtest, ytest))
    print('End training')
    return model, history, args

if __name__ == '__main__':
    #for kernel_size in np.arange(1,3, 1):
        model, history, args = train_lstm()

        best_epoch = np.array(history.history['val_loss']).argmin() + 1
        best_score = np.array(history.history['val_loss']).min()
        print(f'Best val_loss:{best_score},epoch:{best_epoch} with {args}, {dt.now()}')

        # # plot loss and accuracy
        # plt.subplot(211)
        # plt.title("Accuracy")
        # plt.plot(history.history["acc"], color="g", label="Train")
        # plt.plot(history.history["val_acc"], color="b", label="Validation")
        # plt.legend(loc="best")

        # plt.subplot(212)
        # plt.title("Loss")
        # plt.plot(history.history["loss"], color="g", label="Train")
        # plt.plot(history.history["val_loss"], color="b", label="Validation")
        # plt.legend(loc="best")
        #
        # plt.tight_layout()
        # plt.show()

        # # evaluate
        # score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
        # print("Test score: %.3f, accuracy: %.3f" % (score, acc))
        #
        # for i in range(5):
        #     idx = np.random.randint(len(Xtest))
        #     xtest = Xtest[idx].reshape(1,40)
        #     ylabel = ytest[idx]
        #     ypred = model.predict(xtest)[0][0]
        #     sent = " ".join([index2word[x] for x in xtest[0].tolist() if x != 0])
        #     print("%.0f\t%d\t%s" % (ypred, ylabel, sent))


