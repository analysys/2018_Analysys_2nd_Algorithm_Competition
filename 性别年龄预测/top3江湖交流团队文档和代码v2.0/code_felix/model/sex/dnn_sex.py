from keras import models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from tiny.usage import *

from code_felix.tiny.tfidf import *


def get_feature_label_dnn(version, ensemble):
    from code_felix.tiny.util import get_stable_feature
    feature_label = get_stable_feature(version)
    feature_label['sex'] = feature_label['sex'].astype('category')
    feature_label['age'] = feature_label['age'].astype('category')
    feature_label['sex_age'] = feature_label['sex_age'].astype('category')


    if ensemble:
        file_list = [
            ('xgb_age', './sub/baseline_1.999298_3194_xgb_age_.h5'),
            ('xgb',     './sub/baseline_2.606958_2666_xgb_1632_.h5'),
            ('lgb',     './sub/baseline_2.61447_294_lgb_.h5'),
            #
            # # #
            # ('lgb', './output/best/baseline_2.62099_287_lgb_min_data_in_leaf1472.h5'),
            # ('dnn', './output/best/baseline_2.613028_2631_xgb_1615_svd_cmp0.h5'),
        ]
        feature_label = ensemble_feature_other_model(feature_label, file_list)

    return feature_label






def train_dnn(dropout, lr, ensemble, label_col):
    #dropout = 0.7
    version = '1011'

    args = locals()
    logger.debug(f'Run train dnn:{args}')

    feature_label = get_feature_label_dnn(version, ensemble)

    test = feature_label[feature_label['sex'].isnull()]
    train= feature_label[feature_label['sex'].notnull()]

    X_train, X_test, y_train, y_test = split_train(train, label_col=label_col )

    input_dim = X_train.shape[1]

    logger.debug(f'X_train:{X_train.shape}, y_train:{y_train.shape}, score:{test.shape}, input_dim:{input_dim}')


    model = Sequential()
    model.add(Dense(1200, input_shape=(input_dim,)))
    #model.add(Activation('sigmoid'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(dropout))


    model.add(Dense(100))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))


    model.add(Dense(15, ))
    model.add(LeakyReLU(alpha=0.01))

    adam = Adam(lr=lr)

    if label_col == 'sex':

        model.add(Dense(2, ))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=adam,
                      # metrics=['categorical_crossentropy'],
                      )
    else:

        model.add(Dense(11, ))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=adam,
                        #metrics=['categorical_crossentropy'],
                      )
    print(model.summary())
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[categorical_accuracy])


    tmp_model = './model/checkpoint/dnn_sex_age_best_tmp.hdf5'
    check_best = ModelCheckpoint(filepath=tmp_model,
                                monitor='val_loss',verbose=1,
                                save_best_only=True, mode='min')

    early_stop = EarlyStopping(monitor='val_loss',verbose=1,
                               patience=100,
                               )
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                               patience=30, verbose=1, mode='min')

    from keras.utils import np_utils
    print(y_train.shape)
    history = model.fit(X_train, np_utils.to_categorical(y_train),
                        validation_data=(X_test, np_utils.to_categorical(y_test)),
                        callbacks=[check_best, early_stop, reduce],
                        batch_size=128,
                        #steps_per_epoch= len(X_test)//128,
                        epochs=50000,
                        verbose=1,
                        )

    best_epoch = np.array(history.history['val_loss']).argmin()+1
    best_score = np.array(history.history['val_loss']).min()



    classifier = models.load_model(tmp_model)

    pre_x = test.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
    logger.debug(f'Test:{test.shape}, pre_x:{pre_x.shape}')

    logger.debug(f'pre_x.values:{pre_x.values.shape}')
    sub = pd.DataFrame(classifier.predict_proba(pre_x.values))

    sub.columns = train.sex_age.cat.categories
    sub['DeviceID'] = test['device'].values
    sub = sub[
        ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '2-0', '2-1', '2-2',
         '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]


    logger.debug(f'best_score(his):{best_score} @ epoch:{best_epoch}')


    # model_file = f'./model/checkpoint/dnn_best_{best_score}_{args}_epoch_{best_epoch}.hdf5'
    # model.save(model_file,
    #            overwrite=True)

    print(
        f'=============Final train feature({len(feature_label.columns)}):\n{list(feature_label.columns)} \n {len(feature_label.columns)}')

    file = f'./sub/baseline_dnn_{best_score}_{args}_epoch_{best_epoch}.csv'
    from code_felix.tiny.util import replace_invalid_filename_char, save_result_for_ensemble

    file = replace_invalid_filename_char(file)
    logger.info(f'sub file save to {file}')
    sub = round(sub, 10)
    sub.to_csv(file, index=False)

    ###Save result for ensemble
    train_bk = pd.DataFrame(classifier.predict_proba( train.drop(['sex', 'age', 'sex_age', 'device'], axis=1) ),
                         index = train.device,
                         columns= train.sex_age.cat.categories
                         )

    test_bk = pd.DataFrame(classifier.predict_proba(pre_x.values),
                         index = test.device,
                         columns= test.sex_age.cat.categories
                         )
    label_bk = pd.DataFrame({'label':train.sex_age.cat.codes},
                         index = train.device,
                         )

    save_result_for_ensemble(f'{round(best_score,5)}_{best_epoch}_v_{version}_dnn_{args}',
                                 train = train_bk,
                                 test  = test_bk ,
                                 label = label_bk,
                             )


if __name__ == '__main__':
    for drop in [0.65] :
        for lr in [ 0.01]:
            for ensemble in [True]:
                for label_col in [ 'age', 'sex',]:
                    train_dnn(drop, lr, ensemble, label_col)
                #exit(0)
