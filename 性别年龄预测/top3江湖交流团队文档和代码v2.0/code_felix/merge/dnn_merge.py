import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dropout, Dense
from keras.optimizers import Adam

from code_felix.merge.utils import *
from code_felix.tiny.util import attach_device_train_label, replace_invalid_filename_char, train_test_split
from code_felix.utils_.util_log import *


def get_label_cat():
    label =  attach_device_train_label(None)
    return pd.Categorical(label.sex_age).categories



file_list = [
    #'./output/best/2.621213_2510_xgb.h5' ,


    #'./output/best/baseline_2.606958_2666_xgb_1632_.h5',
    './output/best/baseline_all_xgb_col_830_drop_feature800.h5',
    './output/best/baseline_2.61447_294_lgb_.h5',
    './output/best/baseline_2.606412010192871_783_v_1011_dnn_version1011ensembleTruelr0.01dropout0.65.h5',
    './output/best/baseline_2.6187269142150877_770_v_1011_dnn_version1011ensembleFalselr0.01dropout0.65.h5',

    #Age
    './output/best/baseline_1.999298_3194_xgb_age_.h5',

    #Sex(nan loss issue)
    #'./output/best/baseline_0.653098_2794_xgb_sex_0.95.h5'
    
    #Drop col
    './output/best/baseline_2.607383_2865_xgb_col_1430_drop_feature200.h5',
    './output/best/baseline_2.608196_2860_xgb_col_1230_drop_feature400.h5', #Last col
    
    
    # './output/best/baseline_2.613028_2631_xgb_1615_svd_cmp0.h5' ,
    # './output/best/baseline_2.62099_287_lgb_min_data_in_leaf1472.h5' ,
    # './output/best/baseline_2.6243436072031656_1388_v_1011_dnn_ensembleFalselr0.0001dropout0.75.h5',
    # './output/best/baseline_2.569205629603068_1461_v_1011_dnn_ensembleTruelr0.0001dropout0.75.h5',
    # #'./output/best/baseline_2.634297458902995_1433_v_1002_dnn_lr0.0001dropout0.75.h5' ,

    # #2/3 feature
    # './output/best/baseline_2.618598_2727_xgb_72727_svd_cmp0.h5' ,
    # './output/best/baseline_2.620932_2777_xgb_72727_svd_cmp0.h5' ,

   #  #drop columns
   #  './output/best/baseline_2.620313_2482_xgb_831_drop_feature799.h5',
   #   #'./output/best/baseline_2.617977_2619_xgb_1031_drop_feature599.h5',
   # './output/best/baseline_2.614679_2596_xgb_1231_drop_feature399.h5',



    # #Sex
    # './output/best/0.608252_2577_xgb_sex.h5' ,
    # './output/best/0.625989340877533_357_v_1002_dnn.h5' ,
    #
    # #Age
    # './output/best/baseline_2.004356_3384_xgb_age_svd_cmp50.h5' ,


]

if __name__ == '__main__':
    train_list =[]
    label_list = []
    test_list  = []
    for file in file_list:
        train, label, test = read_result_for_ensemble(file)

        train_list.append(train)
        if label is not None: label_list.append(label)
        test_list.append(test)

    train = pd.concat(train_list, axis=1)
    test = pd.concat(test_list, axis=1)
    label = label_list[0]


    train = train.sort_index()
    label = label.sort_index()

    X_train, X_test, y_train, y_test = train_test_split(train, label.iloc[:,0])

    #drop_list = list(np.arange(0.65, 0.7, 0.03))
   # drop_list.reverse()
    for dense in [128]:
      for drop_out in [0.63]:
        drop_out = round(drop_out, 2)
        patience=50
        lr = 0.0005
        #搭建融合后的模型
        inputs = Input((X_train.shape[1:]))

        x = Dropout(drop_out)(inputs)

        x = Dense(dense, activation='relu')(x)

        x = Dropout(drop_out)(x)

        x = Dense(22, activation='softmax')(x)
        model = Model(inputs, x)


        ########################################

        # np.random.seed(1337)
        #
        # import tensorflow as tf
        # tf.set_random_seed(1234)
        #
        # import random as rn
        # rn.seed(12345)

        early_stop = EarlyStopping(monitor='val_loss', verbose=1,
                                   patience=patience,
                                   )

        model_file ='./model/checkpoint/ensemble.h5'
        check_best = ModelCheckpoint(filepath= model_file,
                                     monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')

        reduce = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=patience//2,verbose=1,mode='min')

        from keras.utils import np_utils
        adam = Adam(lr)
        model.compile(loss='categorical_crossentropy', optimizer=adam,
                      # loss="binary_crossentropy", optimizer="adam",
                      # metrics=["accuracy"]
                      )


        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        print(np_utils.to_categorical(y_train).shape)

        history = model.fit(X_train, np_utils.to_categorical(y_train),
                            validation_data=(X_test, np_utils.to_categorical(y_test)),
                            callbacks=[check_best,
                                       early_stop,
                                       reduce,
                                       ],
                            batch_size=128,
                            # steps_per_epoch= len(X_test)//128,
                            epochs=10000,
                            verbose=1,

                            )

        from keras import models
        model_load = models.load_model(model_file)

        best_epoch = np.array(history.history['val_loss']).argmin() + 1
        best_score = np.array(history.history['val_loss']).min()

        #pre_x = test.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
        sub = pd.DataFrame(model_load.predict(test), columns=get_label_cat())


        sub['DeviceID'] = test.index.values
        sub = sub[
            ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '2-0', '2-1', '2-2',
             '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]

        file = f'./sub/ensemble_{best_score}_epoch_{best_epoch}_drop_{drop_out}_dense_{dense}_patience_{patience}_lr_{lr}.csv'
        file = replace_invalid_filename_char(file)
        logger.debug(f'Input dim is {train.shape}')
        logger.info(f'sub file save to {file}')
        sub = round(sub, 10)
        sub.to_csv(file, index=False)


