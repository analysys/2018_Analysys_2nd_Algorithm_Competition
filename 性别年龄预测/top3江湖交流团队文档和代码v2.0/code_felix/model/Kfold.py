from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

from code_felix.tiny.util import get_stable_feature, reorder_train, reorder_test


def learning(model ,Xtrain ,y ,Xtest, number_of_folds= 5, seed = 777, nb_class =22):
    train_index = Xtrain.index
    test_index = Xtest.index

    Xtrain = Xtrain.reset_index(drop=True)
    Xtest  = Xtest.reset_index(drop=True)

    print( 'Model: %s' % model)

    """ Each model iteration """
    train_predict_y = np.zeros((len(y), nb_class))
    test_predict_y = np.zeros((Xtest.shape[0], nb_class))
    ll = 0.
    """ Important to set seed """
    skf = StratifiedKFold(n_splits = number_of_folds ,shuffle=True, random_state=seed)
    """ Each fold cross validation """

    for i, (train_idx, val_idx) in enumerate(skf.split(Xtrain, y)):
        print('Fold ', i + 1)

        model.fit(Xtrain.values[train_idx], y[train_idx],
                  eval_set=[(Xtrain.values[train_idx], y[train_idx]),
                            (Xtrain.values[val_idx],  y[val_idx])],
                  early_stopping_rounds=50, verbose=True)

        results = model.evals_result()

        logger.debug(results)

        best_epoch = np.array(results['validation_1']['mlogloss']).argmin() + 1
        best_score = np.array(results['validation_1']['mlogloss']).min()

        logger.debug(f"Fold#{i+1} arrive {best_score} at {best_epoch}")

        scoring = model.predict_proba(Xtrain.values[val_idx])
        """ Out of fold prediction """
        train_predict_y[val_idx] = scoring
        l_score = log_loss(y[val_idx], scoring)
        ll += l_score
        print('    Fold %d score: %f' % (i + 1, l_score))

        test_predict_y = test_predict_y + model.predict_proba(Xtest.values)

    test_predict_y = test_predict_y / number_of_folds

    print('average val log_loss: %f' % (ll / number_of_folds))
    """ Fit Whole Data and predict """
    print('training whole data for test prediction...')

    # np.save('./output/xgb_train.np', train_predict_y)
    # np.save('./output/xgb_test.np', test_predict_y)


    ###Save result for ensemble
    train_bk = pd.DataFrame(train_predict_y,
                            index=train_index,
                            columns=get_category().categories
                            )

    test_bk = pd.DataFrame(test_predict_y,
                           index=test_index,
                           columns=get_category().categories
                           )

    label_bk = pd.DataFrame({'label': y},
                            index=train_index,
                            )

    save_result_for_ensemble(f'kfold_xgb',
                             train=train_bk,
                             test=test_bk,
                             label=label_bk,
                             )


if __name__ == '__main__':

    from code_felix.model.xgb import *

    feature_label = get_stable_feature('1011')


    train = feature_label[feature_label['sex'].notnull()]
    train  = reorder_train(train)

    test = feature_label[feature_label['sex'].isnull()]
    test = reorder_test(test)


    Y = train['sex_age']
    Y_CAT = pd.Categorical(Y)

    train = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1, errors='ignore' )
    test = test.drop(['sex', 'age', 'sex_age', 'device'], axis=1, errors='ignore' )


    learning(get_model(200000), train, Y_CAT.codes, test )
