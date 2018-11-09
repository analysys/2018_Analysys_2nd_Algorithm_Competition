#import seaborn as sns
import lightgbm as lgb
from tiny.usage import *

from code_felix.tiny.tfidf import *


def gen_sub_by_para(drop_useless_pkg, drop_long):
    args = locals()

    lda_feature = get_lda_from_usage()
    feature = extend_feature(span_no=24, input=lda_feature, drop_useless_pkg=drop_useless_pkg, drop_long=drop_long)
    feature=  extend_device_brand(feature)
    feature_label = attach_device_train_label(feature)



    train=feature_label[feature_label['sex'].notnull()]
    test =feature_label[feature_label['sex'].isnull()]

    X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
    Y = train['sex']
    Y_CAT = pd.Categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.labels)
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {
        'boosting_type': 'gbdt',
        'max_depth': 3,
        'random_state': 47,
        #"min_data_in_leaf":1000,
        'verbose': -1,
        'colsample_bytree': 0.58,
        # 'min_child_samples': 289,
        #'min_child_weight': 0.1,
        'min_data_in_leaf': 1472,
        #'num_leaves': 300,
        'reg_alpha': 3,
        'reg_lambda': 4,
        'subsample': 0.8
    }

    params_age= {
        'metric': {'multi_logloss'},
        'num_class': 11,
        'objective': 'multiclass',
    }

    params_sex= {
        'metric': ['auc', 'binary_logloss'],
        'objective': 'binary',
    }


    try:

        gbm = lgb.train(dict(params, **params_sex),
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                early_stopping_rounds=50)
    except Exception as error:
        print(f'Model input columns:{list(X.columns)}\n dict({X.dtypes.sort_values()})')
        raise error

    best = round(gbm.best_score.get('valid_0').get('binary_logloss'), 5)

    best = "{:.5f}".format(best)

    pre_x = test.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
    sub = pd.DataFrame(gbm.predict(pre_x.values, num_iteration=gbm.best_iteration))

    sub[1] = 1 - sub[0]
    #sub.head(3)

    sub.columns = Y_CAT.categories
    sub['DeviceID'] = test['device'].values
    sub = sub[['DeviceID','1', '2']]
    sub.head(3)

    #lgb.plot_importance(gbm, max_num_features=20)

    print(f'=============Final train feature({len(feature_label.columns)}):\n{list(feature_label.columns)} \n {len(feature_label.columns)}')

    file = f'./sub/baseline_sex_{best}.csv'
    print(f'sub file save to {file}')
    sub.to_csv(file,index=False)


if __name__ == '__main__':
    gen_sub_by_para( True, round(0.3, 2))