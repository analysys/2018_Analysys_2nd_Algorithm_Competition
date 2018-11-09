#import seaborn as sns
from lightgbm import LGBMClassifier
from code_felix.tiny.lda import *
from code_felix.tiny.usage import *

from code_felix.tiny.tfidf import *
from code_felix.tiny.util import split_train


@timed()
def gen_sub_by_para():
    args = locals()
    logger.debug(f'Run train dnn:{args}')

    from code_felix.tiny.util import get_stable_feature
    feature_label = get_stable_feature('1003')
    #feature_label = get_dynamic_feature()
    logger.debug(f'The input feature:{feature_label.shape}')

    test = feature_label[feature_label['sex'].isnull()]
    train=feature_label[feature_label['sex'].notnull()]
    train['sex_age'] = train['sex_age'].astype('category')

    X_train, X_test, y_train, y_test = split_train(train)

    gbm = LGBMClassifier(n_estimators=20000,
                         boosting_type='gbdt',
                         objective='multiclass',
                         num_class=22,
                         random_state=47,
                         metric=['multi_logloss'],
                         verbose=-1,
                         max_depth=3,


                         feature_fraction=0.2,
                         subsample=0.5,
                         min_data_in_leaf=1472,

                         reg_alpha=2,
                         reg_lambda=4,

                         ##########
                         learning_rate=0.05,  # 0.1
                         colsample_bytree=None,  #1
                         min_child_samples=None,  #20
                         min_child_weight=None,  #0.001
                         min_split_gain=None,  #0
                         num_leaves=None,  #31
                         subsample_for_bin=None,  #200000
                         subsample_freq=None,  #1
                         nthread=-1,
                         #device='gpu'

                         )

    # gbm.set_params(**params)

    logger.debug(gbm)

    res=gbm.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=100, verbose=True )
    print(f'Fit return type:{type(res)}')

    print('Feature importances:', list(gbm.feature_importances_))

    print_imp_list(train, gbm)

    best = round(gbm.best_score_.get('valid_1').get('multi_logloss'), 5)
    best_score = best
    best_epoch = gbm.best_iteration_

    print(gbm)


    best = "{:.5f}".format(best)

    pre_x = test.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
    # sub = pd.DataFrame(gbm.predict_proba(pre_x.values, num_iteration=gbm.best_iteration_))
    #
    # sub.columns=train.sex_age.cat.categories
    # sub['DeviceID']=test['device'].values
    # sub=sub[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]
    #
    # # from sklearn.metrics import log_loss
    # # loss = log_loss(y_test, gbm.predict_proba(X_test,num_iteration=gbm.best_iteration_))
    # #
    # # print(f'Loss={loss}, best={best}')
    # #lgb.plot_importance(gbm, max_num_features=20)
    #
    # #print(f'=============Final train feature({len(feature_label.columns)}):\n{list(feature_label.columns)} \n {len(feature_label.columns)}')
    #
    # file = f'./sub/baseline_lg_sci_{best}_{args}.csv'
    # file = replace_invalid_filename_char(file)
    # print(f'sub file save to {file}')
    # sub.to_csv(file,index=False)


    ###Save result for ensemble
    train_bk = pd.DataFrame(gbm.predict_proba(train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)),
                            index=train.device,
                            columns=train.sex_age.cat.categories
                            )

    test_bk = pd.DataFrame(gbm.predict_proba(pre_x),
                           index=test.device,
                           columns=train.sex_age.cat.categories
                           )

    from code_felix.tiny.util import save_result_for_ensemble
    save_result_for_ensemble(f'{best_score}_{best_epoch}_lgb_{args}',
                             train=train_bk,
                             test=test_bk,
                             label=None,
                             )



if __name__ == '__main__':
    # #for learning_rate in np.arange(0.02, 0.02, 0.01):
    # for ratio in np.arange(0, 1, 0.1):
    #     ratio = round(ratio, 2)
        gen_sub_by_para()
    # #for limit in range(100, 1300, 100):
    # for drop in np.arange(0.1, 1.1, 0.1):
    #     gen_sub_by_para(True, round(drop, 2), n_topics=5)
    # gen_sub_by_para(True, 0.4)
    # for drop_long in np.arange(0.1, 1.1, 0.1):
    #     for drop_useless_pkg in [True, False]:
    #
    #         gen_sub_by_para(drop_useless_pkg, round(drop_long,1))
