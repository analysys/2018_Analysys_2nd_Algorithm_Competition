from code_felix.tiny.lda import *
from  code_felix.tiny.util import get_stable_feature, save_result_for_ensemble, train_test_split, print_imp_list
from xgboost import XGBClassifier

from code_felix.tiny.feature_filter import get_cut_feature

try:
    from code_felix.tiny.conf import gpu_params
except :
    # GPU support
    #gpu_params = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}
    gpu_params = {}




def gen_sub_by_para():

    args = locals()

    logger.debug(f'Run train dnn:{args}')
    #feature_label = get_dynamic_feature(None)
    feature_label = get_stable_feature('1011')

    #feature_label = get_cut_feature(feature_label, drop_feature)

    train = feature_label[feature_label['sex'].notnull()]
    test = feature_label[feature_label['sex'].isnull()]

    X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
    Y = train['sex_age']
    Y_CAT = pd.Categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.codes)

    gbm = get_model()
    logger.debug(f"Run the xgb with:{gpu_params}")
    # print(random_search.grid_scores_)
    gbm.fit(X_train, y_train,
                  eval_set=[(X_train , y_train ),  (X_test ,  y_test )],
                  early_stopping_rounds=50,verbose=True )

    results = gbm.evals_result()

    logger.debug(results)

    best_epoch = np.array(results['validation_1']['mlogloss']).argmin() + 1
    best_score = np.array(results['validation_1']['mlogloss']).min()

    logger.debug(f"Xgb arrive {best_score} at {best_epoch}")

    pre_x=test.drop(['sex','age','sex_age','device'],axis=1)

    print_imp_list(X_train, gbm)


    ###Save result for ensemble
    train_bk = pd.DataFrame(gbm.predict_proba(train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)),
                            index=train.device,
                            columns=get_category().categories
                            )

    test_bk = pd.DataFrame(gbm.predict_proba(pre_x),
                           index=test.device,
                           columns=get_category().categories
                           )

    label_bk = pd.DataFrame({'label': Y_CAT.codes},
                            index=train.device,
                            )

    save_result_for_ensemble(f'all_xgb_{args}',
                             train=train_bk,
                             test=test_bk,
                             label=label_bk,
                             )


def get_model(n_estimators=2700):
    gbm = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        # booster='dart',
        num_class=22,
        max_depth=3,
        reg_alpha=10,
        reg_lambda=10,
        subsample=0.7,
        colsample_bytree=0.6,
        n_estimators=n_estimators,

        learning_rate=0.01,

        seed=1,
        missing=None,

        # Useless Paras
        silent=True,
        gamma=0,
        max_delta_step=0,
        min_child_weight=1,
        colsample_bylevel=1,
        scale_pos_weight=1,

        **gpu_params
    )
    return gbm


if __name__ == '__main__':
    # for svd_cmp in range(50, 200, 30):

        gen_sub_by_para()
    #
    # par_list = list(np.round(np.arange(0, 0.01, 0.001), 5))
    # par_list.reverse()
    # print(par_list)
    # for learning_rate in par_list:
    #     #for colsample_bytree in np.arange(0.5, 0.8, 0.1):
    #         gen_sub_by_para(learning_rate)



