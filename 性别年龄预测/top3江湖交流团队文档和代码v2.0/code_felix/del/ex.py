
from tiny.usage import *
from tiny.util import replace_invalid_filename_char

from code_felix.tiny.tfidf import *


@timed()
def gen_sub_by_para(bal_ratio):
    args = locals()

    drop_useless_pkg = True
    drop_long = 0.3
    n_topics = 5

    lda_feature = get_lda_from_usage(n_topics)

    feature = extend_feature(span_no=24, input=lda_feature,
                             drop_useless_pkg=drop_useless_pkg, drop_long=drop_long)

    feature = convert_label_encode(feature)

    feature_label = attach_device_train_label(feature)

    test = feature_label[feature_label['sex'].isnull()]
    train = feature_label[feature_label['sex'].notnull()]
    train['sex_age'] = train['sex_age'].astype('category')

    X_train, X_test, y_train, y_test = split_train(train, bal_ratio)

    classifier = ExtraTreesClassifier(n_estimators=1000,
                                      max_depth=15,
                                      max_features=128,
                                      verbose=1,
                                      n_jobs=-1,
                                      random_state=42)


    print(f'Train begin#{args}')
    classifier.fit(X_train, y_train)
    print('Train End')


    pre_x=test.drop(['sex','age','sex_age','device'],axis=1)
    sub=pd.DataFrame(classifier.predict_proba(pre_x.values))


    sub.columns=train.sex_age.cat.categories
    sub['DeviceID']=test['device'].values
    sub=sub[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]


    from sklearn.metrics import log_loss

    best = log_loss(y_test, classifier.predict_proba(X_test) )

    best = round(best, 4)

    #lgb.plot_importance(gbm, max_num_features=20)

    print(f'=============Final train feature({len(feature_label.columns)}):\n{list(feature_label.columns)} \n {len(feature_label.columns)}')

    file = f'./sub/baseline_rf_ex_{best}_{args}.csv'
    file = replace_invalid_filename_char(file)
    print(f'sub file save to {file}')
    sub = round(sub,10)
    sub.to_csv(file,index=False)

    print_imp_list(X_train, classifier)

if __name__ == '__main__':

    for bal_ratio in np.arange(0, 1, 0.1):
        bal_ratio=round(bal_ratio, 2)
        gen_sub_by_para(bal_ratio)
    # gen_sub_by_para(True, 0.4)
    # for drop_long in np.arange(0.1, 1.1, 0.1):
    #     for drop_useless_pkg in [True, False]:
    #
    #         gen_sub_by_para(drop_useless_pkg, round(drop_long,1))
