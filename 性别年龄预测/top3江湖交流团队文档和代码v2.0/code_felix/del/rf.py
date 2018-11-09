#import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from tiny.usage import *

# from imblearn.ensemble import BalancedRandomForestClassifier
from code_felix.tiny.tfidf import *


@timed()
def gen_sub_by_para(class_weight):
    args = locals()


    feature_label = get_stable_feature('rf01')



    train=feature_label[feature_label['sex'].notnull()]
    test =feature_label[feature_label['sex'].isnull()]

    X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)



    Y = train['sex_age']
    Y_CAT = pd.Categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.labels)

    # X_train.fillna(0, inplace=True)
    # X_test.fillna(0, inplace=True)

    classifier = RandomForestClassifier(n_estimators=6000,
                                        #criterion='entropy',
                                        max_depth = 15,
                                        verbose=1,
                                        n_jobs=-1,
                                        class_weight=class_weight, #"balanced",
                                        random_state=42)

    # classifier = ExtraTreesClassifier(n_estimators=6000,
    #                                   max_depth=15,
    #                                   max_features=128,
    #                                   verbose=1,
    #                                   n_jobs=-1,
    #                                   random_state=42)
    #

    print(f'Train begin#{args}')
    classifier.fit(X_train, y_train)
    print('Train End')


    pre_x=test.drop(['sex','age','sex_age','device'],axis=1)
    sub=pd.DataFrame(classifier.predict_proba(pre_x.values))


    sub.columns=Y_CAT.categories
    sub['DeviceID']=test['device'].values
    sub=sub[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]


    from sklearn.metrics import log_loss

    best = log_loss(y_test, classifier.predict_proba(X_test) )

    best = round(best, 4)

    #lgb.plot_importance(gbm, max_num_features=20)

    print(f'=============Final train feature({len(feature_label.columns)}):\n{list(feature_label.columns)} \n {len(feature_label.columns)}')

    file = f'./sub/baseline_rf_raw_{best}_{args}.csv'
    file = replace_invalid_filename_char(file)
    print(f'sub file save to {file}')
    sub = round(sub,10)
    sub.to_csv(file,index=False)

if __name__ == '__main__':

    # for max_depth in range(4, 40, 1):
        #for max_features in range(1, 10, 1):
    for class_weight in [None, "balanced"]:
        gen_sub_by_para(class_weight)
    # gen_sub_by_para(True, 0.4)
    # for drop_long in np.arange(0.1, 1.1, 0.1):
    #     for drop_useless_pkg in [True, False]:
    #
    #         gen_sub_by_para(drop_useless_pkg, round(drop_long,1))
