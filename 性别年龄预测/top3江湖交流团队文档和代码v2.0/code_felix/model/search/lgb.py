from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from code_felix.tiny.lda import *

from  code_felix.tiny.util import *

deviceid_train = get_stable_feature('0924')

train=deviceid_train[deviceid_train['sex'].notnull()]
test=deviceid_train[deviceid_train['sex'].isnull()]

X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
Y = train['sex_age']


gbm = LGBMClassifier(n_estimators=4000,
                     boosting_type='gbdt',
                     objective='multiclass',
                     max_depth=-1,
                     num_class=22,
                     random_state=47,
                     metric='multi_logloss',
                     verbose=-1,
                     #n_jobs=4,
                    )

#gbm.set_params(**params)

print(gbm)

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

from scipy.stats import randint as sp_randint

params  = {'num_leaves': sp_randint(6, 50),
             "min_data_in_leaf":sp_randint(500, 1500),
             'min_child_samples': sp_randint(100, 500),
             #'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': [0.5, 0.6, 0.7],
             'colsample_bytree': [0,2, 0.3, 0.5,0.6,0.7],
             'reg_alpha': [2,3, 4],
             'reg_lambda': [4,5,6]}

random_search = RandomizedSearchCV(gbm, param_distributions=params,
                                   n_iter=param_comb, scoring='neg_log_loss', #n_jobs=4,
                                   cv=skf.split(X,Y), verbose=3, random_state=1001 )

# print(random_search.grid_scores_)
random_search.fit(X, Y)
# print(random_search.grid_scores_)

print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
