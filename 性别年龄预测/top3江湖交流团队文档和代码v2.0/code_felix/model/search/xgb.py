

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from code_felix.tiny.lda import *
from xgboost import XGBClassifier

import numpy as np
import pandas as pd

from  code_felix.tiny.util import *

try:
    from code_felix.tiny.conf import gpu_params
except :
    #gpu_params = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}
    # GPU support
    gpu_params = {}

feature_label = get_stable_feature('0924')

train = feature_label[feature_label['sex'].notnull()]
test = feature_label[feature_label['sex'].isnull()]

X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
Y = train['sex_age']
# Y_CAT = pd.Categorical(Y)
# X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.codes, test_size=0.3, random_state=666)

xgb = XGBClassifier(learning_rate=0.01, n_estimators=600, objective='multi:softprob',
                    silent=True,  **gpu_params)



folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

params = {
        'min_child_weight': [1, 5, 10],
        'reg_alpha':[10, 11, 12],
        'reg_lambda':[10, 11, 12],
        'gamma': [0, 0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree':  [0.6, 0.7, 0.8],
        'max_depth': [3, 4, 5]
        }

random_search = RandomizedSearchCV(xgb, param_distributions=params,
                                   n_iter=param_comb, scoring='neg_log_loss', n_jobs=4,
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

# bst = xgb.train(params, dtrain, 50, watchlist );

#
# pred = bst.predict(xgb.DMatrix(X_test) );
# print(pred[:2])
#
# print(xgb.plot_importance(bst))
# print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
# do the same thing again, but output probabilities

# params['objective'] = 'multi:softprob'
# bst = xgb.train(params, dtrain, 5, watchlist );
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
# yprob = bst.predict( X_test ).reshape( test_Y.shape[0], 6 )
# ylabel = np.argmax(yprob, axis=1)
# print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))