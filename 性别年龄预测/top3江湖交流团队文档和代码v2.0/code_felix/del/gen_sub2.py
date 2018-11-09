#import seaborn as sns
import xgboost as xgb
from tiny.lda import *

from  code_felix.tiny.util import *

# New add
# deviceid_train.rename({'device_id':'device'}, axis=1, inplace=True)
deviceid_train = get_lda_from_app_install()
deviceid_train = extend_feature(version=version,span_no=6, input=deviceid_train, trunc_long_time=900)




col_drop = [item for item in deviceid_train.columns if 'max_' in str(item)]
deviceid_train.drop(columns=col_drop, inplace=True )



train=deviceid_train[deviceid_train['sex'].notnull()]
test=deviceid_train[deviceid_train['sex'].isnull()]

X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
Y = train['sex_age']
Y_CAT = pd.Categorical(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.codes)


dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)


# tuning parameters
params = {
    'objective': 'multi:softprob', #'multi:softmax',             # produces 0-1 probabilities for binary classification
    'num_class': 22,
    'booster': 'gbtree',                        # base learner will be decision tree
    'eval_metric': 'mlogloss',                       # stop training based on maximum AUC, AUC always between 0-1
    'eta': 0.08,                                # learning rate
    'subsample': 0.9,                           # use 90% of rows in each decision tree
    'colsample_bytree': 0.9,                    # use 90% of columns in each decision tree
    'max_depth': 7,                            # allow decision trees to grow to depth of 15
    #'base_score': base_y,                       # calibrate predictions to mean of y
    'seed': 47   ,                            # set random seed for reproducibility
    'silent':1
}




# watchlist is used for early stopping
watchlist = [(dtrain, 'train'), (dtest, 'eval')]


bst = xgb.train(params, dtrain, 50, watchlist );


pred = bst.predict(xgb.DMatrix(X_test) );
print(pred[:2])

print(xgb.plot_importance(bst))
# print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
# do the same thing again, but output probabilities

# params['objective'] = 'multi:softprob'
# bst = xgb.train(params, dtrain, 5, watchlist );
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
# yprob = bst.predict( X_test ).reshape( test_Y.shape[0], 6 )
# ylabel = np.argmax(yprob, axis=1)
# print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))