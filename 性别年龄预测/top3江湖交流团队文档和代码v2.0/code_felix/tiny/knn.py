


from code_felix.tiny.lda import *

from  code_felix.tiny.util import *
from code_felix.tiny.word2vec import *
from code_felix.utils_.util_cache_file import *
from functools import lru_cache
import numpy as np

@lru_cache()
def get_data():

    model = get_dict(False)
    df = pd.DataFrame(index=list(model.wv.vocab.keys()), data=model[model.wv.vocab])
    df.index.name = 'package'

    df = df.reset_index()

    df['check'] = df.package.apply(lambda val: len(val))

    df = extend_pkg_label(df)
    df.sort_values('check', ascending=False)
    df['tmp__'] = df.p_type
    df.p_type = df.p_type.astype('category')
    df.p_sub_type = df.p_sub_type.astype('category')
    df.combine_type = df.combine_type.astype('category')

    #df = convert_label_encode(df, ['tmp__', 'package'])

    train = df[ (df.tmp__ != 'Unknown') & (df.tmp__ != np.nan)]
    test =  df[ (df.tmp__ == 'Unknown') | (df.tmp__ == np.nan)]

    logger.debug(f'{train.shape}, {test.shape}')
    return train, test

def extend_pkg_label_knn(col, feature):
    app_type = get_app_type_with_knn(col)
    return pd.merge(feature, app_type, how='left')

from functools import lru_cache, lru_cache


@lru_cache()
@timed()
@file_cache(overwrite=True)
def get_app_type_with_knn(col):
    train, test = get_data()
    X = train.iloc[:, 1:21]
    y = train[col]

    # index=list(model.wv.vocab.keys()), data=model[model.wv.vocab]
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y.cat.codes)

    #test =
    test_y = neigh.predict(test.iloc[:, 1:21].values)

    df_train = pd.DataFrame(data ={ 'package':train.package , f'{col}_knn': train[col]} )

    df_test =  pd.DataFrame(data ={ 'package':test.package , f'{col}_knn': test_y, } )

    app_type = pd.concat([df_test, df_train])

    app_type[f'{col}_knn_raw'] = app_type[f'{col}_knn']

    app_type[f'{col}_knn'].replace(range(0, len(y.cat.categories)), y.cat.categories, inplace=True)

    return app_type



if __name__ == '__main__':

    #X = [[0], [1], [2], [3]]
    #y = [0, 0, 1, 1]

    print(get_app_type_with_knn('p_type'))
    # 'p_type', 'p_sub_type','combine_type'
    get_app_type_with_knn('p_sub_type')
    get_app_type_with_knn('combine_type')

    #print(neigh.predict_proba(test))


