from sklearn.decomposition import LatentDirichletAllocation
from code_felix.tiny.package import *
from code_felix.tiny.util import *

from code_felix.utils_.util_cache_file import *


#
# @file_cache(overwrite=False)
# @timed()
# def get_lda_from_app_install(drop=False):
#
#     path = './input/'
#
#     deviceid_packages = pd.read_csv(path + 'deviceid_packages.tsv', sep='\t', names=['device', 'apps'])
#
#     deviceid_packages['apps'] = deviceid_packages['apps'].apply(lambda x: x.split(','))
#     deviceid_packages['app_lenghth'] = deviceid_packages['apps'].apply(lambda x: len(x))
#
#     apps = deviceid_packages['apps'].apply(lambda x: ' '.join(x)).tolist()
#     vectorizer = CountVectorizer()
#     cntTf = vectorizer.fit_transform(apps)
#     if drop:
#         cntTf = pd.DataFrame(data=cntTf.toarray(), index=deviceid_packages.device,
#                                  columns=vectorizer.get_feature_names())
#         cntTf = drop_useless_package(cntTf)
#         import scipy
#         cntTf = scipy.sparse.csr_matrix(cntTf.values)
#     lda = LatentDirichletAllocation(n_topics=5,
#                                     learning_offset=50.,
#                                     random_state=666)
#     docres = lda.fit_transform(cntTf)
#     deviceid_packages = pd.concat([deviceid_packages, pd.DataFrame(docres)], axis=1)
#     deviceid_packages = deviceid_packages.drop('apps', axis=1)
#
#
#     # transformer = TfidfTransformer()
#     # tfidf = transformer.fit_transform(cntTf)
#     # #word = vectorizer.get_feature_names()
#     # weight = tfidf.toarray()
#     # df_weight = pd.DataFrame(weight)
#     # feature = df_weight.columns
#     # df_weight['sum'] = 0
#     # for f in tqdm(feature):
#     #     df_weight['sum'] += df_weight[f]
#     # deviceid_packages['tfidf_sum'] = df_weight['sum']
#
#
#     return deviceid_packages
#


@timed()
def get_lda_from_usage(n_topics):
    drop = 18363


    df_list = [


               get_lda_app_and_usage(group_level='app',   drop=0, agg_col='package', agg_method=None, n_topics=n_topics) ,
               get_lda_app_and_usage(group_level='usage', drop=0, agg_col='package', agg_method='count', n_topics=n_topics) ,
               get_lda_app_and_usage(group_level='usage', drop=0, agg_col='package', agg_method='sum', n_topics=n_topics) ,

               get_lda_app_and_usage(group_level='app',   drop=drop, agg_col='package', agg_method=None, n_topics=n_topics) ,
               get_lda_app_and_usage(group_level='usage', drop=drop, agg_col='package', agg_method='count', n_topics=n_topics) ,
               get_lda_app_and_usage(group_level='usage', drop=drop, agg_col='package', agg_method='sum', n_topics=n_topics) ,

               # get_lda_app_and_usage(group_level='app', drop=True, agg_col=None, agg_method=None),
               # get_lda_app_and_usage(group_level='app', drop=False, agg_col=None, agg_method=None),
               #
               get_lda_app_and_usage(group_level='usage', drop=drop, agg_col='p_sub_type', agg_method='count', n_topics=n_topics),
               get_lda_app_and_usage(group_level='usage', drop=0, agg_col='p_sub_type', agg_method='count', n_topics=n_topics),

                get_lda_app_and_usage(group_level='usage', drop=drop, agg_col='p_sub_type', agg_method='sum', n_topics=n_topics),
                get_lda_app_and_usage(group_level='usage', drop=0, agg_col='p_sub_type', agg_method='sum', n_topics=n_topics),

        # get_lda_app_and_usage('duration', drop=True, group_type=group_type),
               # get_lda_app_and_usage('duration', drop=False, group_type=group_type),

               ]

    for df in df_list:
        if 'device' in df.columns:
            df.set_index('device', inplace=True)


    all = pd.concat(df_list, axis=1)

    all.columns = [f'lda_{col}' for col in all.columns]
    #
    # all = all[[str(i) for i in range(0, n_topics)]]
    print(f'Device_pkg all column:{all.columns}')

    all = all.reset_index()
    return all


@timed()
@file_cache(overwrite=False)
def get_lda_app_and_usage(group_level='usage', drop=False, agg_col='package', agg_method='count', n_topics=5):
    from code_felix.tiny.tfidf import get_cntTf
    cntTf = get_cntTf(group_level, agg_col=agg_col, agg_method=agg_method)

    if drop:
        cntTf = drop_useless_package(cntTf, drop)

    print(f'Try to lda for group_level#{group_level}, agg_col#{agg_col}, agg_method#{agg_method}')

    tmp = cntTf / cntTf

    docres = get_lda_docres(cntTf, n_topics)


    docres['app_length'] = tmp.sum(axis=1)
    docres.columns = [f'{group_level}_{agg_col}_{agg_method}_drop:{drop}_{col}' for col in docres.columns]


    print(f'Already calculate lda for {type} DF')

    #docres.drop(columns=['apps'], inplace=True)
    #print(f'deviceid_packages column:{deviceid_packages.columns}')
    docres.sort_index(inplace=True)
    docres.index.name = 'device'
    docres.reset_index(inplace=True)
    return docres



def get_lda_docres(cntTf, n_topics):
    # Replace point
    print(f'cntTf type:{type(cntTf)}')
    # if not isinstance(cntTf, pd.DataFrame):
    #     cntTf = pd.DataFrame(cntTf.toarray())
    cntTf.fillna(0, inplace=True)
    lda = LatentDirichletAllocation(n_topics=n_topics,
                                    learning_offset=50.,
                                    random_state=666)
    #print(f'cntTf column:{cntTf.columns}')

    import scipy
    print('Convert df to csr_matrix')
    cntTf_sparse =  scipy.sparse.csr_matrix(cntTf.values)
    print('Lda analysis begin')
    docres = lda.fit_transform(cntTf_sparse)
    print('Lda analysis end')
    docres = pd.DataFrame(docres, index=cntTf.index)
    return docres


if __name__ == '__main__':
    # get_lda_from_usage(5)
    #extend_package(version=1)
    drop = 18363
    for n_topics in range(5, 30, 5):
        get_lda_app_and_usage(group_level='usage', drop=drop, agg_col='p_sub_type_knn', agg_method='count', n_topics=n_topics),
        get_lda_app_and_usage(group_level='usage', drop=0,    agg_col='p_sub_type_knn', agg_method='count', n_topics=n_topics),

        get_lda_app_and_usage(group_level='usage', drop=drop, agg_col='p_sub_type_knn', agg_method='sum', n_topics=n_topics),


        #Done
        get_lda_app_and_usage(group_level='usage', drop=0,    agg_col='p_sub_type_knn', agg_method='sum', n_topics=n_topics),



