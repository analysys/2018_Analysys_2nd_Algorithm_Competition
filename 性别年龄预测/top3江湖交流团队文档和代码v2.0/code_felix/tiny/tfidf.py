from sklearn.feature_extraction.text import TfidfTransformer
from code_felix.tiny.lda import *
from code_felix.tiny.util import *

from code_felix.utils_.util_cache_file import *


#
# @timed()
# @file_cache(type='pkl',overwrite=False)
# def get_tfidf(type, drop, limit):
#     tfidf = get_tfidf_app_and_usage(type, drop)
#
#     col = list(tfidf.sum().sort_values(ascending=False)[:limit].index)
#
#     return tfidf[col]
#
# @timed()
# @file_cache(type='pkl',overwrite=False)
# def get_tfidf_app_and_usage(type, drop):
#     cntTf = get_cntTf(type)
#
#     if drop:
#         cntTf = drop_useless_package(cntTf)
#     tfidf =  cal_tfidf(cntTf)
#
#     tfidf = tfidf.to_dense().replace({0: np.nan}).to_sparse().fillna(0)
#     # tmp = tfidf / tfidf
#     # tfidf[f'{type}_tfidf_{drop}_length'] = tmp.sum(axis=1)
#
#     return tfidf
#
@timed()
def cal_tfidf(cntTf):
    index = cntTf.index

    transformer = TfidfTransformer()
    print('Try to sparse cntTF')
    cntTf.fillna(0, inplace=True)
    cntTf = cntTf.to_sparse(fill_value=0)
    print(f'Density before TFIDF:{cntTf.density}')
    #cntTf = scipy.sparse.csr_matrix()
    print(f'Try to cal tfidf for {type(cntTf)}')
    tfidf = transformer.fit_transform(cntTf)

    df_weight = pd.SparseDataFrame(tfidf.toarray(), index=index)
    return df_weight


@timed()
@file_cache(type='pkl', overwrite=False)
def base_on_usage_for_TF(version, mini=False, col='package', thres_hold=0):
    rootdir = './output/start_close/'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    path_list = sorted(list, reverse=True)
    path_list = [os.path.join(rootdir, item) for item in path_list if item.endswith('csv')]


    from multiprocessing import Pool as ThreadPool
    from functools import partial
    pool = ThreadPool(processes=8)
    process_file = partial(cal_tf_for_individual_file, col=col, thres_hold=thres_hold)

    results = pool.map(process_file, path_list)

    df = pd.concat(results)
    df.fillna(0,inplace=True)
    print(f'Share of device package usage is:{df.shape}')

    return df.sort_index()


def cal_tf_for_individual_file(path, col, thres_hold):
    from code_felix.tiny.util import get_start_closed, split_days_all
    from code_felix.tiny.package import extend_package_TF
    if os.path.isfile(path) and 'csv' in path:
        print(f"Try to summary file:{path}")
        df = get_start_closed(path)

        # 需要packge的类型,就扩展app的类型:p_type, p_sub_type
        if type != 'package':
            print(f'Try to merge pkg label for col:{col}')
            from code_felix.tiny.util import extend_pkg_label
            df = extend_pkg_label(df)

            from code_felix.tiny.knn import extend_pkg_label_knn
            df = extend_pkg_label_knn('p_type', df)
            df = extend_pkg_label_knn('p_sub_type', df)
            df = extend_pkg_label_knn('combine_type', df)

        df = split_days_all(df)
        df = df[df.duration >= thres_hold]
        df = extend_package_TF(df, col=col)
        return df
    else:
        return pd.DataFrame()


@timed()
@file_cache(type='pkl')
def get_cntTf( group_level, agg_col, agg_method, thres_hold=0):
    from code_felix.tiny.package import base_on_package_install_for_TF
    version = 4
    mini = False
    if group_level == 'app':
        cntTf_app = base_on_package_install_for_TF(agg_col)
        cntTf = cntTf_app
    elif group_level =='usage' and agg_method == 'count':
        cntTf_all = base_on_usage_for_TF(version=version, mini=mini, col=agg_col, thres_hold=thres_hold)
        cntTf_count = cntTf_all[[col for col in cntTf_all.columns if 'count_' in col]]
        cntTf = cntTf_count
    elif group_level =='usage' and agg_method == 'sum':
        cntTf_all = base_on_usage_for_TF(version=version, mini=mini, col=agg_col)
        cntTf_duration = cntTf_all[[col for col in cntTf_all.columns if 'duration_' in col]]
        cntTf = cntTf_duration
    else:
        cntTf = pd.DataFrame()
        print(f'Unknown params:{group_level}')

    cntTf.fillna(0, inplace=True)
    return cntTf.to_sparse(fill_value=0)

@file_cache(overwrite=False)
def get_tfidf(summary=True):
    cntTf = get_cntTf('app', 'package', None)
    tfidf = cal_tfidf(cntTf);
    if summary:
        return tfidf.sum(axis=1).reset_index(name='tfidf')
    else:
        return tfidf

def attach_tfidf(df):
    import pandas as pd
    return pd.merge(df, get_tfidf(summary=True), how='left', on='device')


def get_svd_tfidf(n_components):

    if n_components==0 or n_components is None:
        return get_stable_svd_feature()
    else:
        cntTf = get_cntTf('usage', agg_col='p_sub_type_knn', agg_method='count')
        df_new =  get_svd_tfidf_individual('usg_tf_sub_type', cntTf, n_components)

        df_stable = get_stable_svd_feature()
        df_stable.set_index('device', inplace=True)

        all = pd.concat([df_stable, df_new], axis=1 ).reset_index()
        logger.debug(f"SVD columns:{all.columns}")
        return all


@file_cache()
@timed()
def get_stable_svd_feature():
    cntTf = get_cntTf('usage', agg_col='p_sub_type_knn', agg_method='count')
    tfidf = cal_tfidf(cntTf)
    df1 =  get_svd_tfidf_individual('usg_sub_type', tfidf, 48)

    cntTf = get_cntTf('usage', agg_col='package', agg_method='count')
    tfidf = cal_tfidf(cntTf)
    df2 =  get_svd_tfidf_individual('usg_pkg', tfidf, 170 )

    cntTf = get_cntTf('usage', agg_col='package', agg_method='sum')
    tfidf = cal_tfidf(cntTf)
    df3 =  get_svd_tfidf_individual('tmp', tfidf, 80 )

    all = pd.concat([df1, df2, df3], axis=1)
    return all.reset_index()



def get_svd_tfidf_individual(level, tfidf, n_components):


    logger.debug('Convert df to csr_matrix for svd')
    import scipy
    X = scipy.sparse.csr_matrix(tfidf.values)

    import sklearn.decomposition as skd
    logger.debug("Try to cal svd")
    trsvd = skd.TruncatedSVD(n_components, random_state=42)
    transformed = trsvd.fit_transform(X)
    print(transformed.shape)
    transformed = pd.DataFrame(transformed, index=tfidf.index)
    transformed.columns = [f'svd_{level}_{n_components}_{item}' for item in transformed.columns]
    transformed.index.name = 'device'
    return transformed


#
# @timed()
# @file_cache()
# def get_cntTf_group(type):
#     cntTf = get_cntTf(type)
#     cntTf_label = extend_pkg_label(cntTf)
#     cntTf_label.replace({0:np.nan})
#     return cntTf_label.groupby(['p_type','p_sub_type']).agg(['sum','count'])

if __name__ == '__main__':

    get_cntTf('usage', agg_col='p_sub_type_knn', agg_method='count')

    # for svd_cmp in range(5, 20, 2):
    #     get_svd_tfidf(svd_cmp)


    #tfidf = cal_tfidf(cntTf);

    # for group_level in ['usage']:
    #     for agg_col in ['p_sub_type', 'package']:
    #         for agg_method in ['sum', 'count']:
    #             get_cntTf(group_level, agg_col, agg_method)
    #
    # get_cntTf('app', 'package', None)
    # get_cntTf('app', 'p_sub_type', None)