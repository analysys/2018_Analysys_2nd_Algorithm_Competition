#from tiny.usage import extend_feature
from code_felix.tiny.usage import extend_feature
from code_felix.utils_.util_cache_file import *

from code_felix.utils_.util_pandas import convert_label_encode
import pandas as pd
import numpy as np

try:
    from code_felix.tiny.conf import *
except :
    mini=False
    version=4

import matplotlib as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


@timed()
def get_brand():
    brand = pd.read_csv('input/deviceid_brand.tsv', sep='\t', header=None)

    brand.columns = ['device', 'brand', 'phone_type']
    #
    # tmp2 = brand.groupby(['brand']).count().sort_values('device', ascending=False)
    # tmp2['cumsum'] = tmp2.device.cumsum()
    # #tmp2['percentage'] = tmp2['cumsum'] / 72554
    # brand_1 = brand[brand.brand.isin(tmp2[:limit].index)]
    #
    # brand_2 = brand[~brand.brand.isin(tmp2[:limit].index)]
    # brand_2.brand = 'Other'
    # brand_2.phone_type = 'Other'
    #
    # print(f'Size of brand1:{len(brand_1)} , brand2:{len(brand_2)}')
    #
    #
    # brand = pd.concat([brand_1, brand_2])
    # brand = brand.sort_values('brand')
    return brand

#
# # Performance issue
# @timed()
# def get_package(limit=None):
#     cache_file = './output/deviceid_package.tsv'
#     if os.path.exists(cache_file):
#         print('load package from cache:%s' % cache_file)
#         return pd.read_csv(cache_file, sep=',')
#     else:
#         tmp = pd.read_csv('input/deviceid_packages.tsv', sep='\t', header=None, nrows=limit)
#         tmp.columns = ['device', 'package_list']
#
#         tmp = tmp[tmp.device.isin(get_test().iloc[:, 0])]
#
#         package_list_all = frozenset.union(*tmp.iloc[:, 1].apply(lambda v: frozenset(v.split(','))))
#
#         print(len(package_list_all))
#
#         i = 1
#         for package in package_list_all:
#             i += 1
#             print(f'{i}/{len(package_list_all)}')
#             tmp[package] = tmp.apply(lambda _: int(package in _.package_list), axis=1)
#
#         tmp.to_csv('./output/deviceid_package.tsv', index=False)
#
#         return tmp


@timed()
def get_package_label(package_list=None):
    package = pd.read_csv('input/package_label.tsv', sep='\t', header=None, )
    package.columns = ['package', 'p_type', 'p_sub_type']
    if package_list is None:
        return package
    else:
        return package[package.package.isin(package_list)]
#
# @timed()
# def get_max_week(df):
#     """
#     到到每个Device使用的最多的那一周
#     :param df:
#     :return:
#     """
#     df = df.groupby(['device', 'weekbegin']).agg({'weekday':'nunique','package':'nunique'}).reset_index()
#
#     # .to_frame().reset_index()
#     # df.sort_values(['device', 'package'])
#
#     df = df.sort_values(by=['device', 'weekday','package', 'weekbegin'], ascending=False).groupby('device').nth(0)
#     df = df.reset_index()
#     df.rename({'package':'package_count'}, axis=1, inplace=True)
#
#     return df


# @timed()
# def get_percent_duration(df, groupby=['device', 'weekday'], prefix=None, span_no=6):
#     prefix = groupby[-1] if prefix is None else prefix
#     sum_duration = get_sum_duration(df, groupby, prefix)
#
#     sum_duration = reduce_time_span(sum_duration, prefix, span_no)
#
#     for col in [item for item in sum_duration.columns if f'{prefix}_span_' in item]:
#         df[f'{col}_p'] = round(df[col] / df[f'{prefix}_total'], 3)
#
#     return df



def get_train():
    columns_1  =[ #sum and percentage
        'total_day_00_0_du',
        'total_day_00_1_du',
        'total_day_00_2_du',
        'total_day_00_3_du',
        '......'
        'total_day_07_3_du',


        'maxmax_day_00_0_du',
        'max_day_00_1_du',
        'max_day_00_2_du',
        'max_day_00_3_du',
        '......'
        'max_day_07_3_du',
    ]

    columns_2 =    [

        #指定时间段,打开pak的统计时长
        'max_package_du_1',  'max_package_du_1_type', 'max_package_du_1_sub_type',
        'max_package_du_2',  'max_package_du_2_type', 'max_package_du_2_sub_type',
        'max_package_du_3',  'max_package_du_3_type', 'max_package_du_3_sub_type',

        #指定时间段,打开pak的次数
        'max_package_cnt_1',  'max_package_cnt_1_type', 'max_package_cnt_1_sub_type',
        'max_package_cnt_2',  'max_package_cnt_2_type', 'max_package_cnt_2_sub_type',
        'max_package_cnt_3',  'max_package_cnt_3_type', 'max_package_cnt_3_sub_type',

        'total_used_package_count', 'per_total_install',
        'weekly_used_package_count', 'per_weekly_install',
        'total_install_package_count',

        'package_top1 ,,,, package_top10',
        'device_brand', 'device_type'

    ]
    pass




def get_test():
    test = pd.read_csv('input/deviceid_test.tsv', sep='\t',header=None)
    return test

def extend_pkg_label(df=None):


    pkg_label = get_package_label()
    #pkg_label.set_index('package', inplace=True)

    pkg_label['combine_type'] = pkg_label.apply(lambda row: f'{row.p_type}_{row.p_sub_type}', axis=1)
    if df is None:
        return pkg_label
    else:
        df = pd.merge(df, pkg_label, on='package', how='left')
        #
        # kmeans = get_app_group()
        # df = pd.merge(df, kmeans, on='package', how='left')
        #
        df[['p_type', 'p_sub_type','combine_type' ]] = df[['p_type','p_sub_type', 'combine_type' ]].fillna('Unknown')

        return df

@timed()
def extend_device_brand(tmp):

    brand = get_brand()
    #print(f'column list:{tmp.columns}')
    if tmp is None:
        return brand
    else:
        if 'device' not in tmp:
            tmp.index.name = 'device'
            tmp.reset_index(inplace=True)
        tmp = tmp.merge(brand, how='left')
        return tmp




# def extend_sum_duration_df(df, groupby=['device', 'weekday'], prefix=None):
#     total = get_sum_duration(df, ['device'], 'total')
#
#     max_week = get_max_week(df)
#
#     merge = df.merge(max_week, on=['device', 'weekbegin'])
#
#     max = get_sum_duration(merge, ['device'], 'max')
#
#     return pd.concat( [total, max], axis=1 ).reset_index()




@timed()
#@file_cache()
def split_days_all(tmp):

    tmp.duration = (tmp.close - tmp.start) / np.timedelta64(1, 'D')

    old_len = len(tmp)
    print(f'Out loop: The original Df size is {old_len}')
    tmp = split_days(tmp, 50)
    tmp = split_days(tmp, 1)
    print(f'Out loop: The new Df size is {len(tmp)}, old df size is {old_len}')

    tmp['start_base'] = tmp['start'].dt.date
    tmp['weekday'] = tmp.start.dt.weekday
    tmp['weekbegin'] = (tmp['start'] -
                        tmp['start'].dt.weekday.astype('timedelta64[D]')).dt.date


    tmp.duration = round(tmp.duration, 6)

    tmp = tmp.sort_values(by = ['device','package','start'])

    return tmp

#@timed()
def split_days(tmp, threshold_days = 100):
    threshold_days = max(1,threshold_days)
    # print(f'The input df#{len(tmp)} before split, with max duration:{tmp.duration.max()} '
    #                              f'and  threshold_days@{threshold_days}')

    # 检查是否有需要截断的数据, 如果没有直接Return, 或者进入小循环
    tmp_todo_big = tmp[tmp.duration > threshold_days]
    if len(tmp_todo_big) == 0 and tmp.duration.max() <= threshold_days:
        print(f'Final return with para:{threshold_days}:{len(tmp)}')
        return tmp


    # 创建新记录,截取最近的时间段(大段)
    tmp_todo_big.start = tmp_todo_big.start.dt.date + pd.DateOffset(threshold_days)
    tmp_todo_big.duration = (tmp_todo_big.close - tmp_todo_big.start) / np.timedelta64(1, 'D')
    tmp_big = split_days(tmp_todo_big, threshold_days)

    # inpu中,已经小于阀值天的
    tmp_small_p1         = tmp[tmp.duration <= threshold_days]
    # 旧记录,保留早期的时间段(小段)
    tmp_small_p2   = tmp[tmp.duration > threshold_days]
    tmp_small_p2.close    = tmp_small_p2.start.dt.date + pd.DateOffset(threshold_days)
    #tmp_small_p2.duration = (tmp_small_p2.close - tmp_small_p2.start) / np.timedelta64(1, 'D')
    tmp_small = pd.concat([tmp_small_p1, tmp_small_p2])
    tmp_small.duration = (tmp_small.close - tmp_small.start) / np.timedelta64(1, 'D')

    # print(f'max duration:{tmp_small_p2.duration.max()} with small threshold:{threshold_days}')

    tmp = tmp_big.append(tmp_small)

    #tmp = tmp.sort_values('duration', ascending=False)
    tmp.reset_index(drop=True, inplace=True)

    # print(f'The output df#{len(tmp)} after split')

    return tmp

@timed()
#@file_cache()
def get_start_closed(file=None):

    file = file if file is not None else './output/start_close/deviceid_package_start_close_40_38_35780089_36720940.csv'

    start_close = pd.read_csv(file,
                              # index_col=0 ,
                              nrows=None,
                              header=None )

    if len(start_close) == 0 :
        return pd.DataFrame()

    start_close.columns = ['device', 'package', 'start_t', 'close_t']

    print(f'Sort the df#{len(start_close)} by device(begin)')
    start_close.sort_values('device', inplace=True)
    print(f'Sort the df by device(end)')

    # start_close.index.name = 'device'


    start_close['start'] = pd.to_datetime(start_close.loc[:, 'start_t'], unit='ms')
    start_close['close'] = pd.to_datetime(start_close.loc[:, 'close_t'], unit='ms')

    len_original = len(start_close)
    start_close = start_close[start_close.start < pd.to_datetime('now')]
    #去除部分异常数据
    print(f'Remove {len_original-len(start_close)} records data')

    # start_close.groupby('device')[['package']].count()
    start_close['duration'] = (start_close.close - start_close.start) / np.timedelta64(1, 'D')

    return start_close

def replace_invalid_filename_char(filename):
    invalid_characaters = '\':"<>|{} ,'
    for char in invalid_characaters:
        filename = filename.replace(char, '')
    return filename


def attach_device_train_label(df):

    deviceid_test = pd.read_csv('./input/deviceid_test.tsv', sep='\t', names=['device'])
    deviceid_train = pd.read_csv('./input/deviceid_train.tsv', sep='\t', names=['device', 'sex', 'age'])

    deviceid_train = pd.concat([deviceid_train, deviceid_test])

    deviceid_train['sex'] = deviceid_train['sex'].apply(lambda x: str(x))
    deviceid_train['age'] = deviceid_train['age'].apply(lambda x: str(x))

    def tool(x):
        if x == 'nan':
            return x
        else:
            return str(int(float(x)))

    deviceid_train['sex'] = deviceid_train['sex'].apply(tool)
    deviceid_train['age'] = deviceid_train['age'].apply(tool)
    deviceid_train['sex_age'] = deviceid_train['sex'] + '-' + deviceid_train['age']
    deviceid_train = deviceid_train.replace({'nan': np.NaN, 'nan-nan': np.NaN})
    if df is not None:
        df = pd.merge(df, deviceid_train, on='device', how='left')
        df.sort_values('device', inplace=True)

        return df
    else :
        return deviceid_train


@timed()
@file_cache()
def get_stable_feature(version):
    return get_dynamic_feature(20)

@timed()

def get_dynamic_feature(svd_cmp):
    from code_felix.tiny.lda import get_lda_from_usage
    drop_useless_pkg = True
    drop_long = 0.3
    n_topics = 5
    lda_feature = get_lda_from_usage(n_topics)
    feature = extend_feature(span_no=24, input=lda_feature,
                             drop_useless_pkg=drop_useless_pkg, drop_long=drop_long, svd_cmp=svd_cmp)
    feature = convert_label_encode(feature)
    feature_label = attach_device_train_label(feature)
    feature_label = feature_label.sort_values('device')
    return feature_label


@timed()
def split_train(df,  label_col='sex_age'):



    #df['sex'] = df['sex'].astype('category')

    X= df.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
    y= df[label_col]
    logger.debug(f'type is {y.dtype}')
    X_train, X_test, y_train, y_test = train_test_split(X, y.cat.codes)

    return  X_train, X_test, y_train, y_test

def train_test_split(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=666)
    logger.debug(f'X_train:{X_train.shape}, X_test:{y_train.shape}')
    return X_train, X_test, y_train, y_test

def balance_train(df, ratio):
    if ratio == 0:
        return df
    else:
        small_part_cnt = df.sex.value_counts().min()
        df = df.sort_values('sex')
        bal = pd.concat([df[:small_part_cnt], df[-small_part_cnt:]])
        return bal



def print_imp_list( train, clf, order_by_wight=True, show_zero=True):
    if hasattr(clf, 'feature_importances_'):
        imp_item = dict(zip(train.columns, clf.feature_importances_))

        imp_list = sorted(imp_item.items(), key=lambda imp: imp[1], reverse=True)

        # for key, value in imp_list:
        #     if value > 0:
        #         print(f'Import {value}: {key}')
        #         print(train[str(key)].dtype.name)
        #     else:
        #         print(f'zeor imp:{key}')
        #

        zero_list = [key for key, value in imp_list if value==0]

        print(f'Full List:{len(train.columns)}, Zero List:{len(zero_list)}, ')


        imp_list = [(key, value, train[key].dtype.name) for key, value in imp_list if value>0]

        if order_by_wight :
            imp_list = sorted(imp_list, key=lambda imp: imp[1], reverse=True)
        else:
            imp_list = sorted(imp_list, key=lambda imp: imp[2])

        import_sn = 0
        for (key, value, dtype) in imp_list:
            import_sn += 1
            logger.info("%03d: %s, %s, %s" % ( import_sn, str(key).ljust(35), str(value).ljust(5), dtype))

        print(f'Full List:{len(train.columns)}, Zero List:{len(zero_list)}, ')

def visual_importnance(X, forest):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()


def save_result_for_ensemble(name, train,  test, label):
    """"
    name = '{score}_name'
    """


    logger.debug(f'Train:{train.shape} , label:{label.shape if label is not None else None } , test: {test.shape}')

    file = f'./sub/baseline_{name}.h5'
    file = replace_invalid_filename_char(file)
    store = pd.HDFStore(file)

    store["train"] = train
    if label is not None: store["label"] = label
    store["test"] = test
    store.close()
    logger.debug(f"Ensamble file save to file: {file}")
    return file

def reduce_low_frequency( words ) :
    from code_felix.tiny.usage import get_app_count_sum
    app_count = get_app_count_sum()
    impact_list = app_count[app_count.count_ >= 2]
    mini = [item for item in words if item in impact_list.package]
    logger.debug(f'{len(words) - len(mini)} words was removed')
    return mini


def random_feature(df, ratio, require_list=['sex', 'age', 'sex_age', 'device']):
    total = len(df.columns)
    mini_size = round(total*ratio)
    mini_df = df.sample(mini_size , axis=1)
    logger.debug(f"Try to convert the column from {total} to {mini_size}")
    for col in require_list:
        mini_df[col] = df[col]
    logger.debug(f"Convert the column from {total} to {mini_size}")
    return mini_df

def ensemble_feature_other_model(df, files):
    from code_felix.merge.dnn_merge import read_result_for_ensemble
    feature_list = []
    for name, file in files:
        train, _, test = read_result_for_ensemble(file)
        feature = pd.concat([train, test])
        feature.columns = [f'{name}_{col}' for col in feature.columns]
        feature_list.append(feature)
    all = pd.concat(feature_list, axis=1)
    all.index.name = 'device'
    all = all.reset_index()
    logger.debug(f"Ensemble {len(all.columns)} column_name for {name}:{all.columns}")
    if df is not None:
        logger.debug("try to merge ensemble columns")
        all = pd.merge(df, all, on='device', how='left')
        logger.debug("End to merge ensemble columns")
    return all




def reorder_train(df):
    df = df.set_index('device')
    train = pd.read_csv("./input/deviceid_train.tsv", encoding='utf8', sep='\t',header=None)
    train.columns=['device','sex','age'] ;
    train=train.set_index('device');
    train=train.drop(['sex','age'],axis=1)
    return train.join(df)



def reorder_test(df):
    print(df.columns)
    df = df.set_index('device')
    test = pd.read_csv("./input/deviceid_test.tsv", encoding='utf8', sep='\t', header=None)
    test.columns = ['device'];
    test = test.set_index('device');
    return test.join(df)



def get_category():
    return pd.Categorical(get_score_column())


def get_score_column():
    return ['1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '2-0', '2-1', '2-2',
         '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']


if __name__ == '__main__':
    # drop_useless_pkg = True
    # drop_long = 0.3
    # feature = extend_feature(span_no=24, input=None,
    #                          drop_useless_pkg=drop_useless_pkg, drop_long=drop_long)


    file_list = [

        ('lgb', './output/best/baseline_2.62099_287_lgb_min_data_in_leaf1472.h5'),
        ('dnn', './output/best/2.6337418931325276_1303_dnn.h5'),

    ]


    tmp = ensemble_feature_other_model(None, file_list)

    logger.debug(tmp.shape)