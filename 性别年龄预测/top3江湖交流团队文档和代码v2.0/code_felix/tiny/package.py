from sklearn.feature_extraction.text import CountVectorizer

from code_felix.utils_.util_date import *
from code_felix.utils_.util_cache_file import *
from functools import lru_cache

@lru_cache()
@timed()
@file_cache()
def get_drop_list_for_install(limit=18363):
    """
    total:35000, 18363 is 0 install base on testing data
    :param limit:
    :return:
    """
    deviceid_packages = pd.read_csv('./input/deviceid_packages.tsv', sep='\t', names=['device', 'apps'])
    deviceid_test = pd.read_csv('./input/deviceid_test.tsv', sep='\t', names=['device'])
    # deviceid_train = pd.read_csv('./input/deviceid_train.tsv', sep='\t', names=['device', 'sex', 'age'])
    #
    # deviceid_train = pd.concat([deviceid_train, deviceid_test])

    deviceid_packages['apps'] = deviceid_packages['apps'].apply(lambda x: x.split(','))
    deviceid_packages['app_lenghth'] = deviceid_packages['apps'].apply(lambda x: len(x))

    apps = deviceid_packages['apps'].apply(lambda x: ' '.join(x)).tolist()
    vectorizer = CountVectorizer()
    cntTf = vectorizer.fit_transform(apps)

    tokens = vectorizer.get_feature_names()
    device_app = pd.DataFrame(data=cntTf.toarray(), index=deviceid_packages.device,
                 columns=tokens)

    device_app_test = device_app[device_app.index.isin(deviceid_test.device)]

    device_app_test = device_app_test.sum()

    device_app_test.index.rename('package', inplace=True)
    device_app_test = device_app_test.sort_values()
    res = device_app_test.to_frame().reset_index()
    if limit == True :
        return res[:18363]
    else:
        return res[:limit]

def drop_useless_package(df, limit=18363):
    useless = get_drop_list_for_install(limit)
    #Drop the package by row
    if 'package' in df:
        old_len = len(df)
        df = df[~df.package.isin(useless.package)]
        print(f'drop_useless_package: {old_len} rows remove useless app({len(useless)}) to f{len(df)})')

    #Drop the package by column
    else:
        # print(f'Column will be remove:{useless.package[:10]}')
        # print(f'Column will be compare:{df.columns[:10]}')
        df.columns = [ col.split('_')[-1]
                       if '_' in col  else col
                       for col in df.columns
                       ]
        columns = [ col for col in df.columns if col in useless.package.values ]
        print(f'drop_useless_package: There are {len(columns)} column will be droped from {len(df.columns)} columns:{columns[:10]}')
        df.drop(columns=columns, inplace=True)
    return df
#
# @timed()
# def get_drop_list_for_usage():
#     df = extend_package(version=1, mini=False)
#     count = df[[col for col in df.columns if 'count_' in col]]
#
#     deviceid_test = pd.read_csv('./input/deviceid_test.tsv', sep='\t', names=['device'])
#
#     device_usage_test = count[count.index.isin(deviceid_test.device)]
#
#     tmp = device_usage_test.sum()
#     return tmp[tmp==0]
#




#Can not save to pkl
#@file_cache(type='pkl', overwrite=False)
@timed()
def base_on_package_install_for_TF(type='package'):
    deviceid_packages = pd.read_csv('./input/deviceid_packages.tsv', sep='\t', names=['device', 'apps'])
    #deviceid_packages=deviceid_packages[:1000]
    deviceid_packages.sort_values('device', inplace=True)
    print(f'Try to load packge for type:{type}')
    deviceid_packages['apps'] = deviceid_packages['apps'].apply(lambda x: x.split(','))
    # deviceid_packages['app_lenghth'] = deviceid_packages['apps'].apply(lambda x: len(x))
    apps = deviceid_packages['apps'].apply(lambda x: ' '.join(x)).tolist()
    vectorizer = CountVectorizer()
    cntTf_app = vectorizer.fit_transform(apps)
    cntTf_app = pd.DataFrame(cntTf_app.toarray(),
                                     columns=vectorizer.get_feature_names(),
                                     index=deviceid_packages.device)

    return cntTf_app



@timed()
#@file_cache(type='pkl', overwrite=True)
def extend_package_count_df(df, col='package'):
    p = df.groupby(['device', col])['start_base'].nunique().reset_index()
    #p = df.groupby(['device', 'package'])['duration'].sum().reset_index()
    p = p.pivot(index='device', columns=col, values='start_base').reset_index()
    print(f'Device_Package: convert {df.shape} to {p.shape} ')
    p.set_index('device', inplace=True)
    p.columns=[f'count_{col}_{item}' for item in p.columns]
    return p

@timed()
#@file_cache(type='pkl', overwrite=True)
def extend_package_duration_df(df, col='package'):
    #p = df.groupby(['device', 'package'])['start_base'].nunique().reset_index()
    p = df.groupby(['device', col])['duration'].sum().reset_index()
    p = p.pivot(index='device', columns=col, values='duration').reset_index()
    print(f'Device_Package: convert {df.shape} to {p.shape} ')
    p.set_index('device', inplace=True)
    p.columns = [f'duration_{col}_{item}' for item in p.columns]
    return p


def extend_package_TF(df, col='package'):
    return pd.concat([
                     extend_package_count_df(df, col=col) ,
                      extend_package_duration_df(df, col=col),
                      ], axis=1)



