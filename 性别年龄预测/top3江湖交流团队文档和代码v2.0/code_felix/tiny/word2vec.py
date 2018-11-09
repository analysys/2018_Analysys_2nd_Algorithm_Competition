#import seaborn as sns

from code_felix.tiny.usage import *

from code_felix.tiny.tfidf import *


def get_device_app_sequence_individual_file(file):
    from code_felix.tiny.usage import cal_duration_for_partition

    tmp = cal_duration_for_partition(file)

    tmp = tmp.sort_values(['device', 'start'])
    #tmp.head()

    df = pd.DataFrame( {'apps':'', 'length':0}, index = tmp.device.unique())
    #df.head()
    for name, group in tmp.groupby('device'):
        #print(type(group.package))
        df.loc[name , 'apps'] = ' '.join(group.package)
        df.loc[name , 'length'] = len(group.package)
    return df


@timed()
def get_device_app_sequence():
    """
    apps
    app1, app2, app3
    """
    rootdir = './output/start_close/'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    path_list = sorted(list, reverse=True)
    path_list = [os.path.join(rootdir, item) for item in path_list if item.endswith('csv')]

    from multiprocessing import Pool as ThreadPool

    pool = ThreadPool(processes=8)

    results = pool.map(get_device_app_sequence_individual_file, path_list)
    pool.close()
    pool.join()

    results = [item for item in results if len(item)>0]

    all = pd.concat(results)
    all.index.name='device'
    all.reset_index(inplace=True)

    app_seq = './output/apps_seq.tsv'
    all[['apps']].to_csv(app_seq, header=None, index=False)
    return app_seq

#
# def get_package_label(package_list=None):
#     package = pd.read_csv('input/package_label.tsv', sep='\t', header=None, )
#     package.columns = ['package', 'p_type', 'p_sub_type']
#     if package_list is None:
#         return package
#     else:
#         return package[package.package.isin(package_list)]


def get_dict(force=False):
    """
    apps, kms_*
    """
    from gensim.models import word2vec
    file = "./output/word2vec.model"
    if not os.path.exists(file) or force:

        #global app_seq
        INPUT_FILE = get_device_app_sequence()

        #sentences = word2vec.Text8Corpus(INPUT_FILE)  # 训练模型，部分参数如下
        sentences = word2vec.LineSentence(INPUT_FILE)
        model_20 = word2vec.Word2Vec(sentences, size=20, hs=1, min_count=0, window=5)

        model = model_20

        model.save(file)

    return  word2vec.Word2Vec.load(file)


#
# @timed()
# @file_cache(overwrite=False)
# def get_app_group():
#     model = get_dict()
#     X = model[model.wv.vocab]
#
#     from sklearn.cluster import KMeans
#     kmeans = KMeans(n_clusters=50)
#     kmeans.fit(X)
#
#     y_kmeans = kmeans.predict(X)
#     #print(y_kmeans.shape)
#
#     df = pd.DataFrame({'package':list(model.wv.vocab.keys()), 'kms_class':y_kmeans, } )
#
#     df['kms_class'] = df['kms_class'].apply(lambda val: f'kms_{val}')
#
#     return df[[ 'package', 'kms_class',]]


if __name__ == '__main__':
    pass
    model = get_dict()
    print(len(model.wv.vocab))
    # all  = get_device_app_sequence()
    # df = get_app_group()
    # print(df.shape)
    #all.to_csv('del.csv')

