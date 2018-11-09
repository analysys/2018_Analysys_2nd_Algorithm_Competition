import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from scipy import sparse
from sklearn.pipeline import FeatureUnion,Pipeline,make_union,make_pipeline
from scipy.sparse import coo_matrix
from common import timer,read_csv,ItemSelector,TextStats

path='../data/'

with timer("Load data"):
    df_apps_train = read_csv(path+"train/apps.csv")
    df_apps_test = read_csv(path+"test/apps.csv")

df_apps = pd.concat([df_apps_train,df_apps_test])

feature_union = make_union(
        make_pipeline(ItemSelector(key="apps"),CountVectorizer(analyzer='word',ngram_range=(1,1))),
        make_pipeline(ItemSelector(key="apps"),TfidfVectorizer(analyzer='word',ngram_range=(1,1),use_idf=False)),
        make_pipeline(ItemSelector(key="apps"),TextStats(), DictVectorizer())
        )

with timer("Fit feature_union"):
        feat = feature_union.fit_transform(df_apps)

out_col = ["app_stat_%s" % (str(i)) for i in range(feat.shape[1])]
output_file='123'
with timer("Save file to %s" % (output_file)):
    data=coo_matrix((feat.todense())).tocsr()
    sparse.save_npz('../data/train/train_app_stat.npz',data[:df_apps_train.shape[0]])
    sparse.save_npz('../data/test/test_app_stat.npz',data[df_apps_train.shape[0]:])

with timer("Load data"):
    df_apps_train = read_csv(path+"train/install_apps_1.csv")
    df_apps_test = read_csv(path+"test/install_apps_1.csv")

df_apps = pd.concat([df_apps_train,df_apps_test])

feature_union = make_union(
        make_pipeline(ItemSelector(key="install_app1"),CountVectorizer(analyzer='word',ngram_range=(1,1))),
        make_pipeline(ItemSelector(key="install_app1"),TfidfVectorizer(analyzer='word',ngram_range=(1,1),use_idf=False)),
        make_pipeline(ItemSelector(key="install_app1"),TextStats(), DictVectorizer())
        )

with timer("Fit feature_union"):
        feat = feature_union.fit_transform(df_apps)

out_col = ["app_stat_%s" % (str(i)) for i in range(feat.shape[1])]
output_file='123'
with timer("Save file to %s" % (output_file)):
    data=coo_matrix((feat.todense())).tocsr()
    sparse.save_npz('../data/train/install_apps_1.npz',data[:df_apps_train.shape[0]])
    sparse.save_npz('../data/test/install_apps_1.npz',data[df_apps_train.shape[0]:])



with timer("Load data"):
    df_apps_train = read_csv(path+"train/use_apps_1.csv")
    df_apps_test = read_csv(path+"test/use_apps_1.csv")

df_apps = pd.concat([df_apps_train,df_apps_test])

feature_union = make_union(
        make_pipeline(ItemSelector(key="app1"),CountVectorizer(analyzer='word',ngram_range=(1,1))),
        make_pipeline(ItemSelector(key="app1"),TfidfVectorizer(analyzer='word',ngram_range=(1,1),use_idf=False)),
        make_pipeline(ItemSelector(key="app1"),TextStats(), DictVectorizer())
        )

with timer("Fit feature_union"):
        feat = feature_union.fit_transform(df_apps)

out_col = ["app_stat_%s" % (str(i)) for i in range(feat.shape[1])]
output_file='123'
with timer("Save file to %s" % (output_file)):
    data=coo_matrix((feat.todense())).tocsr()
    sparse.save_npz('../data/train/use_apps_1.npz',data[:df_apps_train.shape[0]])
    sparse.save_npz('../data/test/use_apps_1.npz',data[df_apps_train.shape[0]:])