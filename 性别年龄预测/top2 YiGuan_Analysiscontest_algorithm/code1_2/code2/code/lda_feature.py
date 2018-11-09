from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
lda = LatentDirichletAllocation(n_topics=50,learning_offset=50.,random_state=666)

path='../data/Demo/'

deviceid_packages=pd.read_csv(path+'deviceid_packages.tsv',sep='\t',names=['device_id','apps'])
deviceid_packages['apps']=deviceid_packages['apps'].apply(lambda x:x.split(','))
apps=deviceid_packages['apps'].apply(lambda x:' '.join(x)).tolist()

deviceid_train=pd.read_csv(path+'deviceid_train.tsv',sep='\t',names=['device_id','sex','age'])
deviceid_test=pd.read_csv(path+'deviceid_test.tsv',sep='\t',names=['device_id'])

vectorizer=CountVectorizer()
cntTf = vectorizer.fit_transform(apps)
docres = lda.fit_transform(cntTf)

deviceid_packages=pd.concat([deviceid_packages,pd.DataFrame(docres)],axis=1)
temp=deviceid_packages.drop('apps',axis=1)

deviceid_train=pd.merge(deviceid_train,temp,on='device_id',how='left')
deviceid_test=pd.merge(deviceid_test,temp,on='device_id',how='left')

deviceid_train=deviceid_train.drop(['sex','age','device_id'],axis=1)
deviceid_test=deviceid_test.drop(['device_id'],axis=1)

deviceid_train.to_csv('../data/train/lda_fea.csv',index=None)
deviceid_test.to_csv('../data/test/lda_fea.csv',index=None)