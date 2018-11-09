import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack


def save_as_sparse(filename, xmtr):
    np.savez(filename,data = xmtr.data ,indices= xmtr.indices,
             indptr =xmtr.indptr, shape=xmtr.shape )

def load_as_sparse(filename):
    tmp = np.load(filename)
    return csr_matrix((tmp['data'], tmp['indices'], tmp['indptr']), shape= tmp['shape'])


# In[3]:


output = 'features/'
data='../input/'


#获得训练集deviceid信息
train = pd.read_csv(data+"deviceid_train.tsv", encoding='utf8', sep='\t',header=None)
train.columns=['deviceid','sex','age'] ; train=train.set_index('deviceid');
train['sex-age']=(train.sex-1)*11+train.age;
train=train.drop(['sex','age'],axis=1)
train['train_index']=np.arange(train.shape[0])

train['sex-age'].to_csv(output+'y_train.csv', index=False)


#获得测试集deviceid信息
test = pd.read_csv(data+"deviceid_test.tsv", encoding='utf8', sep='\t',header=None)
test.columns=['deviceid'] ; test=test.set_index('deviceid');
test['test_index']=np.arange(test.shape[0])


#获取设备的品牌信息
phone_data = pd.read_csv(data+"deviceid_brand.tsv", encoding='utf8', sep='\t',header=None)
phone_data.columns=['deviceid','brand','type'];
phone_brand_modify_data = pd.read_csv(data+"brands_modify.csv", encoding='GBK')
phone_brand_modify_data=phone_brand_modify_data.drop_duplicates(['brand','type']);


phone_brand_modify_data.head(1)


# In[10]:


#补充设备的品牌信息
phone_data=phone_data.merge(phone_brand_modify_data,on=['brand','type'],how='left')
phone_data=phone_data.drop(['brand','count','newbrand','newbrand3'],axis=1)
phone_data.columns=['deviceid','type','brand']
phone_data=phone_data.set_index('deviceid')
phone_data['brand']=phone_data['brand'].apply(str)
phone_data['type']=phone_data['brand'].apply(str)+'-'+phone_data['type'].apply(str)




#获取软件安装情况
deviceid_packages = pd.read_csv(data+'deviceid_packages.tsv',encoding='utf8', sep='\t',header=None)
deviceid_list=[]
packageid_list=[]

for i in range(len(deviceid_packages)):
    deviceid=deviceid_packages.iloc[i,0]
    pkgids=deviceid_packages.iloc[i,1]
    if not pkgids  is None :
        pkg_ids=pkgids.split(',')
        #print(pkg_ids)
        if(len(packageid_list)==0):
            packageid_list=pkg_ids
        else:
            packageid_list.extend(pkg_ids)            
        #print('packageid_list=',packageid_list)
        if(len(deviceid_list)==0):
            deviceid_list=[deviceid]*len(pkg_ids)
        else:
            deviceid_list.extend([deviceid]*len(pkg_ids))
deviceid_packages=pd.DataFrame({'deviceid':deviceid_list,'packageid':packageid_list},index=range(len(deviceid_list)))


#获取软件类型情况
package_label = pd.read_csv(data+'package_label.tsv',encoding='utf8', sep='\t',header=None)
package_label.columns=['packageid','major_class','sub_class']
package_label['sub_class']=package_label['major_class'].apply(str)+'-'+package_label['sub_class'].apply(str)


train['sex-age'].to_csv(output+'Ytrain.csv',header=True,index=False,encoding='utf8')


#获取有安装记录的app的类型信息
package_label = package_label[package_label.packageid.isin(deviceid_packages.packageid.unique())]


#获得每个设备中使用app的次数
reader = pd.read_csv(data+'deviceid_package_start_close.tsv',encoding='GBK', sep='\t',header=None,iterator=True)

loop = True
chunkSize = 5000000
chunks = []
count=0
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        #print(chunk.columns,chunk.index)
        chunks.append(chunk)
        count=count+chunkSize
        print("count=",count)
        #loop=False
    except StopIteration:
        loop = False
        print("Iteration is stopped.")
deviceid_package_usage = pd.concat(chunks, ignore_index=True)
deviceid_package_usage.columns=['deviceid','packageid','starttime','stoptime']


deviceid_package_usage['strdate'] = pd.to_datetime(deviceid_package_usage['starttime'],unit='ms').dt.strftime('%Y-%m-%d')


#获得各设备使用记录的天数
deviceid_usage_daycount=deviceid_package_usage.groupby(['deviceid','strdate'])['packageid'].agg(['size']).reset_index().groupby(['deviceid'])['strdate'].agg(['size']).reset_index()
deviceid_usage_daycount.columns=['deviceid','record_day_count']



#获得各设备中各软件的使用总次数
deviceid_package_usage_count = deviceid_package_usage.groupby(['deviceid','packageid'])['packageid'].agg(['size'])
deviceid_package_usage_count=((deviceid_package_usage_count.join(train['train_index'], how='left')).join(test['test_index'], how='left')).reset_index()
deviceid_package_usage_count.columns=['deviceid','packageid','pkg_click_count_total','train_index','test_index']
#获得各设备中各软件的使用日平均次数
deviceid_package_usage_count=deviceid_package_usage_count.merge(deviceid_usage_daycount,on='deviceid',how='left')
deviceid_package_usage_count['pkg_click_count_per_day']=deviceid_package_usage_count['pkg_click_count_total']/deviceid_package_usage_count['record_day_count']



#获得每个设备中各类型的使用次数和日均使用次数
deviceid_label_usage = deviceid_package_usage_count.merge(package_label,on='packageid')
deviceid_label_usage = deviceid_label_usage.groupby(['deviceid','sub_class'])['pkg_click_count_total','pkg_click_count_per_day'].agg(['sum'])
deviceid_label_usage = ((deviceid_label_usage.join(train['train_index'], how='left')).join(test['test_index'], how='left')).reset_index()
deviceid_label_usage.columns=['deviceid','sub_class','pkg_click_count_total','pkg_click_count_per_day','train_index','test_index']



#把package安装表格中 和 package类型表格中的 packageid ，majorclass,sub_class 都数字化

packageid_encoder = LabelEncoder().fit(deviceid_packages.packageid)
majorclass_encoder = LabelEncoder().fit(package_label.major_class)
sub_class_encoder = LabelEncoder().fit(package_label.sub_class)
phone_brand_encoder = LabelEncoder().fit(phone_data.brand)
phone_type_encoder = LabelEncoder().fit(phone_data.type)

deviceid_packages['packageid'] = packageid_encoder.transform(deviceid_packages['packageid'])
package_label['major_class'] = majorclass_encoder.transform(package_label['major_class'])
package_label['sub_class'] = sub_class_encoder.transform(package_label['sub_class'])
phone_data['brand']= phone_brand_encoder.transform(phone_data['brand'])
phone_data['type']= phone_type_encoder.transform(phone_data['type'])
deviceid_label_usage['sub_class'] = sub_class_encoder.transform(deviceid_label_usage['sub_class'])
deviceid_package_usage_count['packageid']=packageid_encoder.transform(deviceid_package_usage_count['packageid'])



#获得每个训练集和测试集设备的手机品牌信息，并形成训练集和测试集的关于手机品牌的稀疏矩阵

train['brand'] = phone_data['brand']
train['brand']=train['brand'].fillna(train['brand'].median())
test['brand'] = phone_data['brand']
test['brand']=test['brand'].fillna(test['brand'].median())

feature_train_phone_brand = csr_matrix((np.ones(train.shape[0]), (train.train_index, train.brand.apply(int))),shape=(train.shape[0], len(phone_brand_encoder.classes_)))
feature_test_phone_brand = csr_matrix((np.ones(test.shape[0]), (test.test_index, test.brand.apply(int))),shape=(test.shape[0], len(phone_brand_encoder.classes_)))



#获得每个训练集和测试集设备的手机型号信息，并形成训练集和测试集的关于手机品牌的稀疏矩阵
train['type'] = phone_data['type']
train['type']=train['type'].fillna(train['type'].median())
test['type'] = phone_data['type']
test['type']=test['type'].fillna(test['type'].median())

feature_train_phone_type = csr_matrix((np.ones(train.shape[0]), (train.train_index, train.type.apply(int))),shape=(train.shape[0], len(phone_type_encoder.classes_)))
feature_test_phone_type = csr_matrix((np.ones(test.shape[0]), (test.test_index, test.type.apply(int))),shape=(test.shape[0], len(phone_type_encoder.classes_)))


#形成每个设备的手机软件的BAG 稀疏矩阵。1：安装 0：未安装。和日均使用次数矩阵。 数量表示日均open次数

d = deviceid_package_usage_count.dropna(subset=['train_index'])
feature_train_package_usage = csr_matrix((np.ones(d.shape[0]), (d.train_index, d.packageid)), shape=(train.shape[0],len(packageid_encoder.classes_)))
feature_train_package_usage_per_day=csr_matrix((d['pkg_click_count_per_day'], (d.train_index, d.packageid)), shape=(train.shape[0],len(packageid_encoder.classes_)))

d = deviceid_package_usage_count.dropna(subset=['test_index'])
feature_test_package_usage = csr_matrix((np.ones(d.shape[0]), (d.test_index,d.packageid)), shape=(test.shape[0],len(packageid_encoder.classes_)))
feature_test_package_usage_per_day=csr_matrix((d['pkg_click_count_per_day'], (d.test_index, d.packageid)), shape=(test.shape[0],len(packageid_encoder.classes_)))



#形成每个设备的类型编号的BAG 稀疏矩阵。1：是 0：否。和每日类型编号的平均使用次数矩阵

d = deviceid_label_usage.dropna(subset=['train_index'])
feature_train_sub_class = csr_matrix((np.ones(d.shape[0]), (d.train_index,d.sub_class)), shape=(train.shape[0], len(sub_class_encoder.classes_)))
feature_train_sub_class_per_day = csr_matrix((d['pkg_click_count_per_day'], (d.train_index,d.sub_class)), shape=(train.shape[0], len(sub_class_encoder.classes_)))

d = deviceid_label_usage.dropna(subset=['test_index'])
feature_test_sub_class = csr_matrix((np.ones(d.shape[0]), (d.test_index,d.sub_class)),shape=(test.shape[0], len(sub_class_encoder.classes_)))
feature_test_sub_class_per_day = csr_matrix((d['pkg_click_count_per_day'], (d.test_index,d.sub_class)),shape=(test.shape[0], len(sub_class_encoder.classes_)))



#形成每个设备每个小时的点击量和日平均点击比例
deviceid_package_usage['hour'] = pd.to_datetime(deviceid_package_usage['starttime'],unit='ms').dt.hour
stampping = deviceid_package_usage.groupby(['deviceid','hour'])['starttime'].agg(['size'])
stampping = ((stampping.join(train['train_index'], how='left')).join(test['test_index'], how='left')).reset_index()
stampping.columns=['deviceid','hour','pkg_click_count_total_in_this_hour','train_index','test_index']
stampping_hour_total=stampping.groupby(['deviceid'])['pkg_click_count_total_in_this_hour'].agg(['sum']).reset_index()
stampping['total24hr']=stampping_hour_total['sum']
stampping['pkg_click_rate_total_in_this_hour']=stampping['pkg_click_count_total_in_this_hour']/stampping['total24hr']



#形成每个设备每个小时的点击量的稀疏矩阵。1：是 0：否。和点击比例的矩阵 
d = stampping.dropna(subset=['train_index'])
sparse_deviceid_package_usage_time_train = csr_matrix((np.ones(d.shape[0]), (d.train_index, d.hour)), shape=(train.shape[0], 24))
sparse_deviceid_package_usage_percent_time_train = csr_matrix((d['pkg_click_rate_total_in_this_hour'], (d.train_index, d.hour)), shape=(train.shape[0], 24))

d = stampping.dropna(subset=['test_index'])
sparse_deviceid_package_usage_time_test = csr_matrix((np.ones(d.shape[0]), (d.test_index,d.hour)), shape=(test.shape[0], 24))
sparse_deviceid_package_usage_percent_time_test = csr_matrix((d['pkg_click_rate_total_in_this_hour'], (d.test_index,d.hour)), shape=(test.shape[0], 24))



save_as_sparse(output + 'feature_train_phone_brand ',feature_train_phone_brand ) 
save_as_sparse(output + 'feature_test_phone_brand ',feature_test_phone_brand )
save_as_sparse(output + 'feature_train_phone_type ',feature_train_phone_type )
save_as_sparse(output + 'feature_test_phone_type ',feature_test_phone_type )

save_as_sparse(output + 'feature_train_package_usage ',feature_train_package_usage )
save_as_sparse(output + 'feature_test_package_usage ',feature_test_package_usage )
save_as_sparse(output + 'feature_train_package_usage_per_day ',feature_train_package_usage_per_day )
save_as_sparse(output + 'feature_test_package_usage_per_day ',feature_test_package_usage_per_day )

save_as_sparse(output + 'feature_train_sub_class ',feature_train_sub_class )
save_as_sparse(output + 'feature_test_sub_class ',feature_test_sub_class )
save_as_sparse(output + 'feature_train_sub_class_per_day ',feature_train_sub_class_per_day )
save_as_sparse(output + 'feature_test_sub_class_per_day ',feature_test_sub_class_per_day )

save_as_sparse(output + 'sparse_deviceid_package_usage_time_train ',sparse_deviceid_package_usage_time_train )
save_as_sparse(output + 'sparse_deviceid_package_usage_time_test ',sparse_deviceid_package_usage_time_test )
save_as_sparse(output + 'sparse_deviceid_package_usage_percent_time_train ',sparse_deviceid_package_usage_percent_time_train )
save_as_sparse(output + 'sparse_deviceid_package_usage_percent_time_test ',sparse_deviceid_package_usage_percent_time_test )


###
###
###
#  v1 版 使用 口袋模式数据
#  v2 版 使用日平均点击比例
#  v3 版 V1 和 v2的合成
#
###
###


#保存每个手机型号的 手机类型(使用1:0)，手机型号(使用1:0)，app启动信息(使用1:0)，app类型(使用1:0)，app使用时间分布(使用1:0)
Xtrain1 = hstack((feature_train_phone_type,feature_train_phone_brand,   feature_train_package_usage, feature_train_sub_class,sparse_deviceid_package_usage_time_train), format='csr')
Xtest1 =  hstack((feature_test_phone_type,feature_test_phone_brand,  feature_test_package_usage, feature_test_sub_class, sparse_deviceid_package_usage_time_test), format='csr')

save_as_sparse(output + 'feature_train_v1', Xtrain1)
save_as_sparse(output + 'feature_test_v1', Xtest1)



#保存每个手机型号的 手机类型，手机型号，app启动信息(使用日平均启动次数)，app类型(使用日平均启动次数)，app使用时间分布（使用小时分布比例）
Xtrain2 = hstack((feature_train_phone_type,feature_train_phone_brand,   feature_train_package_usage_per_day, feature_train_sub_class_per_day,sparse_deviceid_package_usage_percent_time_train), format='csr')
Xtest2 =  hstack((feature_test_phone_type, feature_test_phone_brand, feature_test_package_usage_per_day, feature_test_sub_class_per_day, sparse_deviceid_package_usage_percent_time_test), format='csr')

save_as_sparse(output + 'feature_train_v2', Xtrain2)
save_as_sparse(output + 'feature_test_v2', Xtest2)

#保存每个手机型号的 手机类型，手机型号，pp启动信息(使用1:0)，app启动信息(使用日平均启动次数)，app类型(使用1:0)，app类型(使用日平均启动次数)，app使用时间分布（使用小时分布比例），app使用时间分布(使用1:0)
Xtrain3 = hstack((feature_train_phone_type,feature_train_phone_brand,   feature_train_package_usage, feature_train_sub_class,  feature_train_package_usage_per_day, feature_train_sub_class_per_day,sparse_deviceid_package_usage_percent_time_train,sparse_deviceid_package_usage_time_train), format='csr')
Xtest3 =  hstack((feature_test_phone_type,feature_test_phone_brand, feature_test_package_usage, feature_test_sub_class,  feature_test_package_usage_per_day, feature_test_sub_class_per_day, sparse_deviceid_package_usage_percent_time_test,sparse_deviceid_package_usage_time_test), format='csr')

save_as_sparse(output + 'feature_train_v3', Xtrain3)
save_as_sparse(output + 'feature_test_v3', Xtest3)

print(Xtrain1.shape,Xtest1.shape,Xtrain2.shape,Xtest2.shape)

