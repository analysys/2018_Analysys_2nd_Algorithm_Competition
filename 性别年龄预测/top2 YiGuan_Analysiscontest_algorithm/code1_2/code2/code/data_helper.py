import pandas as pd
import time
from tqdm import tqdm
import fasttext
import os
def trans_time(timeStamp):
	timeArray = time.localtime(int(timeStamp)/1000)
	return timeArray

path='../data/Demo/'

sex_dict={'1':0,'2':1}
age_dict={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10}
label_dict={'1-0':0,'1-1':1,'1-2':2,'1-3':3,'1-4':4,'1-5':5,'1-6':6,'1-7':7,'1-8':8,'1-9':9,'1-10':10,'2-0':11, '2-1':12, '2-2':13, '2-3':14, '2-4':15, '2-5':16, '2-6':17, '2-7':18, '2-8':19, '2-9':20, '2-10':21}
def create_data_list():
	deviceid_package_start_close=pd.read_csv(path+'deviceid_package_start_close.tsv',sep='\t',names=['device_id','app','start_time','close_time'])
	deviceid_package_start_close['use_time_long']=deviceid_package_start_close['close_time'].map(int)-deviceid_package_start_close['start_time'].map(int)
	deviceid_package_start_close['use_time_long']=deviceid_package_start_close['use_time_long'].map(str)
	deviceid_package_use_time=deviceid_package_start_close.groupby(by='device_id').apply(lambda x:' '.join(x.use_time_long))
	deviceid_package_use_time=deviceid_package_use_time.reset_index()
	deviceid_package_use_time.columns=['device_id','times']
	
	deviceid_package_app=deviceid_package_start_close.groupby(by='device_id').apply(lambda x:' '.join(x.app))
	deviceid_package_app=deviceid_package_app.reset_index()
	deviceid_package_app.columns=['device_id','apps']
	
	deviceid_packages=pd.read_csv(path+'deviceid_packages.tsv',sep='\t',names=['device_id','install_apps'])
	deviceid_packages['install_apps']=deviceid_packages['install_apps'].apply(lambda x:x.replace(',',' '))
	
	deviceid_test=pd.read_csv(path+'deviceid_test.tsv',sep='\t',names=['device_id'])
	deviceid_train=pd.read_csv(path+'deviceid_train.tsv',sep='\t',names=['device_id','sex','age'])
	deviceid_train['sex-age']=deviceid_train['sex'].map(str)+'-'+deviceid_train['age'].map(str)
	deviceid_train['sex-age']=deviceid_train['sex-age'].apply(lambda x:label_dict[x])
	deviceid_train['sex']=deviceid_train['sex'].apply(lambda x:sex_dict[str(x)])
	deviceid_train['age']=deviceid_train['age'].apply(lambda x:age_dict[str(x)])
	device_brand=pd.read_csv(path+'deviceid_brand.tsv',sep='\t',names=['device_id','P1','P2'])
	
	deviceid_train=pd.merge(deviceid_train,deviceid_packages,on='device_id',how='left')
	deviceid_train=pd.merge(deviceid_train,deviceid_package_app,on='device_id',how='left')
	deviceid_train=pd.merge(deviceid_train,device_brand,on='device_id',how='left')
	deviceid_train=pd.merge(deviceid_train,deviceid_package_use_time,on='device_id',how='left')
	
	deviceid_test=pd.merge(deviceid_test,deviceid_packages,on='device_id',how='left')
	deviceid_test=pd.merge(deviceid_test,deviceid_package_app,on='device_id',how='left')
	deviceid_test=pd.merge(deviceid_test,device_brand,on='device_id',how='left')
	deviceid_test=pd.merge(deviceid_test,deviceid_package_use_time,on='device_id',how='left')
	
	deviceid_train[['sex-age']].to_csv('../data/train/label.csv',index=None)
	deviceid_train[['sex']].to_csv('../data/train/sex.csv',index=None)
	deviceid_train[['age']].to_csv('../data/train/age.csv',index=None)
	
	deviceid_train[['install_apps']].to_csv('../data/train/install_apps.csv',index=None)
	deviceid_train[['P1']].to_csv('../data/train/P1.csv',index=None)
	deviceid_train[['P2']].to_csv('../data/train/P2.csv',index=None)
	deviceid_train[['apps']].to_csv('../data/train/apps.csv',index=None)
	deviceid_train[['times']].to_csv('../data/train/use_times.csv',index=None)
	
	deviceid_test[['install_apps']].to_csv('../data/test/install_apps.csv',index=None)
	deviceid_test[['P1']].to_csv('../data/test/P1.csv',index=None)
	deviceid_test[['P2']].to_csv('../data/test/P2.csv',index=None)
	deviceid_test[['apps']].to_csv('../data/test/apps.csv',index=None)
	deviceid_test[['times']].to_csv('../data/test/use_times.csv',index=None)

def apps_label():
	app_dict={}
	with open('../data/Demo/package_label.tsv') as file1:
		for ii in file1:
			app_dict[ii.strip().split('\t')[0]]=[ii.strip().split('\t')[1],ii.strip().split('\t')[2]]
	
	use_apps=open('../data/train/use_apps_1.csv','w+')
	apps=open('../data/train/apps.csv')
	num=0
	for ii in apps:
		num+=1
		if num==1:
			use_apps.write('app1\n')
			use_apps.flush()
		else:
			line=''
			for jj in ii.strip().split(' '):
				try:
					line+=app_dict[jj][0]+' '
				except:
					line+='None '
			use_apps.write(line.strip()+'\n')
			use_apps.flush()
	
	use_apps=open('../data/test/use_apps_1.csv','w+')
	apps=open('../data/test/apps.csv')
	num=0
	for ii in apps:
		num+=1
		if num==1:
			use_apps.write('app1\n')
			use_apps.flush()
		else:
			line=''
			for jj in ii.strip().split(' '):
				try:
					line+=app_dict[jj][0]+' '
				except:
					line+='None '
			use_apps.write(line.strip()+'\n')
			use_apps.flush()
	
	
	install_apps=open('../data/train/install_apps_1.csv','w+')
	apps=open('../data/train/install_apps.csv')
	num=0
	for ii in apps:
		num+=1
		if num==1:
			install_apps.write('install_app1\n')
			install_apps.flush()
		else:
			line=''
			for jj in ii.strip().split(' '):
				try:
					line+=app_dict[jj][0]+' '
				except:
					line+='None '
			install_apps.write(line.strip()+'\n')
			install_apps.flush()
	
	install_apps=open('../data/test/install_apps_1.csv','w+')
	apps=open('../data/test/install_apps.csv')
	num=0
	for ii in apps:
		num+=1
		if num==1:
			install_apps.write('install_app1\n')
			install_apps.flush()
		else:
			line=''
			for jj in ii.strip().split(' '):
				try:
					line+=app_dict[jj][0]+' '
				except:
					line+='None '
			install_apps.write(line.strip()+'\n')
			install_apps.flush()

def time_fea():
	deviceid_package_start_close=pd.read_csv(path+'deviceid_package_start_close.tsv',sep='\t',names=['device_id','app','start_time','close_time'])
	
	deviceid_package_start_close['start_time']=deviceid_package_start_close['start_time'].map(str)
	deviceid_package_start_time=deviceid_package_start_close.groupby(by='device_id').apply(lambda x:' '.join(x.start_time))
	deviceid_package_start_time=deviceid_package_start_time.reset_index()
	deviceid_package_start_time.columns=['device_id','start_times']
	
	deviceid_test=pd.read_csv(path+'deviceid_test.tsv',sep='\t',names=['device_id'])
	deviceid_train=pd.read_csv(path+'deviceid_train.tsv',sep='\t',names=['device_id','sex','age'])
	
	deviceid_train_=pd.merge(deviceid_train,deviceid_package_start_time,on='device_id',how='left')
	deviceid_train_[['start_times']].to_csv('../data/analyse/start_time_train.csv',index=None)
	
	deviceid_test_=pd.merge(deviceid_test,deviceid_package_start_time,on='device_id',how='left')
	deviceid_test_[['start_times']].to_csv('../data/analyse/start_time_test.csv',index=None)
	
	
	deviceid_package_start_close['close_time']=deviceid_package_start_close['close_time'].map(str)
	deviceid_package_clost_time=deviceid_package_start_close.groupby(by='device_id').apply(lambda x:' '.join(x.close_time))
	deviceid_package_clost_time=deviceid_package_clost_time.reset_index()
	deviceid_package_clost_time.columns=['device_id','close_times']
	
	deviceid_train_=pd.merge(deviceid_train,deviceid_package_clost_time,on='device_id',how='left')
	deviceid_train_[['close_times']].to_csv('../data/analyse/close_time_train.csv',index=None)
	
	deviceid_test_=pd.merge(deviceid_test,deviceid_package_clost_time,on='device_id',how='left')
	deviceid_test_[['close_times']].to_csv('../data/analyse/close_time_test.csv',index=None)

def week_time():
	week_list=['w0','w1','w2','w3','w4','w5','w6']
	hours_normal1_list=['h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23']
	hours_normal3_list=['h0-3','h3-6','h6-9','h9-12','h12-15','h15-18','h18-21','h21-24']
	
	aa=open('../data/train/start_time_fea.csv','w+')
	aa.write(','.join(week_list+hours_normal_list+hours_normal3_list)+'\n')
	aa.flush()
	with open('../data/analyse/start_time_train.csv') as file1:
		num=0
		for ii in file1:
			num+=1
			if num==1:
				continue
			week_dict={'w0':0,'w1':0,'w2':0,'w3':0,'w4':0,'w5':0,'w6':0}
			hours_normal1_dict={'h0':0,'h1':0,'h2':0,'h3':0,'h4':0,'h5':0,'h6':0,'h7':0,'h8':0,'h9':0,'h10':0,'h11':0,'h12':0,'h13':0,'h14':0,'h15':0,'h16':0,'h17':0,'h18':0,'h19':0,'h20':0,'h21':0,'h22':0,'h23':0}
			hours_normal3_list={'h0-3':0,'h3-6':0,'h6-9':0,'h9-12':0,'h12-15':0,'h15-18':0,'h18-21':0,'h21-24':0}
			times=[trans_time(x) for x in ii.strip().split(' ')]
			for j_ in times:
				if j_[1]==3 and j_[0]==2017:
					week_dict['w'+str(j_[6])]+=1
					hours_normal1_dict['h'+str(j_[3])]+=1
				else:
					continue
			hours_normal3_list['h0-3']=hours_normal1_dict['h0']+hours_normal1_dict['h1']+hours_normal1_dict['h2']
			hours_normal3_list['h3-6']=hours_normal1_dict['h3']+hours_normal1_dict['h4']+hours_normal1_dict['h5']
			hours_normal3_list['h6-9']=hours_normal1_dict['h6']+hours_normal1_dict['h7']+hours_normal1_dict['h8']
			hours_normal3_list['h9-12']=hours_normal1_dict['h9']+hours_normal1_dict['h10']+hours_normal1_dict['h11']
			hours_normal3_list['h12-15']=hours_normal1_dict['h12']+hours_normal1_dict['h13']+hours_normal1_dict['h14']
			hours_normal3_list['h15-18']=hours_normal1_dict['h15']+hours_normal1_dict['h16']+hours_normal1_dict['h17']
			hours_normal3_list['h18-21']=hours_normal1_dict['h18']+hours_normal1_dict['h19']+hours_normal1_dict['h20']
			hours_normal3_list['h21-24']=hours_normal1_dict['h21']+hours_normal1_dict['h22']+hours_normal1_dict['h23']
			ans=''
			for i in week_list:
				ans+=str(week_dict[i])+','
			for i in hours_normal_list:
				ans+=str(hours_normal1_dict[i])+','
			for i in hours_normal3_list:
				ans+=str(hours_normal3_list[i])+','
			aa.write(ans[:-1]+'\n')
			aa.flush()

	bb=open('../data/test/start_time_fea.csv','w+')
	bb.write(','.join(week_list+hours_normal_list+hours_normal3_list)+'\n')
	bb.flush()
	with open('../data/analyse/start_time_test.csv') as file1:
		num=0
		for ii in file1:
			num+=1
			if num==1:
				continue
			week_dict={'w0':0,'w1':0,'w2':0,'w3':0,'w4':0,'w5':0,'w6':0}
			hours_normal1_dict={'h0':0,'h1':0,'h2':0,'h3':0,'h4':0,'h5':0,'h6':0,'h7':0,'h8':0,'h9':0,'h10':0,'h11':0,'h12':0,'h13':0,'h14':0,'h15':0,'h16':0,'h17':0,'h18':0,'h19':0,'h20':0,'h21':0,'h22':0,'h23':0}
			hours_normal3_list={'h0-3':0,'h3-6':0,'h6-9':0,'h9-12':0,'h12-15':0,'h15-18':0,'h18-21':0,'h21-24':0}
			times=[trans_time(x) for x in ii.strip().split(' ')]
			for j_ in times:
				if j_[1]==3 and j_[0]==2017:
					week_dict['w'+str(j_[6])]+=1
					hours_normal1_dict['h'+str(j_[3])]+=1
				else:
					continue
			hours_normal3_list['h0-3']=hours_normal1_dict['h0']+hours_normal1_dict['h1']+hours_normal1_dict['h2']
			hours_normal3_list['h3-6']=hours_normal1_dict['h3']+hours_normal1_dict['h4']+hours_normal1_dict['h5']
			hours_normal3_list['h6-9']=hours_normal1_dict['h6']+hours_normal1_dict['h7']+hours_normal1_dict['h8']
			hours_normal3_list['h9-12']=hours_normal1_dict['h9']+hours_normal1_dict['h10']+hours_normal1_dict['h11']
			hours_normal3_list['h12-15']=hours_normal1_dict['h12']+hours_normal1_dict['h13']+hours_normal1_dict['h14']
			hours_normal3_list['h15-18']=hours_normal1_dict['h15']+hours_normal1_dict['h16']+hours_normal1_dict['h17']
			hours_normal3_list['h18-21']=hours_normal1_dict['h18']+hours_normal1_dict['h19']+hours_normal1_dict['h20']
			hours_normal3_list['h21-24']=hours_normal1_dict['h21']+hours_normal1_dict['h22']+hours_normal1_dict['h23']
			ans=''
			for i in week_list:
				ans+=str(week_dict[i])+','
			for i in hours_normal_list:
				ans+=str(hours_normal1_dict[i])+','
			for i in hours_normal3_list:
				ans+=str(hours_normal3_list[i])+','
			bb.write(ans[:-1]+'\n')
			bb.flush()

	cc=open('../data/train/close_time_fea.csv','w+')
	cc.write(','.join(week_list+hours_normal_list+hours_normal3_list)+'\n')
	cc.flush()
	with open('../data/analyse/close_time_train.csv') as file1:
		num=0
		for ii in file1:
			num+=1
			if num==1:
				continue
			week_dict={'w0':0,'w1':0,'w2':0,'w3':0,'w4':0,'w5':0,'w6':0}
			hours_normal1_dict={'h0':0,'h1':0,'h2':0,'h3':0,'h4':0,'h5':0,'h6':0,'h7':0,'h8':0,'h9':0,'h10':0,'h11':0,'h12':0,'h13':0,'h14':0,'h15':0,'h16':0,'h17':0,'h18':0,'h19':0,'h20':0,'h21':0,'h22':0,'h23':0}
			hours_normal3_list={'h0-3':0,'h3-6':0,'h6-9':0,'h9-12':0,'h12-15':0,'h15-18':0,'h18-21':0,'h21-24':0}
			times=[trans_time(x) for x in ii.strip().split(' ')]
			for j_ in times:
				if j_[1]==3 and j_[0]==2017:
					week_dict['w'+str(j_[6])]+=1
					hours_normal1_dict['h'+str(j_[3])]+=1
				else:
					continue
			hours_normal3_list['h0-3']=hours_normal1_dict['h0']+hours_normal1_dict['h1']+hours_normal1_dict['h2']
			hours_normal3_list['h3-6']=hours_normal1_dict['h3']+hours_normal1_dict['h4']+hours_normal1_dict['h5']
			hours_normal3_list['h6-9']=hours_normal1_dict['h6']+hours_normal1_dict['h7']+hours_normal1_dict['h8']
			hours_normal3_list['h9-12']=hours_normal1_dict['h9']+hours_normal1_dict['h10']+hours_normal1_dict['h11']
			hours_normal3_list['h12-15']=hours_normal1_dict['h12']+hours_normal1_dict['h13']+hours_normal1_dict['h14']
			hours_normal3_list['h15-18']=hours_normal1_dict['h15']+hours_normal1_dict['h16']+hours_normal1_dict['h17']
			hours_normal3_list['h18-21']=hours_normal1_dict['h18']+hours_normal1_dict['h19']+hours_normal1_dict['h20']
			hours_normal3_list['h21-24']=hours_normal1_dict['h21']+hours_normal1_dict['h22']+hours_normal1_dict['h23']
			ans=''
			for i in week_list:
				ans+=str(week_dict[i])+','
			for i in hours_normal_list:
				ans+=str(hours_normal1_dict[i])+','
			for i in hours_normal3_list:
				ans+=str(hours_normal3_list[i])+','
			cc.write(ans[:-1]+'\n')
			cc.flush()

	dd=open('../data/test/close_time_fea.csv','w+')
	dd.write(','.join(week_list+hours_normal_list+hours_normal3_list)+'\n')
	dd.flush()
	with open('../data/analyse/close_time_test.csv') as file1:
		num=0
		for ii in file1:
			num+=1
			if num==1:
				continue
			week_dict={'w0':0,'w1':0,'w2':0,'w3':0,'w4':0,'w5':0,'w6':0}
			hours_normal1_dict={'h0':0,'h1':0,'h2':0,'h3':0,'h4':0,'h5':0,'h6':0,'h7':0,'h8':0,'h9':0,'h10':0,'h11':0,'h12':0,'h13':0,'h14':0,'h15':0,'h16':0,'h17':0,'h18':0,'h19':0,'h20':0,'h21':0,'h22':0,'h23':0}
			hours_normal3_list={'h0-3':0,'h3-6':0,'h6-9':0,'h9-12':0,'h12-15':0,'h15-18':0,'h18-21':0,'h21-24':0}
			times=[trans_time(x) for x in ii.strip().split(' ')]
			for j_ in times:
				if j_[1]==3 and j_[0]==2017:
					week_dict['w'+str(j_[6])]+=1
					hours_normal1_dict['h'+str(j_[3])]+=1
				else:
					continue
			hours_normal3_list['h0-3']=hours_normal1_dict['h0']+hours_normal1_dict['h1']+hours_normal1_dict['h2']
			hours_normal3_list['h3-6']=hours_normal1_dict['h3']+hours_normal1_dict['h4']+hours_normal1_dict['h5']
			hours_normal3_list['h6-9']=hours_normal1_dict['h6']+hours_normal1_dict['h7']+hours_normal1_dict['h8']
			hours_normal3_list['h9-12']=hours_normal1_dict['h9']+hours_normal1_dict['h10']+hours_normal1_dict['h11']
			hours_normal3_list['h12-15']=hours_normal1_dict['h12']+hours_normal1_dict['h13']+hours_normal1_dict['h14']
			hours_normal3_list['h15-18']=hours_normal1_dict['h15']+hours_normal1_dict['h16']+hours_normal1_dict['h17']
			hours_normal3_list['h18-21']=hours_normal1_dict['h18']+hours_normal1_dict['h19']+hours_normal1_dict['h20']
			hours_normal3_list['h21-24']=hours_normal1_dict['h21']+hours_normal1_dict['h22']+hours_normal1_dict['h23']
			ans=''
			for i in week_list:
				ans+=str(week_dict[i])+','
			for i in hours_normal_list:
				ans+=str(hours_normal1_dict[i])+','
			for i in hours_normal3_list:
				ans+=str(hours_normal3_list[i])+','
			dd.write(ans[:-1]+'\n')
			dd.flush()

def make_app_list():
	deviceid_package_start_close=pd.read_csv(path+'deviceid_package_start_close.tsv',sep='\t',names=['device_id','app','start_time','close_time'])
	deviceid_test=pd.read_csv(path+'deviceid_test.tsv',sep='\t',names=['device_id'])
	deviceid_train=pd.read_csv(path+'deviceid_train.tsv',sep='\t',names=['device_id','sex','age'])
	
	aa=open('../data/train/app_list.csv','w+')
	for k in tqdm(deviceid_train['device_id']):
		text=' '.join(deviceid_package_start_close[deviceid_package_start_close['device_id']==k].sort_values(by="start_time")['app'])
		aa.write(text+'\n')
		aa.flush()
	bb=open('../data/test/app_list.csv','w+')
	for k in tqdm(deviceid_test['device_id']):
		text=' '.join(deviceid_package_start_close[deviceid_package_start_close['device_id']==k].sort_values(by="start_time")['app'])
		bb.write(text+'\n')
		bb.flush()

def get_w2v():
	in_path='../data/app_list.csv'
	model = fasttext.skipgram(in_path, '../w2v/fast_300_model',dim=300)

if __name__=="__main__":
	create_data_list()
	apps_label()
	time_fea()
	week_time()
	make_app_list()
	os.system('cat ../data/train/app_list.csv >> ../data/app_list.csv')
	os.system('cat ../data/test/app_list.csv >> ../data/app_list.csv')
	get_w2v()