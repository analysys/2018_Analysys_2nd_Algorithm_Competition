import pandas as pd
import numpy as np
from tqdm import tqdm

path='../data/'

tr_install_app=open(path+'train/install_apps.csv')
te_install_app=open(path+'test/install_apps.csv')
tr_apps=open(path+'train/apps.csv')
te_apps=open(path+'test/apps.csv')

out_col = ['app_len','use_app_len']
output_file1 = open(path+'train/apps_base.csv','w+')
output_file2 = open(path+'test/apps_base.csv','w+')

output_file1.write(','.join(out_col)+'\n')
output_file1.flush()
output_file2.write(','.join(out_col)+'\n')
output_file2.flush()
num=0
for ii,jj in tqdm(zip(tr_install_app,tr_apps)):
	num+=1
	if num==1:
		continue
	apps_list=ii.strip().split(' ')
	app_use_list=jj.strip().split(' ')
	apps_set=set(apps_list)
	output_file1.write(str(len(apps_set))+','+str(len(app_use_list))+'\n')
	output_file1.flush()

num=0
for ii,jj in tqdm(zip(te_install_app,te_apps)):
	num+=1
	if num==1:
		continue
	apps_list=ii.strip().split(' ')
	app_use_list=jj.strip().split(' ')
	apps_set=set(apps_list)
	output_file2.write(str(len(apps_set))+','+str(len(app_use_list))+'\n')
	output_file2.flush()