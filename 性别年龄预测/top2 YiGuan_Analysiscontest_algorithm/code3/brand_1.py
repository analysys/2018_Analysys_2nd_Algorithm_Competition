
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from help_function import LoadData


# 机型数据：每个设备的品牌和型号【deviceid_brand.tsv】

# In[2]:


device_brand = pd.read_csv('../Demo/deviceid_brand.tsv',sep='\t',
                           names=['device_id','brand','model'])


# ### clean brand

# In[3]:


device_brand.brand = device_brand.brand.apply(lambda x:x.split(' ')[0])

# huawei
device_brand.brand.replace({'HUAWEI':'Huawei','huawei':'Huawei',u'华为':'Huawei',
                            'MediaPad':'Huawei'},inplace=True)
device_brand.brand.replace(regex=r'^HUAWE.*$',value='Huawei',inplace=True)

# htc
device_brand.brand.replace(regex={'HTC':'htc',r'^htc.*$':'htc'},inplace=True)
device_brand.brand.replace({'HTL23':'htc'},inplace=True)

# iphone
device_brand.brand.replace({'Apple':'iphone','X-apple':'iphone','iPhone':'iphone',
                            'APPLE':'iphone','6s':'iphone','6plus':'iphone'},inplace=True)

# samsung
device_brand.brand.replace(regex=[r'^SM.*$',r'^GT.*$',r'^SC.*$',r'^SHV.*$',r'^A[6-9].*$',
                                  r'^SAMSUN.*$'],value='samsung',inplace=True)
device_brand.brand.replace({'Samsung':'samsung','Galaxy':'samsung',u'三星':'samsung',
                            'I9300':'samsung','J5':'samsung','SL600':'samsung'},inplace=True)

# Xiaomi ,HM
device_brand.brand.replace(regex=[r'^MI.*$',r'^UIMI.*$',r'^m[1-9].*$'],value='Xiaomi',inplace=True)
device_brand.brand.replace(regex=[r'^HM.*$',r'^2014.*$',r'^2013.*$'],value='HM',inplace=True)
device_brand.brand.replace({'A5-MI8':'Xiaomi'},inplace=True)

# Honor
device_brand.brand.replace(regex=[r'^PE.*$',r'^CHM.*$',r'^H60.*$',r'^H30.*$',r'^Hol.*$',
                                  r'^Che.*$',r'^CHE.*$',r'^G6.*$',r'^T1.*$'],value='Honor',
                           inplace=True)
device_brand.brand.replace({'HONOR':'Honor','S8-701u':'Honor'},inplace=True)

# Verizon
device_brand.brand.replace({'VERIZON':'Verizon'},inplace=True)
device_brand.brand.replace(regex=r'^verizo.*$',value='Verizon',inplace=True)

# ZTE
device_brand.brand.replace({'zte':'ZTE','N918St':'ZTE','N958St':'ZTE'},inplace=True)

# OPPO
device_brand.brand.replace(regex=[r'^A1.*$',r'^A3.*$',r'^R[2|5|6|7|8].*$',r'^T9',
                                  r'^X9.*$',r'^N5.*$',r'^110.$',r'^300[0-9]',r'^660.*$'],
                           value='OPPO',inplace=True)
device_brand.brand.replace({'N1T':'OPPO','N1W':'OPPO'},inplace=True)

# nubia
device_brand.brand.replace(regex=r'^NX.*$',value='nubia',inplace=True)

# Bird
device_brand.brand.replace({'BIRD':'Bird'},inplace=True)

# Meizu
device_brand.brand.replace(regex=[r'^MX[0-9].*$',r'^M[0-9][0-9][0-9].*$'],value='Meizu',
                          inplace=True)

# motorala
device_brand.brand.replace({'MOTO':'motorola','MT788':'motorola','Moto':'motorola'},inplace=True)

# Hisense
device_brand.brand.replace({'hisense':'Hisense','F5180':'Hisense'},inplace=True)
device_brand.brand.replace(regex=[r'^HS-T.*$',r'^HS-U.*$',r'^HS-X.*$'],value='Hisense',inplace=True)

# xiaolajiao
device_brand.brand.replace(regex=[r'^LA.*$'], value='xiaolajiao',inplace=True)
device_brand.brand.replace({'HLJ-XL':'xiaolajiao'},inplace=True)

# Nokia
device_brand.brand.replace({'Nokia_XL':'Nokia','N900':'Nokia'},inplace=True)

# ONEPLUS
device_brand.brand.replace({'oneplus':'ONEPLUS','A0001':'ONEPLUS'},inplace=True)

# Coolpad
device_brand.brand.replace(regex=[r'^Coolpad.*$'],value='Coolpad',inplace=True)
device_brand.brand.replace({'8085N':'Coolpad','8295':'Coolpad'},inplace=True)

# vivo
device_brand.brand.replace(regex=[r'^viv.*$',r'^VIV.*$'],value='vivo',inplace=True)

device_brand.brand.replace(regex=r'^4G.*$',value='4G',inplace=True)

device_brand.brand.replace({'100Cw':'100jia'},inplace=True)

# asus
device_brand.brand.replace(regex=r'^ASUS.*$',value='asus',inplace=True)

device_brand.brand.replace(regex=r'^BOWA.*$',value='BOWAY',inplace=True)

device_brand.brand.replace(regex=r'^Bamboo.*$',value='Bambook',inplace=True)

# bifer
device_brand.brand.replace(regex=r'^BF.*$',value='bifer',inplace=True)


# ### clean model

# In[4]:


device_brand.model = device_brand.model.astype('str')
device_brand.model = device_brand.model.apply(lambda x:x.split(' ')[-1])
device_brand.model = device_brand.model.apply(lambda x:x.split('.')[0])


# In[5]:


device_brand['btype'] = device_brand.brand+'_'+device_brand.model


# In[6]:


device_brand.to_csv('new_feature/device_brand.csv',index=False)

