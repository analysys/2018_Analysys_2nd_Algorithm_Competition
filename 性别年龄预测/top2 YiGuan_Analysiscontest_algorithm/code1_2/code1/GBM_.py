
# coding: utf-8

# # 数据分析
# 
# - 性别/年龄 与用户**手机品牌**的关系？
#     - 性别-品牌比例，偏好？
#     - 年龄-品牌趋势
#     - 出现日期/价格-年龄？
# 
# 
# - 性别/年龄 与用户**应用**的关系？
#     - 性别-应用的偏好？
#     - 应用LDA为何效果比较好？反应了静态模式？
#     - 如何利用应用类别信息？
#     - 如何提取特定可以反应类别信息的app？
# 
# 
# - 性别/年龄 与用户**行为**的关系？
#     - 用户的使用习惯？（时间、常用app）
#     - 用户使用app的方式？

# # 拆分数据集（方便并行操作）

# In[ ]:


import os, sys, codecs, gc, time

import pandas as pd
import numpy as np

import seaborn as sns
get_ipython().run_line_magic('pylab', 'inline')

deviceid_packages = pd.read_csv('../Demo/deviceid_packages.tsv',names=['id', 'app'], sep='\t')
deviceid_time = pd.read_csv('../Demo/deviceid_package_start_close.tsv',names=['id', 'app', 'start', 'end'], sep='\t')
deviceid_brand = pd.read_csv('../Demo/deviceid_brand.tsv',names=['id', 'brand', 'version'], sep='\t')
package_label = pd.read_csv('../Demo/package_label.tsv',names=['app', 'app_t1', 'app_t2'], sep='\t')

deviceid_train = pd.read_csv('../Demo/deviceid_train.tsv',names=['id', 'gender', 'age'], sep='\t')
deviceid_test = pd.read_csv('../Demo/deviceid_test.tsv',names=['id'], sep='\t')

deviceid_time = pd.merge(deviceid_time, package_label, on='app', how='left')
deviceid_time.sort_values(by=['id', 'start'], inplace=True)
df_index = deviceid_time['id'].value_counts().sort_index().reset_index()

for idx in df_index['index']:
    start = df_index[df_index['index'] < idx]['id'].sum()
    span = df_index[df_index['index'] == idx]['id'].values[0]
    
    tmp_df = deviceid_time.iloc[start :start+span]
    tmp_df.to_hdf('./time/{0}.hdf'.format(str(idx)), 'df')


# # 特征工程
# 
# - deviceid_brand: 每个设备的品牌和型号
#     - id
#     - brand: 品牌
#     - version: 型号
#     - [ ] 品牌编码、大众品牌、小众品牌
#     - [ ] 女性品牌
#     - [ ] 老人机、功能机
#     - [ ] 出产年份、价格
# 
# 
# - package_label: 每个应用的类别信息
#     - app
#     - app_t1:
#     - app_t2:
# 
# 
# - deviceid_packages: 每个设备上的应用安装列表
#     - id
#     - app
#     - [ ] 安装的app统计信息、个数和类别
#     - [ ] app的LDA信息
#     - [ ] app类别的LDA信息
# 
# 
# - deviceid_time: 每个设备上各个应用的打开、关闭行为数据
#     - id
#     - app
#     - start
#     - end
#     - [ ] 打开的小时、计数、weekday、使用时间
#     - [ ] 偏好应用统计
#     - [ ] app序列特征
#     - [ ] app的LADA特征

# In[204]:


#-*- encoding:utf-8 -*-
import os, sys, gc, codecs, shutil
import pandas as pd
import numpy as np
import lightgbm as lgb
get_ipython().run_line_magic('pylab', 'inline')

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import time
from datetime import datetime

from contextlib import contextmanager
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def read_input(datapath='../Demo/'):
    id_app = pd.read_csv(datapath+'deviceid_packages.tsv',names=['id','app'], sep='\t')
    id_time = pd.read_csv(datapath+'deviceid_package_start_close.tsv',names=['id','app','start','end'], sep='\t')
    id_brand = pd.read_csv(datapath+'deviceid_brand.tsv',names=['id', 'brand', 'version'], sep='\t')
    id_label = pd.read_csv(datapath+'package_label.tsv',names=['app', 'app_t1', 'app_t2'], sep='\t')

    id_train = pd.read_csv(datapath+'deviceid_train.tsv',names=['id', 'gender', 'age'], sep='\t')
    id_test = pd.read_csv(datapath+'deviceid_test.tsv',names=['id'], sep='\t')
    return id_app, id_time, id_brand, id_label, id_train, id_test

def brand_preocess(id_brand):
    replace_cols = ['Lenovo A850', 'Lenovo A320t', 'Lenovo A399',
           'Lenovo A858t', 'Lenovo S898t', 'Lenovo A3800-d', 'Lenovo A766',
           'Lenovo S856', 'Lenovo A3000-H', 'Lenovo S90-t', 'Lenovo A388t',
           'Lenovo A916', 'Lenovo A788t', 'Lenovo 2 A7-30TC', 'Lenovo S820',
           'Lenovo S850t', 'Lenovo K30-W', 'Lenovo S810t', 'Lenovo A630',
           'Lenovo A789', 'Lenovo A708t', 'Lenovo A806', 'Lenovo K30-T',
           'Lenovo A3600-d', 'Lenovo P770', 'Lenovo K900', 'Lenovo S720',
           'Lenovo A820t', 'Lenovo A6800', 'Lenovo S850', 'Lenovo A828t',
           'Lenovo A2800-d', 'Lenovo S968t', 'Lenovo A5800-D', 'Lenovo A889',
           'Lenovo A628t', 'Lenovo S60-t', 'Lenovo A936', 'Lenovo A816',
           'Lenovo Z2w', 'Lenovo X2-TO', 'Lenovo A808t', 'Lenovo P70-t',
           'Lenovo X2-CU', 'Lenovo A3900', 'Lenovo A688t', 'Lenovo S930',
           'Lenovo Z2', 'Lenovo A938t', 'Lenovo A3300-T', 'Lenovo S650',
           'Lenovo A805e', 'Lenovo K910', 'Lenovo A820', 'Lenovo B6000-H',
           'Lenovo S939', 'Lenovo A358t', 'Lenovo A880', 'Lenovo K860i',
           'Lenovo A7-60HC', 'Lenovo TAB 2 A10-70LC', 'Lenovo A760',
           'Lenovo+S5000-H', 'Lenovo A670t', 'Lenovo B8000-H',
           'Lenovo A3300-HV', 'Lenovo A5500-HV', 'Lenovo S90-u',
           'Lenovo S960', 'Lenovo S750', 'Lenovo+A3300-T', 'Lenovo A830',
           'Lenovo A360t', 'Lenovo A560', 'Lenovo A3500-HV', 'Lenovo A606',
           'Lenovo A398t', 'Lenovo S5000-H', 'Lenovo S938t', 'Lenovo S8-50LC',
           'Lenovo S658t']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'Lenovo')
        id_brand['brand'] = id_brand['brand'].replace(col, 'Lenovo')

    replace_cols = ['Coolpad 8675-A', 'Coolpad 8675-HD',
           'Coolpad 7620L-W00', 'Coolpad 8017', 'Coolpad 7105',
           'Coolpad 8297', 'Coolpad 7620L', 'Coolpad7295', 'Coolpad 7251',
           'Coolpad 8675-W00', 'Coolpad T2-W01', 'Coolpad 8297-T01',
           'Coolpad 8690', 'Coolpad 8297W', 'Coolpad 8675-FHD',
           'Coolpad 7296', 'Coolpad 8675', 'sk3-02', 'sk3-01', 'sk2-01']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'Coolpad')
        id_brand['brand'] = id_brand['brand'].replace(col, 'Coolpad')

    replace_cols = ['HUAWEI G750-T00', 'HUAWEI G610-U00', 'HUAWEI Y610-U00', 'HUAWEI',
           'HUAWEI MT1-U06', 'HUAWEI P6 S-U06', 'HUAWEI G750-T01',
           'HUAWEI G7-TL00', 'HUAWEI MT7-TL10', 'HUAWEI G7-UL20',
           'HUAWEI G660-L075', 'HUAWEI P7-L07', 'HUAWEI G610-T11',
           'HUAWEI P7-L09', 'HUAWEI SC-UL10', 'HUAWEI MT7-TL00',
           'HUAWEI P7-L00', 'HUAWEI G730-U00', 'HUAWEI MT7-CL00',
           'HUAWEI MT7-UL00', 'HUAWEI+G750-T01', 'HUAWEI G520-5000',
           'HUAWEI G610-T00', 'HUAWEI U8950D', 'HUAWEI G629-UL00',
           'HUAWEI P6-T00', 'HUAWEI+G7-TL00', 'HUAWEI G628-TL00',
           'HUAWEI G716-L070', 'HUAWEI G750-T20', 'HUAWEI G730-T00',
           'HUAWEI Y523-L176', 'HUAWEI U9508', 'HUAWEI G730-L075',
           'HUAWEI G521-L076', 'HUAWEI G700-U00', 'HUAWEI HN3-U01',
           'HUAWEI Y635-TL00', 'HUAWEI Y600-U00', 'HUAWEI Y300-0000',
           'HUAWEI D2-0082', 'HUAWEI G630-U00', 'HUAWEI MT2-L05',
           'HUAWEI T8950', 'HUAWEI Y321-C00', 'HUAWEI P6-U06',
           'HUAWEI MT2-L01', 'HUAWEI Y518-T00', 'HUAWEI+P7-L00',
           'HUAWEI G6-U00', 'HUAWEI ALE-CL00', 'HUAWEI G6-T00',
           'HUAWEI+MT2-L05',
           'che2-tl00m', 'che2-ul00', 'che2-tl00',
           'h30-u10', 'h30-t00', 'h300', 'h30-t10', 'honor h30-l02',
           'chm-tl00', 'chm-tl00h', 'chm-ul00', 'chm-cl00',
           'h60-l02', 'h60-l12', 'h60-l03', 'h60-l01', 'h60-l11',
           'g620s-ul00', 'g621-tl00', 'g621-tl00m', 'g620-l75',
           'pe-tl10', 'pe-ul00', 'pe-tl20', 'pe-tl00m',
           't1-701u', 't1-823l', 't1-a23l',
           'che1-cl20', 'che-tl00h', 's8-701u',
           'hol-u10', 'hol-t00']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'HUAWEI')
        id_brand['brand'] = id_brand['brand'].replace(col, 'HUAWEI')

    replace_cols = ['TCL P332U', 'TCL P335M', 'TCL M2U', 'TCL P331M',
           'TCL J928', 'TCL S838M', 'TCL M2M', 'TCL 302U']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'TCL')
        id_brand['brand'] = id_brand['brand'].replace(col, 'TCL')

    replace_cols = ['vivo Y27', 'vivo Y23L', 'vivo X3t', 'vivo Y613F',
           'vivo Y627', 'vivo Y13L', 'vivo X3L', 'vivo Y13iL', 'vivo X710L',
           'vivo Xplay', 'vivo X3F', 'vivo Xplay3S', 'vivo X5Max',
           'vivo Y19t', 'vivo X520F', 'vivo Y11iW', 'vivo Y18L', 'vivo Y20T',
           'vivo X5S L', 'vivo X1St', 'vivo Y613', 'vivo Y28L', 'vivo Y623',
           'vivo Y11', 'vivo X3S W', 'vivo X1', 'vivo S7i(t)', 'vivo Y17T',
           'vivo Y15T', 'vivo Y628', 'vivo Y13', 'vivo X510t', 'vivo X710F',
           'vivo X5Max S', 'vivo Y22L', 'vivo Y29L', 'vivo X5Max L',
           'vivo X5L', 'vivo Y17W', 'vivo Y22', 'vivo S6', 'vivo Y622',
           'vivo Y22iL', 'vivo S7t', 'vivo+X5S+L', 'vivo+Y613F', 'vivo Y923',
           'vivo Y11t', 'vivo E5', 'vivo S7']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'vivo')
        id_brand['brand'] = id_brand['brand'].replace(col, 'vivo')

    replace_cols = ['HTC D626w', 'HTC D816w', 'HTC D820us', 'HTC X920e', 'HTC D516t',
           'HTC 9088', 'HTC D816t', 'HTC M8Sw', 'HTC D820u', 'HTC D820mu',
           'HTC 802w', 'HTC D816', 'HTC D310w', 'HTC 8088', 'HTC D820ts',
           'HTC Butterfly', 'HTC One', 'HTC One_M8', 'HTC 802t',
           'honor']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'HTC')
        id_brand['brand'] = id_brand['brand'].replace(col, 'HTC')

    replace_cols = ['lephone T708', 'lephone W2']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'lephone')
        id_brand['brand'] = id_brand['brand'].replace(col, 'lephone')

    replace_cols = ['samsung t805s', 'samsung-sm-g900a', 'samsung*',
           'samsung-sm-n900a',
           'gt-n7108', 'gt-i9082i', 'gt-i9508', 'gt-i8552', 'gt-n7100',
           'gt-i8580', 'gt-i9300i', 'gt-n7102', 'gt-n5100', 'gt-i9308',
           'gt-i9082', 'gt-i9158v', 'gt-i9158', 'gt-i9507v', 'gt-i9300',
           'gt-i9500', 'gt-i9235', 'gt-i9168i', 'gt-i8558',
           'gt-i9152', 'gt-i9502', 'gt-i9152p', 'gt-i8262d', 'gt-i9100g',
           'gt-i9200', 'gt-s7568', 'gt-i9168', 'gt-i9128v', 'gt-i9118',
           'gt-i9508v', 'gt-i9082c', 'gt-i8268', 'gt-i9505', 'gt-i9158p',
           'sm-g7508q', 'sm-g5108q', 'sm-n9002', 'sm-p601', 'sm-n9006',
           'sm-g7200', 'sm-a5000', 'sm-g3818', 'sm-n9106w', 'sm-a7000',
           'sm-g7108v', 'sm-t321', 'sm-g3812', 'sm-g3556d', 'sm-n7506v',
           'sm-g9006v', 'sm-n9100', 'sm-g7106', 'sm-n900', 'sm-t231',
           'sm-g8508s', 'sm-n900l', 'sm-e7000', 'sm-g5308w', 'sm-g9006w',
           'sm-a700yd', 'sm-n9008v', 'sm-n7508v', 'sm-t705c', 'sm-n9005',
           'sm-n9008', 'sm-g3508', 'sm-g900f', 'sm-g9008w', 'sm-g900h',
           'sm-g5306w', 'sm-g7108', 'sm-t111', 'sm-g9008v', 'sm-g900l',
           'sm-n9150', 'sm-t311', 'sm-n910s', 'sm-g3586v',
           'shv-e210k', 'shv-e330s', 'shv-e300l', 'shv-e250s', 'shv-e210l',
           'shv-e250', 'sc-02f']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'samsung')
        id_brand['brand'] = id_brand['brand'].replace(col, 'samsung')

    replace_cols = ['mi 3', 'mi 4lte', 'mi+4lte', 'm1 note',
           'mi note lte', 'mi 3w', 'mi 4w', 'uimi4', 'a5-mi8',
           'mi 2s', 'mi 2', 'mi 2a', 'mi 3c', 'mi+note+lte',
           'mi+4w', 'mi 4c', 'mi+4c', 'mi 2sc', 'hm+note+1lte',
           'hm note 1lte', 'hm+2a', 'hm note 1s', 'hm note 1td',
           'hm 2a', 'hm note 1w', 'hm+note+1ltew',
           'hm+note+1w',
           'hm note 1ltetd', 'hm note 1ltew', 'hm 1sw', 'hm 1s',
           '2014813', '2014011', '2013023', '2014501', '2013022',
           '2014811', '2014112', '2014812']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'xiaomi')
        id_brand['brand'] = id_brand['brand'].replace(col, 'xiaomi')

    replace_cols = ['oppo r7', 'oppo r7s', 'oppo r7st', 'oppo r7t',
            'x9007', 'x909t', 'x9000', 'x903s',
            'r8200', 'r8207', 'r8107', 'r831s', 'r829t', 'r830', 'r8007',
           'r827t', 'r8205', 'r820', 'r830s',
           'n5207', 'n5117', 'n5209', '6607', '1107',
           'a31', 'a31u', 'a31t', 'a31c', 'u707t', 'r7c',
           'r2010', 'r2017', 'r2007', 'r12']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'oppo')
        id_brand['brand'] = id_brand['brand'].replace(col, 'oppo')

    replace_cols = ['zte q7', 'zte q705u', 'zte g717c', 'zte q503u',
           'zte g720t', 'v9180', 'n918st', 'n958st',
           'nx403a', 'nx507j', 'n918st', 'n958st']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'zte')
        id_brand['brand'] = id_brand['brand'].replace(col, 'zte')

    replace_cols = ['hisense h910', 'hisense e602m', 'hs-u980', 'hs-t950', 'hs-u980']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'hisense')
        id_brand['brand'] = id_brand['brand'].replace(col, 'hisense')

    replace_cols = ['htccn_chs_cmcc', 'htc_europe', 'htccn_chs', 'htc_asia_tw',
           'htccn_chs_cu', 'htc_asia_wwe', 'htl23']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'htc')
        id_brand['brand'] = id_brand['brand'].replace(col, 'htc')

    replace_cols = ['lg-d858', 'lg-d857', 'lg-ls980', 'lg-f400l']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'lg')
        id_brand['brand'] = id_brand['brand'].replace(col, 'lg')

    replace_cols = ['la2-s', 'la2-sn', 'la3-l', 'la2-t', 'la5-w',]
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'xiaolajiao')
        id_brand['brand'] = id_brand['brand'].replace(col, 'xiaolajiao')

    replace_cols = ['m351', 'm356', 'm355', 'm353', 'mx4']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'meizu')
        id_brand['brand'] = id_brand['brand'].replace(col, 'meizu')

    replace_cols = ['yq601', 'yq603', 'sm705', 'sm701',]
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'smartisan')
        id_brand['brand'] = id_brand['brand'].replace(col, 'smartisan')

    replace_cols = ['ereneben', 'eben t6', 'eben m2', 'eben t7', 'eben']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'ereneben')
        id_brand['brand'] = id_brand['brand'].replace(col, 'ereneben')

    replace_cols = ['asus_x002', 'asus_t00g', 'asus_t00f']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'asus')
        id_brand['brand'] = id_brand['brand'].replace(col, 'asus')

    replace_cols = ['amoi', 'amoi n821', 'amoi a920w',]
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'amoi')
        id_brand['brand'] = id_brand['brand'].replace(col, 'amoi')

    replace_cols = ['l35h', 'l36h', 'l39h', 'l50u', 'l55u', 'xm50h',
        's39h', 's55u', 'c6603', 'c630lw', 'c6903']
    for col in replace_cols:
        id_brand['brand'] = id_brand['brand'].replace(col.lower(), 'sony')
        id_brand['brand'] = id_brand['brand'].replace(col, 'sony')

    id_brand.loc[158, ['brand', 'version']] = ['Huawei', 'Y610-U00']
    id_brand.loc[379, ['brand', 'version']] = ['samsung', 'SM-N900V']
    id_brand.loc[466, ['brand', 'version']] = ['samsung', 'SM-G3518']
    id_brand.loc[483, ['brand', 'version']] = ['K-Touch', 'K-Touch S5']
    id_brand.loc[1000, ['brand', 'version']] = ['DOOV', 'L3C']
    id_brand.loc[1084, ['brand', 'version']] = ['samsung', 'GT-N7105']
    id_brand.loc[1243, ['brand', 'version']] = ['UMI', 'UMI R1']
    id_brand.loc[1330, ['brand', 'version']] = ['IUNI', 'IUNI i1']
    id_brand.loc[1559, ['brand', 'version']] = ['Lenovo', 'Lenovo A766']
    id_brand.loc[1701, ['brand', 'version']] = ['vivo', 'vivo Y627']
    id_brand.loc[1834, ['brand', 'version']] = ['samsung', 'SCL23']
    id_brand.loc[2475, ['brand', 'version']] = ['Changhong', 'A100']
    id_brand.loc[2575, ['brand', 'version']] = ['samsung', 'SM-N910P']
    id_brand.loc[3298, ['brand', 'version']] = ['samsung', 'GT-S7566']
    id_brand.loc[3363, ['brand', 'version']] = ['OPPO', 'N1W']
    id_brand.loc[3471, ['brand', 'version']] = ['OPPO', 'R6007']
    id_brand.loc[3660, ['brand', 'version']] = ['samsung', 'SHV-E400K']
    id_brand.loc[5075, ['brand', 'version']] = ['samsung', 'GT-S7572']
    id_brand.loc[5672, ['brand', 'version']] = ['Huawei', 'MT7-CL00']
    id_brand.loc[5871, ['brand', 'version']] = ['Nokia', 'N900']
    id_brand.loc[6782, ['brand', 'version']] = ['samsung', 'GT-S7278']
    id_brand.loc[7137, ['brand', 'version']] = ['motorola', 'XT910']
    id_brand.loc[7173, ['brand', 'version']] = ['samsung', 'GT-S7278U']
    id_brand.loc[7304, ['brand', 'version']] = ['Meitu', 'M4s']
    id_brand.loc[7609, ['brand', 'version']] = ['Coolpad', '9976D']
    id_brand.loc[7787, ['brand', 'version']] = ['Lenovo', 'Lenovo']
    id_brand.loc[8302, ['brand', 'version']] = ['Hasee', 'W960']
    id_brand.loc[8436, ['brand', 'version']] = ['Huawei', 'PE-CL00']
    id_brand.loc[8906, ['brand', 'version']] = ['Lenovo', 'A630']
    id_brand.loc[8911, ['brand', 'version']] = ['Lenovo', 'A789']
    id_brand.loc[8986, ['brand', 'version']] = ['samsung', 'GT-I9308I']
    id_brand.loc[9156, ['brand', 'version']] = ['Hisense', 'E622M']
    id_brand.loc[9295, ['brand', 'version']] = ['samsung', 'SHV-E250K']
    id_brand.loc[9640, ['brand', 'version']] = ['samsung', 'SM-G850S']
    id_brand.loc[9945, ['brand', 'version']] = ['Huawei', 'G520-5000']
    id_brand.loc[10370, ['brand', 'version']] = ['samsung', 'SM-N9200']
    id_brand.loc[10448, ['brand', 'version']] = ['Huawei', 'G610-T00']
    id_brand.loc[10661, ['brand', 'version']] = ['samsung', 'T805S']
    id_brand.loc[10702, ['brand', 'version']] = ['LG', 'F350L']
    id_brand.loc[10732, ['brand', 'version']] = ['HTC', 'X920e']
    id_brand.loc[11172, ['brand', 'version']] = ['Philips', 'T3500']
    id_brand.loc[11442, ['brand', 'version']] = ['asus', 'K01F']
    id_brand.loc[11711, ['brand', 'version']] = ['Huawei', 'U8950D']
    id_brand.loc[12029, ['brand', 'version']] = ['Philips', 'I999']
    id_brand.loc[12487, ['brand', 'version']] = ['Sony', 'D6708']
    id_brand.loc[12511, ['brand', 'version']] = ['samsung', 'GT-I9220']
    id_brand.loc[12954, ['brand', 'version']] = ['Lenovo', 'P770']
    id_brand.loc[13247, ['brand', 'version']] = ['LG', 'D802']
    id_brand.loc[14083, ['brand', 'version']] = ['Huawei', 'G7-TL00']
    id_brand.loc[14320, ['brand', 'version']] = ['zte', 'U819']
    id_brand.loc[14634, ['brand', 'version']] = ['LG', 'L22']
    id_brand.loc[14990, ['brand', 'version']] = ['Sony', 'C6602']
    id_brand.loc[15131, ['brand', 'version']] = ['TCL', 'P335M']
    id_brand.loc[15364, ['brand', 'version']] = ['HTC', '9088']
    id_brand.loc[16028, ['brand', 'version']] = ['samsung', 'GT-I9260']
    id_brand.loc[16273, ['brand', 'version']] = ['Lenovo', 'A6800']
    id_brand.loc[16750, ['brand', 'version']] = ['samsung', 'SM-G3558']
    id_brand.loc[16844, ['brand', 'version']] = ['ZTE', 'V5S']
    id_brand.loc[16849, ['brand', 'version']] = ['Lenovo', 'A828t']
    id_brand.loc[17208, ['brand', 'version']] = ['Coolpad', '9976T']
    id_brand.loc[17218, ['brand', 'version']] = ['GiONEE', 'GN9000']
    id_brand.loc[17221, ['brand', 'version']] = ['Meitu', 'M2']
    id_brand.loc[17255, ['brand', 'version']] = ['Hisense', 'X68T']
    id_brand.loc[17615, ['brand', 'version']] = ['Hisense', 'F5180']
    id_brand.loc[18669, ['brand', 'version']] = ['Hisense', 'T967']
    id_brand.loc[18718, ['brand', 'version']] = ['Haier', 'HT-X50']
    id_brand.loc[18770, ['brand', 'version']] = ['Lenovo', 'A5800-D']
    id_brand.loc[18910, ['brand', 'version']] = ['samsung', 'SM-G900V']
    id_brand.loc[19220, ['brand', 'version']] = ['GELI', 'G0111']
    id_brand.loc[19295, ['brand', 'version']] = ['samsung', 'GT-P3100']
    id_brand.loc[19725, ['brand', 'version']] = ['Lenovo', 'A628t']
    id_brand.loc[19832, ['brand', 'version']] = ['Lenovo', 'S60-t']
    id_brand.loc[20529, ['brand', 'version']] = ['Lenovo', 'A816']
    id_brand.loc[20661, ['brand', 'version']] = ['LEPHONE', 'T708']
    id_brand.loc[20905, ['brand', 'version']] = ['Huawei', 'Y523-L176']
    id_brand.loc[21191, ['brand', 'version']] = ['Coolpad', '8085Q']
    id_brand.loc[21360, ['brand', 'version']] = ['SANY', 'V8']
    id_brand.loc[21451, ['brand', 'version']] = ['Huawei', 'U9508']
    id_brand.loc[21661, ['brand', 'version']] = ['Bambook', 'S1']
    id_brand.loc[21697, ['brand', 'version']] = ['vivo', 'Y628']
    id_brand.loc[21775, ['brand', 'version']] = ['samsung', 'T950S']
    id_brand.loc[22080, ['brand', 'version']] = ['Coolpad', 'Y75']
    id_brand.loc[22920, ['brand', 'version']] = ['InFocus', 'M2']
    id_brand.loc[23306, ['brand', 'version']] = ['samsung', 'SM-N910F']
    id_brand.loc[23398, ['brand', 'version']] = ['ZTE', 'Q802T']
    id_brand.loc[23998, ['brand', 'version']] = ['Xiaomi', 'MI 1S']
    id_brand.loc[24039, ['brand', 'version']] = ['HTC', '802d']
    id_brand.loc[24157, ['brand', 'version']] = ['Huawei', 'G521-L076']
    id_brand.loc[24757, ['brand', 'version']] = ['Sony', 'D2533']
    id_brand.loc[24911, ['brand', 'version']] = ['samsung', 'A6-Plus']
    id_brand.loc[25015, ['brand', 'version']] = ['Lenovo', 'A3900']
    id_brand.loc[25247, ['brand', 'version']] = ['Philips', 'S388']
    id_brand.loc[26178, ['brand', 'version']] = ['samsung', 'SM-G3588V']
    id_brand.loc[26327, ['brand', 'version']] = ['Huawei', 'HN3-U01']
    id_brand.loc[26586, ['brand', 'version']] = ['samsung', 'GT-S7272C']
    id_brand.loc[26601, ['brand', 'version']] = ['Lenovo', 'S930']
    id_brand.loc[26674, ['brand', 'version']] = ['Lenovo', 'Z2']
    id_brand.loc[26807, ['brand', 'version']] = ['GiONEE', 'E3']
    id_brand.loc[27010, ['brand', 'version']] = ['TCL', 'P331M']
    id_brand.loc[27796, ['brand', 'version']] = ['Huawei', 'Y635-TL00']
    id_brand.loc[27822, ['brand', 'version']] = ['Lenovo', 'A938t']
    id_brand.loc[27949, ['brand', 'version']] = ['Huawei', 'Y600-U00']
    id_brand.loc[28301, ['brand', 'version']] = ['Sony', 'XM50t']
    id_brand.loc[28404, ['brand', 'version']] = ['ZTE', 'ZTE U969']
    id_brand.loc[28406, ['brand', 'version']] = ['Lenovo', 'S650']
    id_brand.loc[28425, ['brand', 'version']] = ['Mikee', 'M128-M29']
    id_brand.loc[28639, ['brand', 'version']] = ['Huawei', 'S7']
    id_brand.loc[28934, ['brand', 'version']] = ['CHANGHONG', 'Q8000']
    id_brand.loc[29324, ['brand', 'version']] = ['vivo', 'X5Max']
    id_brand.loc[29467, ['brand', 'version']] = ['Sony', 'XL39h']
    id_brand.loc[29504, ['brand', 'version']] = ['samsung', 'R850']
    id_brand.loc[29509, ['brand', 'version']] = ['OPPO', 'R833T']
    id_brand.loc[29567, ['brand', 'version']] = ['Lenovo', 'A805e']
    id_brand.loc[29649, ['brand', 'version']] = ['Xiaomi', 'HM+NOTE+1LTETD']
    id_brand.loc[30541, ['brand', 'version']] = ['samsung', 'SM-G900K']
    id_brand.loc[30633, ['brand', 'version']] = ['GiONEE', 'W800']
    id_brand.loc[30675, ['brand', 'version']] = ['ZTE', 'X9180']
    id_brand.loc[30920, ['brand', 'version']] = ['samsung', 'SM-T705']
    id_brand.loc[30989, ['brand', 'version']] = ['Huawei', 'Y300-0000']
    id_brand.loc[31633, ['brand', 'version']] = ['HTC', 'D816']
    id_brand.loc[31912, ['brand', 'version']] = ['Huawei', 'D2-0082']
    id_brand.loc[32444, ['brand', 'version']] = ['samsung', 'SHV-E250L']
    id_brand.loc[33576, ['brand', 'version']] = ['samsung', 'SM-C101']
    id_brand.loc[33971, ['brand', 'version']] = ['Lenovo', 'A368']
    id_brand.loc[33974, ['brand', 'version']] = ['GiONEE', 'E6T']
    id_brand.loc[34127, ['brand', 'version']] = ['samsung', '']
    id_brand.loc[34265, ['brand', 'version']] = ['samsung', 'SM-N9008S']
    id_brand.loc[34623, ['brand', 'version']] = ['samsung', 'SM-G906K']
    id_brand.loc[34816, ['brand', 'version']] = ['ZTE', 'V970']
    id_brand.loc[35264, ['brand', 'version']] = ['Huawei', 'CHE-TL00']
    id_brand.loc[35472, ['brand', 'version']] = ['Haier', 'HL-Y500']
    id_brand.loc[36122, ['brand', 'version']] = ['samsung', 'SM-G7105']
    id_brand.loc[36221, ['brand', 'version']] = ['Coolpad', '7295C']
    id_brand.loc[36565, ['brand', 'version']] = ['samsung', 'SM-G3508I']
    id_brand.loc[36580, ['brand', 'version']] = ['Lenovo', 'B6000-H']
    id_brand.loc[36589, ['brand', 'version']] = ['TCL', 'J928']
    id_brand.loc[36648, ['brand', 'version']] = ['Nibiru', 'H1c']
    id_brand.loc[36806, ['brand', 'version']] = ['smartisan', 'YQ605']
    id_brand.loc[36864, ['brand', 'version']] = ['samsung', 'Galaxy Nexus']
    id_brand.loc[36999, ['brand', 'version']] = ['Coolpad', '8297']
    id_brand.loc[37021, ['brand', 'version']] = ['Lenovo', 'S939']
    id_brand.loc[37480, ['brand', 'version']] = ['OPPO', 'N5206']
    id_brand.loc[37499, ['brand', 'version']] = ['OPPO', 'R811']
    id_brand.loc[37650, ['brand', 'version']] = ['samsung', 'SM-N9007']
    id_brand.loc[37679, ['brand', 'version']] = ['samsung', 'SM-G5108']
    id_brand.loc[37863, ['brand', 'version']] = ['Xiaomi', 'm1 note']
    id_brand.loc[37890, ['brand', 'version']] = ['xiaolajiao', 'HLJ-XL']
    id_brand.loc[38473, ['brand', 'version']] = ['ZTE', 'U968']
    id_brand.loc[38513, ['brand', 'version']] = ['Lenovo', 'A358t']
    id_brand.loc[39178, ['brand', 'version']] = ['OPPO', 'R809T']
    id_brand.loc[39457, ['brand', 'version']] = ['Sony', 'C6916']
    id_brand.loc[39475, ['brand', 'version']] = ['samsung', 'SHV-E270K']
    id_brand.loc[39620, ['brand', 'version']] = ['Coolpad', '8297-W01']
    id_brand.loc[40145, ['brand', 'version']] = ['Lenovo', 'K860i']
    id_brand.loc[41016, ['brand', 'version']] = ['Coolpad', '7320']
    id_brand.loc[41088, ['brand', 'version']] = ['vivo', 'Y22']
    id_brand.loc[42428, ['brand', 'version']] = ['samsung', 'Galaxy Series 8']
    id_brand.loc[42462, ['brand', 'version']] = ['Sony', 'D6653']
    id_brand.loc[42620, ['brand', 'version']] = ['vivo', 'S6']
    id_brand.loc[42739, ['brand', 'version']] = ['Coolpad', 'SK1-02']
    id_brand.loc[42995, ['brand', 'version']] = ['OPPO', 'R7005']
    id_brand.loc[43167, ['brand', 'version']] = ['HTC', 'D310w']
    id_brand.loc[43178, ['brand', 'version']] = ['samsung', 'SM-N900S']
    id_brand.loc[43222, ['brand', 'version']] = ['HTC', '8088']
    id_brand.loc[43296, ['brand', 'version']] = ['ONEPLUS', 'A0001']
    id_brand.loc[43320, ['brand', 'version']] = ['ZZBAO', 'Z5S']
    id_brand.loc[43558, ['brand', 'version']] = ['samsung', 'GT-I9205']
    id_brand.loc[43561, ['brand', 'version']] = ['samsung', 'SM-G3502']
    id_brand.loc[43569, ['brand', 'version']] = ['samsung', 'GT-P6200']
    id_brand.loc[43606, ['brand', 'version']] = ['xiaolajiao', 'LA2-W']
    id_brand.loc[44050, ['brand', 'version']] = ['Lenovo', 'A10-70LC']
    id_brand.loc[44163, ['brand', 'version']] = ['samsung', 'SM-G900L/S/K']
    id_brand.loc[44467, ['brand', 'version']] = ['Huawei', 'T8950']
    id_brand.loc[44545, ['brand', 'version']] = ['motorola', 'MT788']
    id_brand.loc[45373, ['brand', 'version']] = ['Lenovo', 'S5000H']
    id_brand.loc[45386, ['brand', 'version']] = ['Lenovo', 'A670t']
    id_brand.loc[45464, ['brand', 'version']] = ['Coolpad', '7061']
    id_brand.loc[45867, ['brand', 'version']] = ['Meizu', 'MX4 Pro']
    id_brand.loc[46118, ['brand', 'version']] = ['vivo', 'S7t']
    id_brand.loc[47003, ['brand', 'version']] = ['Hisense', 'HS-T958']
    id_brand.loc[47225, ['brand', 'version']] = ['ZTE', 'U5']
    id_brand.loc[47235, ['brand', 'version']] = ['TCL', 'S838M']
    id_brand.loc[47285, ['brand', 'version']] = ['Coolpad', '8670']
    id_brand.loc[47540, ['brand', 'version']] = ['samsung', 'SM-G3568V']
    id_brand.loc[47633, ['brand', 'version']] = ['OPPO', 'R821T']
    id_brand.loc[48037, ['brand', 'version']] = ['Huawei', 'MT2-L01']
    id_brand.loc[48672, ['brand', 'version']] = ['motorola', 'ME865']
    id_brand.loc[48810, ['brand', 'version']] = ['Xiaomi', 'HM 1STD']
    id_brand.loc[49084, ['brand', 'version']] = ['OPPO', 'R5']
    id_brand.loc[49396, ['brand', 'version']] = ['HTC', 'HTC One']
    id_brand.loc[49551, ['brand', 'version']] = ['Lenovo', 'A3300-HV']
    id_brand.loc[49845, ['brand', 'version']] = ['OPPO', 'R831T']
    id_brand.loc[50110, ['brand', 'version']] = ['samsung', 'GT-I9128']
    id_brand.loc[50501, ['brand', 'version']] = ['Lenovo', 'S2005A-H']
    id_brand.loc[51134, ['brand', 'version']] = ['Sony', 'M51w']
    id_brand.loc[51182, ['brand', 'version']] = ['samsung', 'GT-I9301I']
    id_brand.loc[51309, ['brand', 'version']] = ['OPPO', 'T9']
    id_brand.loc[51511, ['brand', 'version']] = ['HTC', 'HTC Butterfly']
    id_brand.loc[51528, ['brand', 'version']] = ['samsung', 'GT-I9208']
    id_brand.loc[51610, ['brand', 'version']] = ['Coolpad', '7270']
    id_brand.loc[51614, ['brand', 'version']] = ['Philips', 'W6618']
    id_brand.loc[51629, ['brand', 'version']] = ['Huawei', 'Y518-T00']
    id_brand.loc[51952, ['brand', 'version']] = ['GiONEE', 'GN137']
    id_brand.loc[52267, ['brand', 'version']] = ['samsung', 'SM-G3606']
    id_brand.loc[52328, ['brand', 'version']] = ['GiONEE', 'E7']
    id_brand.loc[52603, ['brand', 'version']] = ['HTC', 'HTC S720t 16GB']
    id_brand.loc[52617, ['brand', 'version']] = ['HTC', 'HTC One']
    id_brand.loc[52683, ['brand', 'version']] = ['Meizu', 'M045']
    id_brand.loc[53050, ['brand', 'version']] = ['Coolpad', '7295']
    id_brand.loc[53126, ['brand', 'version']] = ['Coolpad', '8720L']
    id_brand.loc[53333, ['brand', 'version']] = ['HTC', 'HTC6435LVW']
    id_brand.loc[53538, ['brand', 'version']] = ['Lenovo', 'S90-u']
    id_brand.loc[54509, ['brand', 'version']] = ['K-Touch', 'T85']
    id_brand.loc[54854, ['brand', 'version']] = ['Lenovo', 'S960']
    id_brand.loc[54855, ['brand', 'version']] = ['OPPO', 'X909']
    id_brand.loc[55177, ['brand', 'version']] = ['samsung', 'SM-G3502U']
    id_brand.loc[55231, ['brand', 'version']] = ['Coolpad', '7296']
    id_brand.loc[55498, ['brand', 'version']] = ['HTC', 'HTC One']
    id_brand.loc[55518, ['brand', 'version']] = ['samsung', 'SHV-E330L']
    id_brand.loc[55830, ['brand', 'version']] = ['samsung', 'SL600']
    id_brand.loc[55941, ['brand', 'version']] = ['Meitu', 'MK150']
    id_brand.loc[55950, ['brand', 'version']] = ['Coolpad', '8730L']
    id_brand.loc[56222, ['brand', 'version']] = ['samsung', 'SM-T805C']
    id_brand.loc[56293, ['brand', 'version']] = ['Lenovo', 'S750']
    id_brand.loc[56573, ['brand', 'version']] = ['samsung', 'SM-G900']
    id_brand.loc[56702, ['brand', 'version']] = ['Coolpad', '9190']
    id_brand.loc[57045, ['brand', 'version']] = ['K-Touch', 'W88']
    id_brand.loc[57538, ['brand', 'version']] = ['vivo', 'X5S']
    id_brand.loc[57630, ['brand', 'version']] = ['samsung', 'SM-G900A']
    id_brand.loc[57858, ['brand', 'version']] = ['Coolpad', '8675-HD']
    id_brand.loc[58690, ['brand', 'version']] = ['samsung', 'GT-I9305']
    id_brand.loc[58840, ['brand', 'version']] = ['ZTE', 'U960s3']
    id_brand.loc[58965, ['brand', 'version']] = ['samsung', 'N9106']
    id_brand.loc[59058, ['brand', 'version']] = ['Sony', 'SO-04E']
    id_brand.loc[59773, ['brand', 'version']] = ['samsung', 'SM-G3502I']
    id_brand.loc[59859, ['brand', 'version']] = ['ZTE', 'Q505T']
    id_brand.loc[60144, ['brand', 'version']] = ['Huawei', 'P7-L00']
    id_brand.loc[60522, ['brand', 'version']] = ['K-Touch', 'T619']
    id_brand.loc[60528, ['brand', 'version']] = ['K-Touch', 'Tou ch 1']
    id_brand.loc[60546, ['brand', 'version']] = ['Lenovo', 'A3300-T']
    id_brand.loc[60561, ['brand', 'version']] = ['ZTE', 'Grand S II LTE']
    id_brand.loc[61098, ['brand', 'version']] = ['samsung', 'I9300']
    id_brand.loc[61107, ['brand', 'version']] = ['samsung', 'SM-G900W8']
    id_brand.loc[61123, ['brand', 'version']] = ['samsung', 'GT-I9268']
    id_brand.loc[61185, ['brand', 'version']] = ['Lenovo', 'A830']
    id_brand.loc[61467, ['brand', 'version']] = ['samsung', 'GT-S7562i']
    id_brand.loc[62134, ['brand', 'version']] = ['Haier', 'HL-Y1']
    id_brand.loc[62263, ['brand', 'version']] = ['Huawei', 'G6-U00']
    id_brand.loc[62330, ['brand', 'version']] = ['DOOV', 'D330']
    id_brand.loc[62573, ['brand', 'version']] = ['Huawei', 'ALE-CL00']
    id_brand.loc[62675, ['brand', 'version']] = ['Coolpad', '8085N']
    id_brand.loc[62761, ['brand', 'version']] = ['vivo', 'vivo']
    id_brand.loc[62858, ['brand', 'version']] = ['Hisense', 'D1-M']
    id_brand.loc[63029, ['brand', 'version']] = ['samsung', 'SM-G3608']
    id_brand.loc[63174, ['brand', 'version']] = ['Lenovo', 'A560']
    id_brand.loc[63597, ['brand', 'version']] = ['vivo', 'Y613F']
    id_brand.loc[63788, ['brand', 'version']] = ['IdeaTab', 'S6000-H']
    id_brand.loc[63799, ['brand', 'version']] = ['IdeaTab', 'A1010-T']
    id_brand.loc[63819, ['brand', 'version']] = ['HASEE', 'H45 T3']
    id_brand.loc[64160, ['brand', 'version']] = ['samsung', 'SHV-E160L']
    id_brand.loc[64191, ['brand', 'version']] = ['Coolpad', '7298A']
    id_brand.loc[64351, ['brand', 'version']] = ['vivo', 'Y923']
    id_brand.loc[65254, ['brand', 'version']] = ['Lenovo', ' A3500-HV']
    id_brand.loc[65298, ['brand', 'version']] = ['Lenovo', 'A606']
    id_brand.loc[65368, ['brand', 'version']] = ['nubia', 'NX505H']
    id_brand.loc[65622, ['brand', 'version']] = ['vivo', 'Y11t']
    id_brand.loc[65966, ['brand', 'version']] = ['Lenovo', '5000-H']
    id_brand.loc[66105, ['brand', 'version']] = ['DOOV', 'L1']
    id_brand.loc[66340, ['brand', 'version']] = ['Lenovo', 'S938t']
    id_brand.loc[66669, ['brand', 'version']] = ['samsung', 'GT-S7898']
    id_brand.loc[66696, ['brand', 'version']] = ['OPPO', 'R823T']
    id_brand.loc[67165, ['brand', 'version']] = ['Huawei', 'T8620']
    id_brand.loc[67188, ['brand', 'version']] = ['Xiaomi', 'HM+1S']
    id_brand.loc[67329, ['brand', 'version']] = ['vivo', 'E5']
    id_brand.loc[67375, ['brand', 'version']] = ['samsung', 'GT-I8190N']
    id_brand.loc[67927, ['brand', 'version']] = ['vivo', 'S7']
    id_brand.loc[68069, ['brand', 'version']] = ['OPPO', 'Find7']
    id_brand.loc[68692, ['brand', 'version']] = ['Lenovo', 'S8-50LC']
    id_brand.loc[68855, ['brand', 'version']] = ['HTC', 'HTC One_M8']
    id_brand.loc[68883, ['brand', 'version']] = ['samsung', 'SM-N915S']
    id_brand.loc[69441, ['brand', 'version']] = ['Coolpad', '8705']
    id_brand.loc[70031, ['brand', 'version']] = ['samsung', 'SHV-E300K']
    id_brand.loc[70130, ['brand', 'version']] = ['Huawei', 'MT2-L05']
    id_brand.loc[70751, ['brand', 'version']] = ['samsung', 'GT-S7562C']
    id_brand.loc[70782, ['brand', 'version']] = ['samsung', '-SM-N900A']
    id_brand.loc[70842, ['brand', 'version']] = ['Lenovo', 'S658t']
    id_brand.loc[71496, ['brand', 'version']] = ['HTC', '802t']

    id_brand['brand'] = id_brand['brand'].replace('GT-N7100', 'samsung')
    id_brand['brand'] = id_brand['brand'].replace('SM-N9006', 'samsung')
    id_brand['brand'] = id_brand['brand'].replace('H60-L02', 'Huawei')
    id_brand['brand'] = id_brand['brand'].replace('CHM-UL00', 'Huawei')
    id_brand['brand'] = id_brand['brand'].replace('PE-UL00', 'Huawei')
    id_brand['brand'] = id_brand['brand'].replace('Che2-UL00', 'Huawei')
    id_brand['brand'] = id_brand['brand'].replace('che-tl00h', 'Huawei')
    id_brand['brand'] = id_brand['brand'].replace('Che1-CL20', 'Huawei')
    id_brand['brand'] = id_brand['brand'].replace('PE-TL10', 'Huawei')
    id_brand['brand'] = id_brand['brand'].replace('HM NOTE 1LTE', 'Xiaomi')
    id_brand['brand'] = id_brand['brand'].replace('HM NOTE 1S', 'Xiaomi')
    id_brand['brand'] = id_brand['brand'].replace('MI 4LTE', 'Xiaomi')
    id_brand['brand'] = id_brand['brand'].replace('MI NOTE LTE', 'Xiaomi')
    id_brand['brand'] = id_brand['brand'].replace('HM NOTE 1LTETD', 'Xiaomi')
    id_brand['brand'] = id_brand['brand'].replace('Coolpad 8675', 'Coolpad')
    id_brand['brand'] = id_brand['brand'].replace('doov t20l', 'doov')
    id_brand['brand'] = id_brand['brand'].replace('philips v989', 'philips')
    id_brand['brand'] = id_brand['brand'].replace('meitu m4', 'meitu')
    id_brand['brand'] = id_brand['brand'].replace('meitu2', 'meitu')
    id_brand['brand'] = id_brand['brand'].replace('moto', 'motorola')
    
    id_brand['brand'] = id_brand['brand'].apply(lambda x: str(x).lower())
    return id_brand

print('-' * 80)
print('Reading Input')
print('-' * 80)
# with timer('Reading Input'):
#    id_app, id_time, id_brand, app_label, id_train, id_test = read_input()

# 映射label
lbl = []
for gender in [1, 2]:
    for age in range(11):
        lbl.append(str(gender) + '-' + str(age))
lbl_dict = {k: v for v, k in enumerate(lbl)}

id_train['label'] = id_train['gender'].astype(str) + '-' + id_train['age'].astype(str)
id_train['label'] = id_train['label'].apply(lambda x: lbl_dict[x])
    
print('-' * 80)
print('Data prepocess')
print('-' * 80)
with timer('brand process'):
    id_brand = brand_preocess(id_brand)

print('-' * 80)
print('Feature Engineering')
print('-' * 80)

# app_label = {
#     'game': ['ACT(动作类游戏)', 'RPG(角色扮演游戏)', 'SIM(模拟游戏)', 'SLG(策略游戏)'
#                 'TAB(桌面游戏)', '休闲', '卡牌', '射击', '竞速',
#                 '游戏媒体', '游戏工具', '游戏平台'],
#     'traffic': ['交通出行', '汽车', '地图导航'],
#     'business': ['企业级应用', '商务办公'],
#     'sports': ['体育', '体育竞技', '健康'],
#     'fun': ['视频', '视频直播', '音频娱乐'],
# }

train_test = pd.concat([id_train[['id']], id_test[['id']]], axis=0, ignore_index=True)


# In[136]:


def LDA_feature(df, col, prefix, ntopics=5):
    '''提取列取值的LDA特征
    
    Args:
        df:
        col:
        prefix:
        ntopics:
    
    Returns:
        
    '''
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()

    cntTf = vectorizer.fit_transform(df[col])
    tfidf = transformer.fit_transform(cntTf)
    word = vectorizer.get_feature_names()

    weight = tfidf.toarray()
    df_weight = pd.DataFrame(weight)

    feature = df_weight.columns
    df_weight['sum'] = 0

    for f in feature:
        df_weight['sum'] += df_weight[f]
    df['tfidf_sum'] = df_weight['sum']

    lda = LatentDirichletAllocation(n_topics=ntopics,
                                    learning_offset=50., n_jobs=20,
                                    random_state=666)
    docres = lda.fit_transform(cntTf)

    docres = pd.DataFrame(docres)
    docres.columns = [prefix + str(x) for x in range(ntopics)]
    df = pd.concat([df['id'], pd.DataFrame(docres)], axis=1)
    return df


# In[137]:


id_app['app'] = id_app['app'].apply(lambda x: x.split(','))
id_app['app_lenghth'] = id_app['app'].apply(lambda x:len(x))
id_app['app'] = id_app['app'].apply(lambda x: ' '.join(x))

feat_app_lda = LDA_feature(id_app[['id', 'app']], 'app', 'app_lda', 5)


# In[139]:


import time
def ctime2time(x):
    return time.strftime('%Y/%m/%d %H:%M:%S',  time.gmtime(x/1000.))

# 半夜、早上、中午、晚上
def hour_map(x):
    if x >= 22 or x < 6:
        return 0
    elif x >= 6 and x < 12:
        return 1
    elif x >= 12 and x < 18:
        return 2
    elif x >= 18 and x < 22:
        return 3
hour_map = np.vectorize(hour_map)

def time_feat(id):
    df = pd.read_hdf('./time/' + id + '.hdf', 'df')
    df.sort_values(by='start', inplace=True)
    
    df['start'] = df['start'].apply(ctime2time)
    df['end'] = df['end'].apply(ctime2time)

    # TODO 剔除异常的时间记录
    df['start_day'] = df['start'].apply(lambda x: x[:10])
    df['start_hour'] = df['start'].apply(lambda x: x[11:13])
    df['start_minute'] = df['start'].apply(lambda x: x[11:16])
    df['start_time'] = df['start'].apply(lambda x: x[:16])
    
    df['end_day'] = df['end'].apply(lambda x: x[:10])
    df['end_hour'] = df['end'].apply(lambda x: x[11:13])
    df['end_minute'] = df['end'].apply(lambda x: x[11:16])
    df['end_time'] = df['end'].apply(lambda x: x[:16])
    
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df['span'] = df['end'] - df['start']
    df['totaltime'] = df['span'].apply(lambda x: x.total_seconds())
    
    feat_arr = [id]
    
    # 统计时间信息
    feat_arr.append(df['start_minute'].nunique())
    feat_arr.append(df['start_hour'].nunique())
    
    # 使用小时编码
    df['start_hour'] = df['start_hour'].astype(int)
    df_hour = np.zeros(24, dtype=int)
    df_hour[df['start_hour'].unique() - 1] = 1
    feat_arr += list(df_hour)
    
    # 时间小时段编码
    df_hour = np.zeros(4, dtype=int)
    df_hour[hour_map(df['start_hour'].unique() - 1)] = 1
    feat_arr += list(df_hour)
    
    # app信息统计
    app_arr = [df['app'].nunique(), df['app_t1'].nunique(), df['app_t2'].nunique()]
    app_arr += [float(x) / df['start_day'].nunique() for x in app_arr[:3]]
    app_arr += [float(x) / df['start_minute'].nunique() for x in app_arr[:3]]
    app_arr += [float(x) / df['start_hour'].nunique() for x in app_arr[:3]]
    feat_arr += app_arr
    
    return feat_arr


# In[140]:


from sklearn.externals.joblib import Parallel, delayed
get_ipython().run_line_magic('time', "feat_time = Parallel(n_jobs=40)(delayed(time_feat)(id) for id in train_test['id'].values[:])")

feat_time = pd.DataFrame(feat_time)
feat_time.columns = ['id'] + ['time' + str(i) for i in range(1, feat_time.shape[1])]


# In[298]:


train_test = pd.concat([id_train[['id']], id_test[['id']]], axis=0, ignore_index=True)

train_test = pd.merge(train_test, feat_app_lda, on='id', how='left')
# train_test = pd.merge(train_test, feat_app_lda2, on='id', how='left')
train_test = pd.merge(train_test, feat_app_lda3, on='id', how='left')
train_test = pd.merge(train_test, feat_time, on='id', how='left')
# train_test = pd.merge(train_test, feat_app_type, on='id', how='left')
# train_test = pd.merge(train_test, feat_app_tfidf, on='id', how='left')


# In[299]:


train = train_test.iloc[:id_train.shape[0], :]
test = train_test.iloc[id_train.shape[0]:, :]


# In[300]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

skf = StratifiedKFold(n_splits = 8, random_state = 1, shuffle = True)

params = {
    'learning_rate': 0.01,
    'min_child_samples': 10,
    'max_depth': 8, 
    'lambda_l1':2,
    'boosting': 'gbdt', 
    'objective': 'multiclass', 
    'num_class': 22,
    'metric': 'multi_logloss',
    'feature_fraction': .95,
    'bagging_fraction': .75,
    'seed': 99,
    'num_threads': 40,
    'verbose': 0
}

train_loss, val_loss = [], []
train_pred = np.zeros((id_train.shape[0], 22))
test_pred = np.zeros((id_test.shape[0], 22))

test_x = test.drop(['id'], axis=1)

for i, (train_idx, val_idx) in enumerate(skf.split(id_train['label'], id_train['label'].values)):
    tr_x, val_x = train.drop(['id'], axis=1).iloc[train_idx], train.drop(['id'], axis=1).iloc[val_idx]
    tr_y, val_y = id_train.iloc[train_idx]['label'].values, id_train.iloc[val_idx]['label'].values
    
    print(tr_x.shape, val_x.shape)
    clf = lgb.train(params, lgb.Dataset(tr_x, label=tr_y), 4000,
                    valid_sets=[lgb.Dataset(tr_x, label=tr_y), lgb.Dataset(val_x, label=val_y)],
                    verbose_eval=100, 
                    early_stopping_rounds=200)
    
    tr_pred = clf.predict(tr_x, num_iteration=clf.best_iteration)
    val_pred = clf.predict(val_x, num_iteration=clf.best_iteration)
    
    train_pred[val_idx, :] = clf.predict(val_x, num_iteration=clf.best_iteration)
#     test_pred += clf.predict(test_df.drop(['file_id', 'api', 'api_unique'], axis=1), num_iteration=clf.best_iteration)
    test_pred += clf.predict(test_x, num_iteration=clf.best_iteration)
    
    print("Fold: {0} {1:5f}/{2:5f}".format(i+1, log_loss(tr_y, tr_pred), log_loss(val_y, val_pred)))
    print(classification_report(val_y, np.argmax(val_pred, 1), digits=7))
    print('')
    
    
    train_loss.append(log_loss(tr_y, tr_pred))
    val_loss.append(log_loss(val_y, val_pred))

test_pred /= 8

print(train_loss, val_loss)
print('Train{0:5f}_Test{1:5f}'.format(np.mean(train_loss), np.mean(val_loss)))
print(classification_report(id_train['label'].values, np.argmax(train_pred, 1), digits=7))


# In[301]:


test_pred = np.hstack([test['id'].values.reshape(22727, 1), test_pred])
test_pred = pd.DataFrame(test_pred, columns=['DeviceID'] + lbl)

train_pred = np.hstack([train['id'].values.reshape(50000, 1), train_pred])
train_pred = pd.DataFrame(train_pred, columns=['DeviceID'] + lbl)


# In[302]:


test_pred.to_csv('./GBM2735_test.csv', index=None)
train_pred.to_csv('./GBM2735_train.csv', index=None)

