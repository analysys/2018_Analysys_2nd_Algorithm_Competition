使用到开源工具如下:

scikit-learn              0.19.2
lightgbm                  2.1.2    
xgboost                   0.80    
tensorflow                1.11.0     
Keras                     2.2.2 
pandas                    0.20.3     

环境:

Ubuntu : 16.04 
pyhotn : 3.5.3


执行顺序:

1. device_applist.ipynb    

设备数据特征提取,针对性别年龄进行10折交叉验证训练,使用stacking算法生成新的数据集,
用于第二层训练,同时保存提取的原始特征

2. app_label_1.ipynb       

应用的类别数据,对性别年龄进行10折交叉验证训练,使用stacking生成新的数据集,同时保存原始特征

3. brand_1.ipynb           

设备品牌信息的简单清洗,保存在features文件夹下

4. brand_2.ipynb           

提取特征, 对性别年龄进行10折交叉验证训练,使用stacking算法生成新的数据集,同时保存原始特征

5. behavior_1.ipynb        

应用数据,提取两组特征, 分别进行10折的交叉验证训练,使用stacking算法生成新的数据集,同时保留原始特征

6. behavior_2.ipynb        

应用数据中的时间特征,24个时间点,首先是对性别年龄22分类,进行10折的交叉验证训练,stacking生成新的训练数据.
然对分别对年龄和性别两个子任务进行,10折的交叉验证训练.stacking生成新的训练数据.

7. lgbcnt.ipynb

使用全部的原始特征,对年和性别进行10折的交叉验证训练,stacking生成新的训练数据.

8. lgbtfidf.ipynb

也是使用全部的特征,先对性别年龄(22分类任务进行10折的交叉验证训练),stacking生成新的特征数据,
然后, 仅用设备的品牌特征进行,性别和年龄,这两个子任务的10折交叉验证训练.stacking生成新的数据.

9. DNN_keras_2.ipynb

最后,使用以上交叉训练和stacking,得到的新特征数据进行,10折的交叉验证训练,生成最终的结果,在submit文件夹下
