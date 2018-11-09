单模型2575：
	cd code;python data_helper.py
	#得到分割数据和训练的词向量
	python nn_1.py 
	#在stack目录下得到meta_nn1_test.csv 单模型线上可达0.2575
其他：
	cd code
	python lda_feature.py
	python make_app_use_stat.py
	python make_base.py
	python make_tfidf.py
	python one_hot.py
	python lgb_v1.py
	python nn_2.py
	python nn_5.py
	#得到简单变体的模型结果
	运行code1 里代码得到模型结果分别将模型生成的元特征和队友的原特征根据stacking_1.py里的配置路径保存
	运行stacking_1.py得到结果取平均得到0.2547的结果