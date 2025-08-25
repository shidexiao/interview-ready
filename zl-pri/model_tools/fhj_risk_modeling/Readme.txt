__author__ = 'fenghaijie / hjfeng0630@qq.com'
修订时间：2019年6月

本风控建模脚本所包含的函数目录如下：
---------------------------------------------------------
-- get_hive_data.py
模块描述：获取数据（Get Data, GD）
功能包括：
1. 取数（Get Data, GD）: 连接Hive数据库，获取数据并保存在本地
---------------------------------------------------------
-- exploratory_data_analysis.py
模块描述：探索性数据分析（Exploratory Data Analysis, EDA）
功能包括：
1. 探索数据分布（Explore Data Distribution, EDD）——（连续变量版 + 离散变量版）
    1.1 edd_for_continue_var
    1.2 edd_for_discrete_var
2. 缺失率统计（Missing Rate）
3. 目标变量统计(Target Rate)
---------------------------------------------------------
-- feature_process.py
模块描述：特征处理（Feature Process, FP）
功能包括：
1. vif_table          :方差膨胀因子（VIF）计算, 只适用于LR模型
2. iv_grouply_table   :信息量分组计算
3. var_cluster        :变量聚类实现降维, 再进行变量筛选
4. correlation_plot   :计算各变量的Pearson相关性系数, 并可视化
---------------------------------------------------------
-- feature_select.py
模块描述：特征筛选（Feature Select, FS）
功能包括：
1. psi_based_feature_select :根据psi_grouply_table()函数生成的psi_table, 筛选出给定分组上psi 小于 threshold的变量
2. cv_feature_select        :根据cv_grouply_table()函数生成的cv_table, 筛选出给定分组上cv 小于 threshold的变量
3. merge_feature_index
----------------------
首先，自动通过EDA(Exploratory Data Analysis)对特征进行初步筛选。例如：
(1) 筛选掉缺失率较高，方差恒定不变
(2) 在时间维度上不够稳定。

其次，自动综合多种常用算法和规则，进一步筛选出有效的特征。例如：
(1) 变量聚类；
>>> var_cluster(input_df=df, n_clusters=3, var_list=None)
>>>  
	var	cluster
0	Pclass	0
1	SibSp	0
2	Parch	0
3	PassengerId	1
4	Age	1
5	Survived	2
6	Fare	2
7	score	2

(2) IV值筛选；
>>> iv_grouply_table(input_df=df, target_var=target, var_list=c_cols, group_var='group')
>>> 
	var	seg1	seg2	mean	std	cv	iv_rank	target
1	Fare	0.648085	0.571201	0.609643	0.054365	0.089029	1	Survived
0	Age	0.222415	0.265318	0.243866	0.030337	0.123892	2	Survived

(3) 树模型输出重要性；
>>> feat_importance(model=trained_model, feat_list=c_cols, accumulate_score=0.95)
>>>
累积score达到0.95时的特征序号为0
	var	score	score_rank	topk
0	Age	0.436644	1	
1	Fare	0.416096	2	
2	SibSp	0.090753	3	
3	Pclass	0.056507	4	

(4) 变量shap值；
>>> shap_df = shap_value(trained_model=trained_rf,
                         X_train=develop_data_ins[feats].fillna(999999),
                         var_list=c_cols)
>>> shap_df
	var	shap_value
0	Age	-17.900106
1	Fare	-49.150763

⑤ 共线性筛选；
最后，选择出综合排序TopN，以及各个类别中排名靠前的特征。可以把几千维度的特征降低到几百个维度的范围内，并且在减少特征的同时，保留特征的多样性。
得到：
	var	cluster	iv	iv_rank	feat_importance	shap_value
0	Pclass	0	0.000000	4.0	0.056507	9.648493
1	SibSp	0	0.017276	3.0	0.090753	4.380519
2	Parch	0	NaN	NaN	NaN	NaN
3	PassengerId	1	NaN	NaN	NaN	NaN
4	Age	1	0.243866	2.0	0.436644	4.860779
5	Survived	2	NaN	NaN	NaN	NaN
6	Fare	2	0.609643	1.0	0.416096	-90.666223
7	score	2	NaN	NaN	NaN	NaN
---------------------------------------------------------
-- logistic_regression.py
模块描述：逻辑斯特回归（Logistic Regression, LR）
功能包括：
1. statsmodels.lr:  调用statsmodels模块中的lr模型进行回归
2. sklearn_lr:      调用sklean模块中的lr模型进行回归，可调整权重
---------------------------------------------------------
-- model_fitting.py
模块描述：模型拟合(Model Fittiing)
功能包括：
1. train_model: 模型训练
2. test_model:  模型预测
3. feat_importance: 根据训练完毕的模型得到特征重要性
---------------------------------------------------------
-- model_scaling.py  
模块描述：模型分数变化(Model Scaling)
功能包括：
1. scorecard_scaling: 评分卡分数变换，将LR输出结果进行尺度变换（按Odds校准）
---------------------------------------------------------   
-- model_deploy.py
模块描述：模型部署（Model Deploy）
功能包括：
1. binmap_to_sql       :根据binmap文件生成SQL语句，用于评分卡SQL部署
2. scorecard_transform :读取评分卡binmap文件, 对输入的单变量取值判断落在哪个分箱进行预测
3. scorecard_predict   :根据生成的bimap文件，批量对入模变量进行评分卡预测
---------------------------------------------------------
-- ranking_evaluation.py
模块描述：模型排序性指标（Ranking）
功能包括：
1.model_ranking_eval  :根据ks_table，可视化观察模型排序性
1) 放款层bad_rate
2) 放款层lift
3) 放款层log(odds)
4) 申请层reject_rate  
---------------------------------------------------------   
-- stability_evaluation.py
模块描述：稳定性评估指标(population stability index ,PSI)
功能包括：
1. psi_for_continue_var      :针对连续变量的population_stability_index基础函数
2. psi_for_discrete_var      :针对离散变量的population_stability_index基础函数
3. psi_grouply_table         :按样本集[INS/OOS/OOT]或时间窗分组计算变量列表中所有变量的PSI
4. coefficent_stability_test :多次训练LR模型, 得到各变量权重系数估计结果, 分析点估计稳定性。
5. coefficent_stability_analysis: 根据coefficent_stability_test()结果可视化分析
---------------------------------------------------------      
-- ks_evaluation.py
模块描述：模型区分度KS评价指标（Kolmogorov-Smirnov, KS）
功能包括：
1. ks_compute              :利用scipy库函数计算ks指标
2. ks_grouply_calculate    :利用scipy库函数计算每组中各变量的KS指标
3. ks_table                :生成ks_table, 可观察每个bin内的正负样本数，几率odds，lift和ks
4. ks_grouply_table        :分组计算ks_table
5. ks_plot                 :ks计算可视化图
6. ks_table_plot           :读取ks_table()生成的ks_table可视化绘制KS曲线
7. var_marginal_ks         :单变量边际ks
---------------------------------------------------------
-- save_load_model.py
模块描述：保存及载入模型文件（Save + Load Model）
功能包括：
1. pkl_save:    sklearn机器学习模型, 保存为pkl格式
2. pmml_save:   sklearn机器学习模型, 保存为pmml格式
3. binmap_save: 评分卡模型, 保存为binmap.csv格式
4. pkl_to_pmml: 将sklearn机器学习模型pkl文件转换为pmml文件
5. load_pkl_model: 载入pkl文件。
6. load_pkl_predict: 载入pkl文件直接预测
---------------------------------------------------------
-- reject_inference.py
模块描述：拒绝推断(Reject Inference, RI)
功能包括：
1. proportional_assignment: 简单随机推断(分散打包法)
2. simple_augmentation:  拒绝样本按欺诈评分降序排列，取头部样本赋予1，其余赋予0。
3. parcelling: 分箱随机赋值。
---------------------------------------------------------