# 三方数据自动评估脚本

## 1.数据获取模块

data_fetch.py文件Fetcher类，用于pyspark关联hive表，直接加载业务表至本地关联评估数据

对象：

1.hive_sc, spark或者hive context

2.driver, 评估数据driver表

3.key, 评估数据的主键列表(三要素+back_date)

4.evalute_data, 评估三方数据

* get_target(self)
  获取评估样本Y表中的目标变量，包括dpd1@mob1(入催)、dpd30@mob1/3/6(风险)，营销活动标签(营销)
* get_score(self)
  获取WB/PBOC/Titan/GAIA系列模型分数
* get_base_variables(self)
  获取Titan系列最新已使用数据源baseline变量
* get_segment(self)
  获取最新的客户分群标签
* get_business_label(self)
  获取客户的是否授信通过、授信成功3/30天后是否支用、是否入催dpd1@mob1、额度使用率、
  是否逾期30+(count/amount)、风险、回款(dpd1_mob1, dpd30_mob3)
* merge(self)
  关联以上所有数据，合并为最终宽表

## 2.单变量评估模块

feature_evalutor.py文件single_evalutor类，用于单变量数据评估

对象：

1.evalute_data, 评估样本宽表X+Y

2.condicate_variables，待评估三方数据样本

3.target_list, 评估目标变量列表

* preprocess(self)
  数据预处理
* coverage(self)
  统计变量查得率
* efficacy(self)
  单变量IV，KS，Lift

### 3.模型增益模块

gain_evalutor.py文件Gain类，评估数据源子分及数据源相较Titan/Gaia系列增益

对象：

1.evalute_data

1.evalute_data, 评估样本宽表X+Y

2.condicate_variables，待评估三方数据样本

3.base_variables，Titan系列在用数据源列表

4.target，评估目标变量

5.split_node, train及OOT划分时间节点
