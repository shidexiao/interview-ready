# -*- coding:utf-8 -*-
from __future__ import division
__author__ = 'fenghaijie / hjfeng0630@qq.com'

import math
import operator
import pandas as pd
import numpy as np
# import shap

"""
模块描述：特征筛选（Feature Select, FS）
功能包括：
1.psi_based_feature_select :根据psi_grouply_table()函数生成的psi_table, 筛选出给定分组上psi < threshold的变量
2.cv_feature_select        :根据cv_grouply_table()函数生成的cv_table, 筛选出给定分组上cv < threshold的变量
3.merge_feature_index

首先，自动通过EDA(Exploratory Data Analysis)对特征进行初步筛选。例如：
① 筛选掉缺失率较高，方差恒定不变
② 在时间维度上不够稳定。

其次，自动综合多种常用算法和规则，进一步筛选出有效的特征。例如：
① 变量聚类；
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

② IV值筛选；
>>> iv_grouply_table(input_df=df, target_var=target, var_list=c_cols, group_var='group')
>>> 
	var	seg1	seg2	mean	std	cv	iv_rank	target
1	Fare	0.648085	0.571201	0.609643	0.054365	0.089029	1	Survived
0	Age	0.222415	0.265318	0.243866	0.030337	0.123892	2	Survived

③ 树模型输出重要性；
>>> feat_importance(model=trained_model, feat_list=c_cols, accumulate_score=0.95)
>>>
累积score达到0.95时的特征序号为0
	var	score	score_rank	topk
0	Age	0.436644	1	
1	Fare	0.416096	2	
2	SibSp	0.090753	3	
3	Pclass	0.056507	4	<<<

四 变量SHAP值
>>> shap_df = shap_value(trained_model=trained_rf,
                         X_train=develop_data_ins[feats].fillna(999999),
                         var_list=c_cols)
>>> shap_df
	var	shap_value
0	Age	-17.900106
1	Fare	-49.150763

五 共线性筛选。
最后，选择出综合排序TopN，以及各个类别中排名靠前的特征。可以把几千维度的特征降低到几百个维度的范围内，并且在减少特征的同时，保留特征的多样性。
得到：
	var	cluster	iv	iv_rank	feat_importance	shap_value
0	Pclass	0	0.000000	4.0	0.056507	9.648493
1	SibSp	0	0.017276	3.0	0.090753	4.380519
2	Parch	0	NaN	NaN	NaN	NaN
3	PassengerId	1	NaN	NaN	NaN	NaN
4	Age	1	0.243866	2.0	0.436644	4.860779
5	Survived	2	NaN	NaN	NaN	NaN
6	Fare	2	0.6s09643	1.0	0.416096	-90.666223
7	score	2	NaN	NaN	NaN	NaN
"""


def psi_based_feature_select(psi_table, compare_group_list=None, threshold=0.2, rule='min'):
    '''
    ----------------------------------------------------------------------    
    功能：根据psi_grouply_table()函数生成的psi_table, 筛选出给定分组上psi < threshold的变量
    ----------------------------------------------------------------------
    :param psi_table: pd.DataFrame, 输入的psi_table, 由psi_grouply_table()函数生成
    :param compare_group_list: list, 计算psi时的分组。例如以'INS'为期望组，则比较组为['OOS', 'OOT1', 'OOT2']
    :param threshold: float, 阈值，默认0.2
    :param rule: str, 筛选规则。默认值rule ='min'
                      1) rule ='min',  代表只要各分组得到的psi值中, min(psi)  > threshold, 则剔除该特征
                      2) rule ='max',  代表只要各分组得到的psi值中, max(psi)  > threshold, 则剔除该特征
                      3) rule ='mean', 代表只要各分组得到的psi值中, mean(psi) > threshold, 则剔除该特征
    ----------------------------------------------------------------------    
    :return selected_varlist: 筛选后的特征列表
    ----------------------------------------------------------------------
    示例：
    >>> psi_t1 = psi_grouply_table(input_df=df, 
                                   group_var='group',
                                   benchmark_list=['seg1'], 
                                   compare_list=None, 
                                   c_var_list=['Age','Fare', 'score'],
                                   d_var_list=['Age'])
    >>> psi_t1
    	seg2	mean	max	benchmark
    Fare	0.0349	0.0349	0.0349	seg1
    score	0.3857	0.3857	0.3857	seg1
    Age	4.7407	4.7407	4.7407	seg1
    ----------
    >>> psi_based_feature_select(psi_table=psi_t1, compare_group_list=None, threshold=0.2)
    >>> ['Fare', 'score']
    ----------------------------------------------------------------------
    '''
    if rule not in set(['min', 'max', 'mean']):
        raise Exception('筛选规则只能是：min、max、mean')
        
    if compare_group_list is None:
        compare_group_list = list(set(psi_table.columns) \
                                - set(['mean', 'max', 'benchmark']) - set(psi_table['benchmark']))
        
    input_df = psi_table.copy()
    if rule == 'min':
        input_df['drop_flag'] = input_df.apply(lambda row: 1 if min(row[compare_group_list]) > threshold else 0, axis=1)
    elif rule == 'max':
        input_df['drop_flag'] = input_df.apply(lambda row: 1 if max(row[compare_group_list]) > threshold else 0, axis=1)
    else:
        input_df['drop_flag'] = input_df.apply(lambda row: 1 if row['mean'] > threshold else 0, axis=1)
        
    selected_varlist = list(input_df[input_df['drop_flag'] == 0].index)

    return selected_varlist


def cv_feature_select(cv_table, threshold=0.2):
    """
    功能：根据cv_grouply_table()函数生成的cv_table, 筛选出给定分组上cv < threshold的变量
    ----------------------------------------------------------------------
    :param cv_table:  pd.dataframe, 由cv_grouply_table()生成
    :param threshold: float, 阈值，默认0.2
    ----------------------------------------------------------------------
    :return selected_varlist: 筛选后的特征列表
    ----------------------------------------------------------------------
    示例：
    >>> edd1 = edd_for_continue_var(input_df=df, group_var='group')
    >>> cv_t1 = cv_grouply_table(edd_table=edd1, eval_index='mean')
    >>> cv_t1
    	seg1	seg2	mean	std	cv	eval_index
    PassengerId	446.223230	445.638235	445.930733	0.413654	0.000928	mean
    Fare	32.592944	31.574228	32.083586	0.720341	0.022451	mean
    Pclass	2.372051	2.205882	2.288967	0.117499	0.051310	mean
    >>> selected_varlist = cv_feature_select(cv_t1, threshold=0.2)
    >>> selected_varlist
        ['x1', 'x2']
    ----------------------------------------------------------------------
    """
    selected_varlist = list(cv_table[cv_table['cv'] <= threshold].index)
    return selected_varlist


def merge_feature_index(var_cluster_df, iv_df=None, importance_df=None, shap_df=None):
    """
    功能：将各类特征指标汇总后筛选
    ----------------------------------------------------------------------
    :param var_cluster_df: 由var_cluster()计算得到
    :param iv_df: 由iv_grouply_table()计算得到
    :param importance_df: 由feat_importance()计算得到
    :param shap_df: 由shap_value()计算得到
    ----------------------------------------------------------------------
    :return output_df: pd.DataFrame, 包括特征，特征重要度等
    ----------------------------------------------------------------------
    示例：
    >>> varclus_df = var_cluster(input_df=df, n_clusters=3, var_list=None)
    >>> iv_df = iv_grouply_table(input_df=df, target_var=target, var_list=c_cols, group_var='group')
    >>> import_df = feat_importance(model=fitted_model, feat_list=c_cols, accumulate_score=0.95)
    >>> shap_df = shap_value(trained_model=fitted_model, X_train=df[c_cols], var_list=c_cols)
    
    >>> merge_feature_index(var_cluster_df=varclus_df, 
                    iv_df=iv_df, 
                    importance_df=import_df,
                    shap_df=shap_df)
    >>>  
    	var	cluster	iv	iv_rank	tree_import	tree_import_rank	shap_value
    0	Parch	0	NaN	NaN	NaN	NaN	NaN
    1	Pclass	0	0.000000	4.0	0.056507	4.0	9.648493
    2	SibSp	0	0.017276	3.0	0.090753	3.0	4.380519
    3	Age	1	0.243866	2.0	0.436644	1.0	4.860779
    4	PassengerId	1	NaN	NaN	NaN	NaN	NaN
    5	Survived	2	NaN	NaN	NaN	NaN	NaN
    6	Fare	2	0.609643	1.0	0.416096	2.0	-90.666223
    7	score	2	NaN	NaN	NaN	NaN	NaN
    ----------------------------------------------------------------------
    """
    output_df = var_cluster_df
    if iv_df is not None:
        iv_df = iv_df[['var', 'mean', 'iv_rank']]
        iv_df.columns = ['var', 'iv', 'iv_rank']
        output_df = pd.merge(output_df, iv_df, on='var', how='left')
    if importance_df is not None:
        importance_df = importance_df[['var', 'score', 'score_rank']]
        importance_df.columns = ['var', 'tree_import', 'tree_import_rank']
        output_df = pd.merge(output_df, importance_df, on='var', how='left')
    if shap_df is not None:
        output_df = pd.merge(output_df, shap_df, on='var', how='left')
        
    return output_df

