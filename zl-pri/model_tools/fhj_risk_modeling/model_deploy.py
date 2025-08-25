# -*- coding:utf-8 -*-
__author__ = 'fenghaijie / hjfeng0630@qq.com'

import math
import pandas as pd
import numpy as np

"""
模块描述：模型部署（Model Deploy）
功能包括：
1.binmap_to_sql       :根据binmap文件生成SQL语句，用于评分卡SQL部署
2.scorecard_transform :读取评分卡binmap文件, 对输入的单变量取值判断落在哪个分箱进行预测
3.scorecard_predict   :根据生成的bimap文件，批量对入模变量进行评分卡预测
"""

def binmap_to_sql(input_df, source_table='dm_ds_fraud.fhj_source_table'):
    """
    ----------------------------------------------------------------------
    功能：根据binmap文件生成SQL语句，用于评分卡SQL部署
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 由评分卡脚本生成的binmap.csv文件
    :param source_table: str, 数据源表
    ----------------------------------------------------------------------
    :return sql: str, sql语句
    ----------------------------------------------------------------------
    示例：
    >>> alpha, beta = scorecard_scaling()
    >>> 评分卡类型为信用评分卡, 分数转换公式为：score = 697.672264890213 + -36.06737602222409 * ln(odds)
    
    >>> binmap_df['weight'] = -0.2
    >>> binmap_df['score']  = round(binmap_df['weight'] * beta * binmap_df['woe'])
    >>> binmap_df.head()
    >>>
    	feature	labels	bins	min_score	max_score	obs	bad	good	bad_rate	good_rate	odds_good	woe	bin_iv	iv	weight	score
    0	Age	0	-inf	-inf	18.000000	139	70	69	0.503597	0.496403	0.985714	-0.487676	0.038524	0.087583	-0.2	-4.0
    1	Age	1	18.000000	18.000000	23.000000	107	33	74	0.308411	0.691589	2.242424	0.334270	0.012802	0.087583	-0.2	2.0
    
    >>> sql = bimap_to_sql(input_df=final_woe_df)
    >>> sql
    select order_id, 
           600 as basescore, 
           case 
               when Age <= 18.0 then -4.0 
               when Age >  18.0 and Age <= 23.0 then 2.0 
               when Age >  23.0 and Age <= 28.0 then -0.0 
               when Age >  28.0 and Age <= 34.0 then -1.0 
               when Age >  34.0 and Age <= 44.0 then -1.0 
               when Age >  44.0 then 0.0 
               else 3.0 
           end as Age_score, 
           case 
               when Embarked in ('C') then -7.0 
               when Embarked in ('Q') then -0.0 
               when Embarked in ('S') then 2.0 
               else -20.0 
           end as Embarked_score, 
    from dm_ds_fraud.fhj_source_table
    ----------------------------------------------------------------------
    """
    if 'basescore' in set(input_df.columns):
        base_score = list(input_df['basescore'])[0]
    else:
        base_score = 600
    
    def var_sql(input_df, var):

        bin_label_list = list(input_df['labels'])
        min_score_list = list(input_df['min_score'])
        max_score_list = list(input_df['max_score'])
        score_list = list(input_df['score'])
        bucket_num = len(min_score_list)

        def c_var_sql(var, min_score, max_score, bin_score):
            """
            ----------------------------------------------------------------------
            功能：连续变量生成SQL语句
            ----------------------------------------------------------------------
            :param var: str, 变量名
            :param min_score: float, 分箱下界
            :param max_score: float, 分箱上界
            :param bin_score: int, 分箱得分
            ----------------------------------------------------------------------
            :return sql: str, sql语句
            ----------------------------------------------------------------------
            """
            if math.isinf(min_score):
                sql = "           when {} <= {} then {} \n".format(var, max_score, bin_score)
            elif math.isinf(max_score):
                sql = "           when {} >  {} then {} \n".format(var, min_score, bin_score)
            elif not (math.isnan(min_score) or math.isnan(max_score)):
                sql = "           when {} >  {} and {} <= {} then {} \n".format(var, min_score, var, max_score, bin_score)
            else:
                sql = "           else {} \n       end as {}_score, \n".format(bin_score, var)

            return sql

        def d_var_sql(var, min_score, max_score, bin_score):
            """
            ----------------------------------------------------------------------
            功能：离散变量生成SQL语句
            ----------------------------------------------------------------------
            :param var: str, 变量名
            :param min_score: str, 分箱下界
            :param max_score: str, 分箱上界
            :param bin_score: int, 分箱得分
            ----------------------------------------------------------------------
            :return sql: str, sql语句
            ----------------------------------------------------------------------
            """
            if type(min_score) == str:
                sql = "           when {} in ('{}') then {} \n".format(var, min_score, bin_score)
            else:
                sql = "           else {} \n       end as {}_score, \n".format(bin_score, var)

            return sql

        var_sql = "       case \n"
        for idx in range(bucket_num):
            min_score = min_score_list[idx]
            max_score = max_score_list[idx]
            bin_score = score_list[idx]
            if str(bin_label_list[0]).startswith('d_'):
                sql = d_var_sql(var, min_score, max_score, bin_score)
            else:
                sql = c_var_sql(var, min_score, max_score, bin_score)
            var_sql += sql

        return var_sql 
    
    output_sql =  "select order_id, \n"
    output_sql += "       {} as basescore, \n".format(base_score)
    var_list = sorted(list(set(input_df['feature'])))
    for var in var_list:
        var_input_df = input_df[input_df['feature'] == var]
        output_sql += var_sql(var_input_df, var)
    output_sql += "from {}".format(source_table)
    
    return output_sql


def scorecard_transform(x, bins, v_type='c', spec_values=None, replace_missing=None):
    """
    功能: 读取评分卡binmap文件, 对输入的单变量取值判断落在哪个分箱进行预测
    ----------------------------------------------------------------------
    :param x: pd.Series, 变量取值
    :param bins: pd.DataFrame, 当前变量的binmap分箱表
    :param v_type: str, 变量类型, ‘c’=连续变量, ‘d’=离散变量
    ----------------------------------------------------------------------
    :return output_list, pd.Series, 某个变量取值所对应的分箱分数
    ----------------------------------------------------------------------
    示例：
    >>> scorecard_transform(pd.Series(df['Age']), age_bin_df, 'c')['score']
    >>>
    index  score
    0      2.0
    1     -1.0
    2     -0.0
    3     -1.0
    ----------------------------------------------------------------------
    """
    if type(spec_values) == dict:  # Parsing special values to dict for cont variables
        spec_values = {}
        for k, v in spec_values.items():
            if v.startswith('d_'):
                spec_values[k] = v
            else:
                spec_values[k] = 'd_' + v
    else:
        if spec_values is None:
            spec_values = {}
        else:
            spec_values = {i: 'd_' + str(i) for i in spec_values}

    if not isinstance(x, pd.Series):
        raise TypeError("pandas.Series type expected")
    if bins is None:
        raise Exception('Get Bins Mapping, please')
    df = pd.DataFrame({"X": x, 'order': np.arange(x.size)})
    
    def _split_sample(df):
        if v_type == 'd':
            return df, None
        sp_values_flag = df['X'].isin(spec_values.keys()).values | df['X'].isnull().values
        df_sp_values = df[sp_values_flag].copy()
        df_cont = df[np.logical_not(sp_values_flag)].copy()
        return df_sp_values, df_cont
    
    df_sp_values, df_cont = _split_sample(df)
    tr_bins = bins[['woe', 'labels','score']].copy()
    
    if replace_missing is not None:
        tr_bins = tr_bins.append({'labels': 'd__transform_missing_replacement__', 'woe': replace_missing}, ignore_index=True)
        
    # function checks existence of special values, raises error if sp do not exist in training set
    def get_sp_label(x_):
        if x_ in spec_values.keys():
            return spec_values[x_]
        else:
            str_x = 'd_' + str(x_)
            if str_x in list(bins['labels']):
                return str_x
            else:
                if replace_missing is not None:
                    return 'd__transform_missing_replacement__'
                else:
                    raise ValueError('Value {} does not exist in the training set'.format(str_x))
                    
    def __get_cont_bins(bins):
        """
        Helper function
        :return: return continous part of self.bins
        """
        return bins[bins['labels'].apply(lambda z: not z.startswith('d_'))]
    
    # assigning labels to discrete part
    df_sp_values['labels'] = df_sp_values['X'].apply(get_sp_label)
    
    # assigning labels to continuous part
    c_bins = __get_cont_bins(bins)
    
    if v_type != 'd':
        cuts = pd.cut(df_cont['X'], bins=np.append(c_bins["bins"].astype(float), (float("inf"),)), labels=c_bins["labels"])
        df_cont['labels'] = cuts.astype(str)
        
    # Joining continuous and discrete parts
    df = df_sp_values.append(df_cont)
    
    # assigning woe
    df = pd.merge(df, tr_bins[['woe', 'labels','score']].drop_duplicates(), left_on=['labels'], right_on=['labels'])
    # returning to original observation order
    
    df.sort_values('order', inplace=True)
    
    output_list = df.set_index(x.index)
    
    return output_list


def scorecard_predict(input_df, binmap, d_varlist=None, c_varlist=None, k_varlist=['order_id']):
    """
    ----------------------------------------------------------------------
    功能: 根据生成的bimap文件，批量对入模变量进行评分卡预测
         输入数据表在预测前建议做最坏情况填充，以应对训练集无空分箱的情况。
         离散变量若有分箱合并map，需在原始输入数据中预先map
    ----------------------------------------------------------------------
    :param input_df : Pandas DataFrame. 原始数据
    :param binmap_path : str. binmap文件路径
    :param d_varlist : list. 离散变量列表，元素为str类型
    :param c_varlist : list. 连续变量列表，元素为str类型
    :param k_varlist : list. 主键，如['user_id', 'order_id', 'loan_type']
    ----------------------------------------------------------------------
    :return output_df : Pandas DataFrame. 订单及对应的评分卡得分
    ----------------------------------------------------------------------
    示例：
    >>> scorecard_predict(input_df=df, binmap=final_woe_df, 
                          d_varlist=['Embarked'], c_varlist=['Age'], k_varlist=['PassengerId'])
    >>>
    	PassengerId	Embarked_score	Age_score	basescore	finalscore
    0	1	2.0	2.0	600	604.0
    1	2	-7.0	-1.0	600	592.0
    2	3	2.0	-0.0	600	602.0
    ----------------------------------------------------------------------
    """
    input_df_copy = input_df.copy()
    
    cols = set(input_df_copy.columns)
    if not set(c_varlist).issubset(set(cols)):
        raise Exception('参数c_varlist取值包含不属于input_df的变量，请检查!')
    if not set(d_varlist).issubset(set(cols)):
        raise Exception('参数d_varlist取值包含不属于input_df的变量，请检查!')
    if not set(k_varlist).issubset(set(cols)):
        raise Exception('参数k_varlist取值包含不属于input_df的变量，请检查!')
            
    input_df_copy[c_varlist] = input_df_copy[c_varlist].astype(float)
    
    if not type(binmap) == str:
        binmap_df = binmap
    else:
        binmap_df = pd.read_csv(binmap, encoding='utf-8')
        
    if not set(['basescore']).issubset(set(binmap_df.columns)):
        raise Exception('binmap文件中必须包含字段basescore，请检查!')
       
    """预测"""
    for feature in set(d_varlist):
        print(feature)
        feature_woe = feature + '_woe'
        feature_score = feature + '_score' 
        var_binmap_df = binmap_df[binmap_df['feature'] == feature]
        input_df_copy.loc[:, feature_score] = scorecard_transform(pd.Series(input_df_copy[feature]), \
                                                                  var_binmap_df, 'd')['score']
        
    for feature in set(c_varlist):
        print(feature)
        feature_woe = feature + '_woe'
        feature_score = feature + '_score'
        var_binmap_df = binmap_df[binmap_df['feature'] == feature]
        input_df_copy.loc[:, feature_score] = scorecard_transform(pd.Series(input_df_copy[feature].astype(float)), \
                                                                  var_binmap_df, 'c')['score']

    var_list = d_varlist + c_varlist
    var_list = [var + '_score' for var in var_list]
    input_df_copy.loc[:, 'basescore'] = binmap_df['basescore'].unique()
    input_df_copy.loc[:, 'finalscore'] = input_df_copy[var_list].sum(axis=1) + binmap_df['basescore'].unique()
    
    output_cols = k_varlist + var_list + ['basescore', 'finalscore']
    output_df = input_df_copy[output_cols]
    
    return output_df