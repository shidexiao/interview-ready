# -*- coding:utf-8 -*-
__author__ = 'fenghaijie / hjfeng0630@qq.com'

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
# from sklearn.externals.joblib import Parallel, delayed

"""
模块描述：探索性数据分析（Exploratory Data Analysis, EDA）
功能包括：
1.探索数据分布（Explore Data Distribution, EDD）——（连续变量版 + 离散变量版）
1.1 edd_for_continue_var
1.2 edd_for_discrete_var
2.缺失率统计（Missing Rate）
3.目标变量统计(Target Rate)
4.变异系数(Coefficient of Variation)
"""

def edd_for_continue_var(input_df, var_list=None, group_var=None, group_list=None, save_file_path=None):
    """
    ----------------------------------------------------------------------
    功能：探索数据分布（Explore Data Distribution, EDD）——连续变量版本
         统计每个组的数据分布, 包括以下指标:
         [样本总数total、覆盖数cover_cnt、覆盖率cover_rate、均值mean、标准差std、最小值min、分位数P25、分位数P50、分位数P75、最大值max]
    注意：只对【int/float/double】数值型变量进行统计, 不对类别型变量统计
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param var_list: list, 需要统计缺失率的变量列表，默认值=None，对所有变量统计。必须要在input_df中，否则抛异常。
    :param group_var: string, 分组依据，如按[组，月，周]。默认值=None, 对整个数据集统计。示例: 'apply_month'
    :param group_list: list, 分组元素取值, 若为None，则默认为所有取值。示例: [201701, 201702]
    :param save_file_path: string, csv文件保存路径. 默认值=None
    ----------------------------------------------------------------------
    :return output_df: pd.DataFrame, 返回结果
    ----------------------------------------------------------------------
    示例:
    >>> edd_continue_var_distribution(input_df=df)
    >>>
    	total	cover_cnt	cover_rate	mean	std	min	25%	50%	75%	max	group
    Age	891	714	0.801347	29.699118	14.526497	0.420000	20.125000	28.000000	38.000000	80.000000	all
    Fare	891	891	1.000000	32.204208	49.693429	0.000000	7.910400	14.454200	31.000000	512.329200	all
    ----------------------------------------------------------------------
    """
    input_df_copy = input_df.copy()
    if not isinstance(input_df_copy, pd.core.frame.DataFrame):
        raise Exception('参数input_df_copy的格式错误，应为pandas.core.frame.DataFrame')

    cols = list(input_df_copy.columns)
    if group_var is not None:
        if group_var not in set(cols):
            raise Exception('参数group_var取值包含不属于input_df_copy的变量，请检查!')
    else:
        group_var = 'group'
        input_df_copy[group_var] = 'all'
        
    if group_list is None: 
        group_list = sorted(list(set(input_df_copy[group_var])))
    else:
        group_list = sorted(list(set(group_list)))

    if var_list is not None:
        if not set(var_list).issubset(set(cols)):
            raise Exception('参数var_list取值包含不属于input_df_copy的变量，请检查!')
        else:
            cols = var_list
            
    """对各变量统计分布"""
    output_df = pd.DataFrame()
    for gp in tqdm(group_list):
        grp_input_df = input_df_copy[input_df_copy[group_var] == gp] # 分组
        tmp_df = pd.DataFrame(grp_input_df[cols].describe().T, )  # 分布描述函数
        tmp_df.loc[:, 'total'] = grp_input_df[cols].fillna(-999999).count()
        tmp_df.loc[:, 'cover_cnt'] = tmp_df['count'].apply(lambda x: int(x))
        tmp_df.loc[:, 'cover_rate'] = tmp_df['cover_cnt'] / tmp_df['total']
        tmp_df.loc[:, 'zero_cnt'] = grp_input_df[grp_input_df[cols] == 0][cols].count()
        tmp_df.loc[:, 'zero_rate'] = tmp_df['zero_cnt'] / tmp_df['cover_cnt']
        tmp_df.loc[:, 'group'] = gp
        output_df = pd.concat([output_df, tmp_df], axis=0, sort=False)
        
    output_df['var'] = output_df.index
    output_df = output_df.sort_values(by=['var', 'group'], ascending=[1, 1])
    output_df = output_df.drop(['var'], axis=1)
    
    output_df = output_df[['group', 'total', 'cover_cnt', 'cover_rate', 'zero_cnt', 'zero_rate', \
                           'min', '25%', '50%', '75%', 'max', 'mean', 'std']]

    """文件保存至指定路径"""
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.csv'):
            raise Exception('参数save_file_path不是csv文件后缀，请检查!')
        output_df.to_csv(save_file_path, encoding='utf-8', index=True)

    return output_df


def edd_for_discrete_var(input_df, var_list=None, group_var=None, group_list=None, save_file_path=None):
    """
    ----------------------------------------------------------------------
    功能：探索数据分布（Explore Data Distribution, EDD）——离散变量版本
         统计每个组的数据分布, 包括以下指标:
         [离散取值value、样本总数total、覆盖数cover_cnt、覆盖率cover_rate]
    注意：只对类别型变量统计
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param var_list: list, 需要统计缺失率的变量列表，默认值=None，对所有变量统计。必须要在input_df中，否则抛异常。
    :param group_var: string, 分组依据，如按[组，月，周]。默认值=None, 对整个数据集统计。示例: 'apply_month'
    :param group_list: list, 分组元素取值, 若为None，则默认为所有取值。示例: [201701, 201702]
    :param save_file_path: string, csv文件保存路径. 默认值=None
    ----------------------------------------------------------------------
    :return output_df: pd.DataFrame, 返回结果
    ----------------------------------------------------------------------
    示例:
    >>> edd_discrete_var_distribution(input_df=df).head()
    >>>
    	group	variable	value	total	cover_cnt	cover_rate
    0	all	Name	Abbing, Mr. Anthony	891	1	0.001122
    1	all	Name	Abbott, Mr. Rossmore Edward	891	1	0.001122
    ----------------------------------------------------------------------
    知识：
    离散变量是指取值是有限个离散值的变量，比如变量age取值为1~100的整数。animal变量取值{cat, dog}
    观察变量不同取值下的样本数、样本占比等指标，可帮助理解数据全貌。
    ----------------------------------------------------------------------
    """
    input_df_copy = input_df.copy()
    cols = list(input_df_copy.columns)
    if var_list is not None: # 用户输入离散变量列表
        if not set(var_list).issubset(set(cols)):
            raise Exception('参数var_list取值包含不属于input_df的变量，请检查!')
        else:
            d_cols = var_list
    else:
        # 用户不指定，则对所有string类型的变量进行统计
        cols_type_dict = dict(input_df_copy.dtypes)
        d_cols = [x for x in cols if cols_type_dict[x] == object]
        
    if group_var is not None:
        if group_var not in set(cols):
            raise Exception('参数group_var取值包含不属于input_df的变量，请检查!')
    else:
        group_var = 'group'
        input_df_copy[group_var] = 'all'
    
    if group_list is None: 
        group_list = sorted(list(set(input_df_copy[group_var])))
    else:
        group_list = sorted(list(set(group_list)))
                
    def discrete_stat(df, var):
        """功能: 单变量分布统计"""
        row_num = df.shape[0]
        df['dummy'] = 1
        a = df.groupby(var).count()['dummy']
        null_num = row_num - sum(list(a.values))    # 空值
        value_lst = [str(x) for x in list(a.index)] # 离散取值
        cnt_lst = list(a.values)                    # 样本数
        if null_num > 0:
            value_lst += ['missing']
            cnt_lst += [null_num]
        res = pd.DataFrame()
        res['value_cnt'] = len(value_lst)
        res['value'] = value_lst
        res['cover_cnt'] = cnt_lst
        res['variable'] = var 
        res['cover_rate'] = res['cover_cnt'].apply(lambda x: x / row_num)
        res['total'] = row_num
        res = res.sort_values(by=['cover_cnt', 'value'], ascending=[0, 1]).reset_index(drop=1)
        res = res[['variable', 'value_cnt', 'value', 'total', 'cover_cnt', 'cover_rate']]

        return res
    
    """对各变量统计分布"""
    output_df = pd.DataFrame(columns=['variable', 'value_cnt', 'value', 'total', 'cover_cnt', 'cover_rate', 'group'])
    for gp in tqdm(group_list):
        for x in tqdm(d_cols):
            tmp_df = discrete_stat(df=input_df_copy, var=x)
            tmp_df.loc[:, 'group'] = gp
            output_df = pd.concat([output_df, tmp_df], axis=0, sort=False)
        
    """文件保存至指定路径"""
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.csv'):
            raise Exception('参数save_file_path不是csv文件后缀，请检查!')
        output_df.to_csv(save_file_path, encoding='utf-8', index=True)

    return output_df
    

def target_rate_stat(input_df, target_var_list, group_var=None, group_list=None, save_file_path=None):
    """
    ----------------------------------------------------------------------
    功能：统计每个组的目标变量好坏样本数，比例等. 支持多个目标变量统计。
         ['group', 'total', 'effs', 'goods', 'bads', 'odds', 'bad_rate', 'bad_def']
    含义：[分组，样本数，有效样本数（进入表现期），负样本数，正样本数，坏好比, 正样本率, 坏样本定义]
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param target_var_list: list, 每个元素类型为string, 目标变量。示例: ['s1d30','s3d15']
    :param group_var: string, 分组依据，如按组，月，周。默认值=None, 则对整个数据集统计。示例: 'apply_month'
    :param group_list: list, 分组元素取值, 若为None，则默认为所有取值。示例: [201701, 201702]
    :param save_file_path: string, csv文件保存路径. 默认值=None
    ----------------------------------------------------------------------
    :return output_df: pd.DataFrame, 返回结果
    ----------------------------------------------------------------------
    示例:
    >>> target_rate_stat(input_df=df, target_var_list=[target])
    >>>
    	group	total	effs	effs_rate	inters	goods	bads	odds	bad_rate	bad_def
    0	all	891	891	1.0	0	549	342	0.622951	0.383838	Survived
    ----------------------------------------------------------------------
    知识：
    在确定建模目标变量后，一般按样本集(train/valid/test)或按时间窗(monthly/weekly)观察客群质量波动性。
    通常根据bad rate来分析当前样本集客群质量好或坏的原因，以及进行样本划分(INS/OOS/OOT)
    INS: in the sample, 样本内训练，= 训练集
    OOS: out of sample, 样本外测试，= 验证集
    OOT: out of time,   时间外测试，= 测试集
    ----------------------------------------------------------------------
    """
    input_df_copy = input_df.copy()
    if not isinstance(input_df_copy, pd.core.frame.DataFrame):
        raise Exception('参数input_df的格式错误，应为pandas.core.frame.DataFrame')
    if not set(target_var_list).issubset(set(input_df_copy.columns)):
        raise Exception('参数target_var_list取值包含不属于input_df的变量，请检查!')
        
    if group_var is not None:
        if group_var not in set(input_df_copy.columns):
            raise Exception('参数group_var取值包含不属于input_df的变量，请检查!')
    else:
        group_var = 'group'
        input_df_copy[group_var] = 'all'
            
    if group_list is None: 
        group_list = sorted(list(set(input_df_copy[group_var])))
    else:
        group_list = sorted(list(set(group_list)))

    def target_var_stat_(df, target_var_, group_var_):
        df['dummy'] = 1
        out_df = pd.DataFrame()
        out_df.loc[:, 'total'] = df.groupby(group_var_)['dummy'].count()
        group_values = df.groupby(group_var_).count().index
        
        df = df.loc[(df[target_var_].isin([0, 1])), :]
        out_df.loc[:, 'effs'] = df.groupby(group_var_)[target_var_].count()
        out_df.loc[:, 'effs_rate'] = out_df.effs * 1.0 / out_df.total
        out_df.loc[:, 'bads'] = df.groupby(group_var_)[target_var_].sum()
        out_df.loc[:, 'goods'] = out_df.effs - out_df.bads
        out_df.loc[:, 'inters'] = out_df.total - out_df.effs
        out_df.loc[:, 'odds'] = out_df.bads * 1.0 / out_df.goods
        out_df.loc[:, 'bad_rate'] = out_df.bads * 1.0 / out_df.effs
        out_df.loc[:, 'bad_def'] = target_var_
        out_df.loc[:, 'group'] = group_values
        out_df = out_df.reset_index(drop=True)
        out_df = out_df[['group','total','effs','effs_rate','inters','goods','bads','odds','bad_rate','bad_def']]
        out_df = out_df.fillna(0)
        out_df[['total','effs','inters','goods','bads']] = out_df[['total','effs','inters','goods','bads']].astype('int')

        return out_df

    output_df = pd.DataFrame()
    for target_var in tqdm(target_var_list):
        stat_df = target_var_stat_(input_df_copy, target_var, group_var)
        output_df = pd.concat([output_df, stat_df], axis=0, sort=False)

    """文件保存至指定路径"""
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.csv'):
            raise Exception('参数save_file_path不是csv文件后缀，请检查!')
        output_df.to_csv(save_file_path, encoding='utf-8', index=False)

    return output_df


def missing_rate_stat(input_df, var_list=None, group_var=None, group_list=None, save_file_path=None):
    """
    ----------------------------------------------------------------------
    功能：缺失率统计（Missing Rate）
         统计每个组的缺失率，以及缺失率的变异系数（cv）
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param var_list: list, 需要统计缺失率的变量列表，默认值=None，对所有变量统计。必须要在input_df中，否则抛异常。
    :param group_var: string, 分组依据，如按[组，月，周]。默认值=None, 对整个数据集统计。示例: 'apply_month'
    :param group_list: list, 分组元素取值, 若为None，则默认为所有取值。示例: [201701, 201702]
    :param save_file_path: string, csv文件保存路径. 默认值=None
    ----------------------------------------------------------------------
    :return output_df: pd.DataFrame, 返回结果
    ----------------------------------------------------------------------
    示例:
    >>> missing_rate_stat(input_df=df).head()
    >>>
    	all	mean	std	cv
    Cabin	0.771044	0.771044	NaN	NaN
    Age	0.198653	0.198653	NaN	NaN
    ----------------------------------------------------------------------
    知识：
    缺失率用于分析某个数据源（变量）在样本上的覆盖度，覆盖率越高，价值越大。
    对某些不应该出现缺失的数据源，需追溯原因。
    对存在缺失的数据源，给出业务解释。例如：设备app信息只覆盖安卓人群，那么app数据源将存在50～60%的缺失率。
    ----------------------------------------------------------------------
    """
    def sub_missing_rate_stat(df):
        total = df.isnull().sum().sort_values(ascending=False)
        percent = (df.isnull().sum() * 1.0 / df.isnull().count()).sort_values(ascending=False)
        out_df = pd.concat([total, percent], axis=1, keys=['missing_cnt', 'missing_rate'], sort=False)
        return out_df
    
    input_df_copy = input_df.copy()
    if not isinstance(input_df_copy, pd.core.frame.DataFrame):
        raise Exception('参数input_df的格式错误，应为pandas.core.frame.DataFrame')

    cols = list(input_df_copy.columns)
    if group_var is not None:
        if group_var not in set(input_df_copy.columns):
            raise Exception('参数group_var取值包含不属于input_df的变量，请检查!')
        input_df_copy[group_var] = input_df_copy[group_var].apply(lambda x: str(x))
    else:
        group_var = 'group'
        input_df_copy[group_var] = 'all'
          
    if group_list is None: 
        group_list = sorted(list(set(input_df_copy[group_var])))
    else:
        group_list = sorted(list(set(group_list)))
    group_list = [str(x) for x in group_list]

    if var_list is not None:
        if not set(var_list).issubset(set(cols)):
            raise Exception('参数var_list取值包含不属于input_df的变量，请检查!')
        else:
            cols = var_list

    output_df = None
    for g in tqdm(group_list):
        seg_input_df_copy = input_df_copy[input_df_copy[group_var] == g][cols]
        stat_df = sub_missing_rate_stat(seg_input_df_copy)[['missing_rate']]
        stat_df[str(g)] = stat_df['missing_rate']
        stat_df = stat_df[[str(g)]]
        if output_df is None:
            output_df = stat_df
        else:
            output_df = pd.concat([output_df, stat_df], axis=1, sort=False)
    output_df['mean'] = output_df.apply(lambda x: x[group_list].mean(), axis=1)
    output_df['std'] = output_df.apply(lambda x: x[group_list].std(), axis=1)
    output_df['temp'] = 0.001
    output_df['cv'] = output_df['std'] / (output_df['temp'] + output_df['mean'])
    output_df = output_df.sort_values(by=['mean'], ascending=0)
    output_df = output_df.drop(['temp'], axis=1)

    # 文件保存
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.csv'):
            raise Exception('参数save_file_path不是csv文件后缀，请检查!')
        output_df.to_csv(save_file_path, encoding='utf-8', index=False)

    return output_df


def cv_grouply_table(edd_table, eval_index='mean'):
    """
    ----------------------------------------------------------------------
    功能: 根据edd_for_continue_var()得到的edd_table, 计算单变量的变异系数(Coefficient of Variation, CV)
    ----------------------------------------------------------------------
    :param edd_table: pd.dataframe, 由edd_for_continue_var()生成
    :param eval_index: str, 评估指标，默认值=‘mean’. 其他取值:
                            [样本总数total、覆盖数cover_cnt、覆盖率cover_rate、
                            均值mean、标准差std、最小值min、分位数25%、分位数50%、分位数75%、最大值max]
    ----------------------------------------------------------------------
    :return cv_table: dataframe, 包含单变量在每份数据集上的mean，以及cv
    ----------------------------------------------------------------------
    示例：
    >>> edd1 = edd_for_continue_var(input_df=df, group_var='group')
    >>> cv_grouply_table(edd_table=edd1, eval_index='mean')
    >>> 
    	seg1	seg2	mean	std	cv	eval_index
    PassengerId	446.223230	445.638235	445.930733	0.413654	0.000928	mean
    Fare	32.592944	31.574228	32.083586	0.720341	0.022451	mean
    Pclass	2.372051	2.205882	2.288967	0.117499	0.051310	mean
    ----------------------------------------------------------------------
    知识：
    标准差与平均数的比值称为变异系数，记为CV
    变异系数又称“标准差率”，是衡量指标中各观测值变异程度的一个统计量。
    当对两个或多个指标的变异程度进行比较时，如果其度量单位与平均数均相同，可以直接利用标准差来比较；
    若二者不同则需计算变异系数，消除数据的绝对大小对变异程度的影响
    ----------------------------------------------------------------------
    """
    if eval_index not in set(['total','cover_cnt','cover_rate','mean','std','min','25%','50%','75%','max']):
        raise Exception('参数eval_index取值不正确!')
        
    feats = sorted(list(set(edd_table.index)))
    cv_table = None
    for var in feats:
        var_edd_df = edd_table[edd_table.index == var][[eval_index]].T
        group_cols = list(edd_table[edd_table.index == var]['group'])
        var_edd_df.columns = group_cols
        var_edd_df['var'] = var
        
        cols = ['var'] + group_cols
        var_edd_df = var_edd_df[cols]
        var_edd_df['mean'] = var_edd_df.apply(lambda row: row[group_cols].mean(), axis=1)
        var_edd_df['std'] = var_edd_df.apply(lambda row: row[group_cols].std(), axis=1)
        var_edd_df['temp'] = 0.001
        var_edd_df['cv'] = var_edd_df['std'] / (var_edd_df['temp'] + var_edd_df['mean'])
        var_edd_df['cv'] = var_edd_df['cv'].apply(lambda x: abs(x))
        if cv_table is None:
            cv_table = var_edd_df
        else:
            cv_table = pd.concat([cv_table, var_edd_df])
            
    cv_table.index = list(cv_table['var'])
    cv_table = cv_table.sort_values(by=['cv'], ascending=1)
    cv_table['eval_index'] = eval_index
    cv_table = cv_table.drop(['var', 'temp'], axis=1)
    
    return cv_table