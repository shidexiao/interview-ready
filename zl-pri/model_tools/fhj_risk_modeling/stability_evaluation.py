# -*- coding:utf-8 -*-
from __future__ import division

import math
import numpy as np
import pandas as pd
import logistic_regression as py_lr

"""
模块描述：稳定性评估指标(population stability index ,PSI)
功能包括：
1.psi_for_continue_var      :针对连续变量的population_stability_index基础函数
2.psi_for_discrete_var      :针对离散变量的population_stability_index基础函数
3.psi_grouply_table         :按样本集[INS/OOS/OOT]或时间窗分组计算变量列表中所有变量的PSI
4.coefficent_stability_test :多次训练LR模型, 得到各变量权重系数估计结果, 分析点估计稳定性。
5.coefficent_stability_analysis: 根据coefficent_stability_test()结果可视化分析
"""

def psi_for_continue_var(expected_array, actual_array, bins=10, bucket_type='bins', detail=False, save_file_path=None):
    '''
    ----------------------------------------------------------------------
    功能: 计算连续型变量的群体性稳定性指标（population stability index ,PSI）
    ----------------------------------------------------------------------
    :param expected_array: numpy array of original values，基准组
    :param actual_array: numpy array of new values, same size as expected，比较组
    :param bins: number of percentile ranges to bucket the values into，分箱数, 默认为10
    :param bucket_type: string, 分箱模式，'bins'为等距均分，'quantiles'为按等频分箱
    :param detail: bool, 取值为True时输出psi计算的完整表格, 否则只输出最终的psi值
    :param save_file_path: string, csv文件保存路径. 默认值=None. 只有当detail=Ture时才生效.
    ----------------------------------------------------------------------
    :return psi_value: 
            当detail=False时, 类型float, 输出最终psi计算值;
            当detail=True时, 类型pd.DataFrame, 输出psi计算的完整表格。最终psi计算值 = list(psi_value['psi'])[-1]
    ----------------------------------------------------------------------
    示例：
    >>> psi_for_continue_var(expected_array=df['score'][:400],
                             actual_array=df['score'][401:], 
                             bins=5, bucket_type='bins', detail=0)
    >>> 0.0059132756739701245
    ------------
    >>> psi_for_continue_var(expected_array=df['score'][:400],
                             actual_array=df['score'][401:], 
                             bins=5, bucket_type='bins', detail=1)
    >>>
    	score_range	expecteds	expected(%)	actucals	actucal(%)	ac - ex(%)	ln(ac/ex)	psi	max
    0	[0.021,0.2095]	120.0	30.00	152.0	31.02	1.02	0.033434	0.000341	
    1	(0.2095,0.398]	117.0	29.25	140.0	28.57	-0.68	-0.023522	0.000159	
    2	(0.398,0.5865]	81.0	20.25	94.0	19.18	-1.07	-0.054284	0.000577	<<<<<<<
    3	(0.5865,0.7751]	44.0	11.00	55.0	11.22	0.22	0.019801	0.000045	
    4	(0.7751,0.9636]	38.0	9.50	48.0	9.80	0.30	0.031087	0.000091	
    5	>>> summary	400.0	100.00	489.0	100.00	NaN	NaN	0.001214	<<< result
    ----------------------------------------------------------------------
    知识:
    公式： psi = sum(（实际占比-预期占比）* ln(实际占比/预期占比))
    一般认为psi小于0.1时候变量稳定性很高，0.1-0.25一般，大于0.25变量稳定性差，建议重做。
    相对于变量分布(EDD)而言, psi是一个宏观指标, 无法解释两个分布不一致的原因。但可以通过观察每个分箱的sub_psi来判断。
    ----------------------------------------------------------------------
    '''
    import math
    import numpy as np
    import pandas as pd
    
    expected_array = pd.Series(expected_array).dropna()
    actual_array = pd.Series(actual_array).dropna()

    try:
        if isinstance(list(expected_array)[0], str) or isinstance(list(actual_array)[0], str):
            return np.nan
    except:
        return np.nan
        
    """step1: 确定分箱间隔"""
    def scale_range(input_array, scaled_min, scaled_max):
        '''
        ----------------------------------------------------------------------
        功能: 对input_array线性放缩至[scaled_min, scaled_max]
        ----------------------------------------------------------------------
        :param input_array: numpy array of original values, 需放缩的原始数列
        :param scaled_min: float, 放缩后的最小值
        :param scaled_min: float, 放缩后的最大值
        ----------------------------------------------------------------------
        :return input_array: numpy array of original values, 放缩后的数列
        ----------------------------------------------------------------------
        '''
        input_array += -np.min(input_array) # 此时最小值放缩到0
        if scaled_max == scaled_min:
            raise Exception('放缩后的数列scaled_min = scaled_min, 值为{}, 请检查expected_array数值！'.format(scaled_max))
        scaled_slope = np.max(input_array) * 1.0 / (scaled_max - scaled_min)
        input_array /= scaled_slope
        input_array += scaled_min
        return input_array
    
    # 异常处理，所有取值都相同时, 说明该变量是常量, 返回999999
    if np.min(expected_array) == np.max(expected_array):
        return 999999
    
    breakpoints = np.arange(0, bins + 1) / (bins) * 100 # 等距分箱百分比
    if 'bins' == bucket_type:        # 等距分箱
        breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
    elif 'quantiles' == bucket_type: # 等频分箱
        breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])
 
    """step2: 统计区间内样本占比"""
    def generate_counts(arr, breakpoints):
        '''
        ----------------------------------------------------------------------
        功能: Generates counts for each bucket by using the bucket values 
        ----------------------------------------------------------------------
        :param arr: ndarray of actual values
        :param breakpoints: list of bucket values
        ----------------------------------------------------------------------
        :return cnt_array: counts for elements in each bucket, length of breakpoints array minus one
        :return score_range_array: 分箱区间
        ----------------------------------------------------------------------
        '''
        def count_in_range(arr, low, high, start):
            '''
            ----------------------------------------------------------------------
            功能: 统计给定区间内的样本数(Counts elements in array between low and high values)
            ----------------------------------------------------------------------
            :param arr: ndarray of actual values
            :param low: float, 左边界
            :param high: float, 右边界
            :param start: bool, 取值为Ture时，区间闭合方式[low, high],否则为(low, high]
            ----------------------------------------------------------------------
            :return cnt_in_range: int, 给定区间内的样本数
            ----------------------------------------------------------------------
            '''
            if start:
                cnt_in_range = len(np.where(np.logical_and(arr >= low, arr <= high))[0])
            else:
                cnt_in_range = len(np.where(np.logical_and(arr > low, arr <= high))[0])
            return cnt_in_range
 
        cnt_array = np.zeros(len(breakpoints) - 1)
        score_range_array = [''] * (len(breakpoints) - 1)
        for i in range(1, len(breakpoints)):
            cnt_array[i-1] = count_in_range(arr, breakpoints[i-1], breakpoints[i], i==1)
            if 1 == i:
                score_range_array[i-1] = '[' + str(round(breakpoints[i-1], 4)) + ',' + str(round(breakpoints[i], 4)) + ']'
            else:   
                score_range_array[i-1] = '(' + str(round(breakpoints[i-1], 4)) + ',' + str(round(breakpoints[i], 4)) + ']'
                                                                                
        return (cnt_array, score_range_array)
 
    expected_cnt = generate_counts(expected_array, breakpoints)[0]
    expected_percents = expected_cnt / len(expected_array)
    actual_cnt = generate_counts(actual_array, breakpoints)[0]
    actual_percents = actual_cnt / len(actual_array)
    delta_percents = actual_percents - expected_percents
    score_range_array = generate_counts(expected_array, breakpoints)[1]
                                                                               
    """step3: 区间放缩"""
    def sub_psi(e_perc, a_perc):
        '''
        ----------------------------------------------------------------------
        功能: 计算单个分箱内的psi值。Calculate the actual PSI value from comparing the values.
             Update the actual value to a very small number if equal to zero
        ----------------------------------------------------------------------
        :param e_perc: float, 期望占比
        :param a_perc: float, 实际占比
        ----------------------------------------------------------------------
        :return value: float, 单个分箱内的psi值
        ----------------------------------------------------------------------
        '''
        if a_perc == 0: # 实际占比
            a_perc = 0.001
        if e_perc == 0: # 期望占比
            e_perc = 0.001
        value = (e_perc - a_perc) * np.log(e_perc * 1.0 / a_perc)
        return value
    
    """step4: 得到最终稳定性指标"""
    sub_psi_array = [sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))]
    if detail:
        psi_value = pd.DataFrame()
        psi_value['score_range'] = score_range_array
        psi_value['expecteds'] = expected_cnt
        psi_value['expected(%)'] = expected_percents * 100
        psi_value['actucals'] = actual_cnt
        psi_value['actucal(%)'] = actual_percents * 100
        psi_value['ac - ex(%)'] = delta_percents * 100
        psi_value['actucal(%)'] = psi_value['actucal(%)'].apply(lambda x: round(x, 2))
        psi_value['ac - ex(%)'] = psi_value['ac - ex(%)'].apply(lambda x: round(x, 2))
        psi_value['ln(ac/ex)'] = psi_value.apply(lambda row: np.log((row['actucal(%)'] + 0.001) \
                                                                  / (row['expected(%)'] + 0.001)), axis=1)
        psi_value['psi'] = sub_psi_array
        flag = lambda x: '<<<<<<<' if x == psi_value.psi.max() else ''
        psi_value['max'] = psi_value.psi.apply(flag)
        psi_value = psi_value.append([{'score_range':'>>> summary', 
                                       'expecteds': sum(expected_cnt),
                                       'expected(%)':100, 
                                       'actucals': sum(actual_cnt),
                                       'actucal(%)':100, 
                                       'ac - ex(%)': np.nan,
                                       'ln(ac/ex)': np.nan,
                                       'psi': np.sum(sub_psi_array),
                                       'max':'<<< result'}], ignore_index=True)
        if save_file_path:
            if not isinstance(save_file_path, str):
                raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
            elif save_file_path.endswith('.csv'):
                raise Exception('参数save_file_path不是csv文件后缀，请检查!')
            psi_value.to_csv(save_file_path, encoding='utf-8', index=1)
    else:
        psi_value = np.sum(sub_psi_array)
 
    return psi_value


def psi_for_discrete_var(expected_array, actual_array, max_bins=100, detail=False, save_file_path=None):
    '''
    ----------------------------------------------------------------------
    功能: 计算离散型变量的群体性稳定性指标（population stability index ,PSI）
    ----------------------------------------------------------------------
    :param expected_array: numpy array of original values，基准组
    :param actual_array: numpy array of new values, same size as expected，比较组
    :param max_bins: int, 最大允许的离散型变量分箱数，若大于max_bins,则将抛出异常
    :param detail: bool, 取值为True时输出psi计算的完整表格, 否则只输出最终的psi值
    :param save_file_path: string, csv文件保存路径. 默认值=None. 只有当detail=Ture时才生效.
    ----------------------------------------------------------------------
    :return psi_value: 
            当detail=False时, 类型float, 输出最终psi计算值;
            当detail=True时, 类型pd.DataFrame, 输出psi计算的完整表格。最终psi计算值 = list(psi_value['psi'])[-1]
    ----------------------------------------------------------------------
    示例：
    >>> psi_for_discrete_var(expected_array=df[df['group'] == 'seg1']['Age'], 
                             actual_array=df[df['group'] == 'seg2']['Age'], 
                             detail=1)
    >>> 
    score_range	expected(%)	actucal(%)	ac - ex(%)	ln(ac/ex)	psi	max
    5	1	1.27	0.00	-1.27	-7.147559	0.096958	
    6	2	1.81	0.00	-1.81	-7.501634	0.145003	
    7	3	1.09	0.00	-1.09	-6.994850	0.081423	
    ----------
    >>> psi_for_discrete_var(expected_array=df[df['group'] == 'seg1']['Age'], 
                             actual_array=df[df['group'] == 'seg2']['Age'], 
                             detail=0)
    >>> 18.36553668288402
    ----------------------------------------------------------------------
    知识:
    公式： psi = sum(（实际占比-预期占比）* ln(实际占比/预期占比))
    一般认为psi小于0.1时候变量稳定性很高，0.1-0.25一般，大于0.25变量稳定性差，建议重做。
    相对于变量分布(EDD)而言, psi是一个宏观指标, 无法解释两个分布不一致的原因。但可以通过观察每个分箱的sub_psi来判断。
    ----------------------------------------------------------------------
    '''
    """step1: 类型判断"""
    expected_array = pd.Series(expected_array).dropna()
    actual_array = pd.Series(actual_array).dropna()
    expected_bins = set(np.unique(expected_array))
    actual_bins = set(np.unique(actual_array))
    union_bins = sorted(list(expected_bins.union(actual_bins))) # 并集bins
    
    if len(union_bins) > max_bins:
        raise Exception("输入数据expected_array和actual_array" + 
                        "类别超过max_bins, 实际有{}个类别, 请确认是否是离散型变量!".format(len(union_bins)))
 
    """step2: 统计分箱内样本占比"""
    def generate_counts(arr):
        '''
        ----------------------------------------------------------------------
        功能: Generates counts for each bucket by using the bucket values 
        ----------------------------------------------------------------------
        :param arr: ndarray of actual values
        ----------------------------------------------------------------------
        :return cnt_array: counts for elements in each bucket, length of breakpoints array minus one
        ----------------------------------------------------------------------
        '''
        cnt_dict = {}
        for k in arr:
            cnt_dict[k] = cnt_dict.get(k, 0) + 1
        return cnt_dict
    
    expected_cnt_dict = generate_counts(expected_array)
    actual_cnt_dict = generate_counts(actual_array)
    
    expected_cnt = [expected_cnt_dict.get(b, 0) for b in union_bins]
    expected_percents = [expected_cnt_dict.get(b, 0.001) * 1.0 / len(expected_array) for b in union_bins]
    actual_cnt = [actual_cnt_dict.get(b, 0) for b in union_bins]
    actual_percents = [actual_cnt_dict.get(b, 0.001) * 1.0 / len(actual_array) for b in union_bins]
    delta_percents = [actual_percents[i] - expected_percents[i] for i in range(len(union_bins))]
    score_range_array = union_bins
                                                                               
    """step3: 区间放缩"""
    def sub_psi(e_perc, a_perc):
        '''
        ----------------------------------------------------------------------
        功能: 计算单个分箱内的psi值。Calculate the actual PSI value from comparing the values.
             Update the actual value to a very small number if equal to zero
        ----------------------------------------------------------------------
        :param e_perc: float, 期望占比
        :param a_perc: float, 实际占比
        ----------------------------------------------------------------------
        :return value: float, 单个分箱内的psi值
        ----------------------------------------------------------------------
        '''
        if a_perc == 0: # 实际占比
            a_perc = 0.001
        if e_perc == 0: # 期望占比
            e_perc = 0.001
        value = (e_perc - a_perc) * np.log(e_perc * 1.0 / a_perc)
        return value
    
    """step4: 得到最终稳定性指标"""
    sub_psi_array = [sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))]
    if detail:
        psi_value = pd.DataFrame()
        psi_value['score_range'] = score_range_array
        psi_value['expecteds'] = expected_cnt
        psi_value['expected(%)'] = expected_percents * 100
        psi_value['actucals'] = actual_cnt
        psi_value['actucal(%)'] = actual_percents * 100
        psi_value['ac - ex(%)'] = delta_percents * 100
        psi_value['actucal(%)'] = psi_value['actucal(%)'].apply(lambda x: round(x, 2))
        psi_value['ac - ex(%)'] = psi_value['ac - ex(%)'].apply(lambda x: round(x, 2))
        psi_value['ln(ac/ex)'] = psi_value.apply(lambda row: np.log((row['actucal(%)'] + 0.001) \
                                                                  / (row['expected(%)'] + 0.001)), axis=1)
        psi_value['psi'] = sub_psi_array
        flag = lambda x: '<<<<<<<' if x == psi_value.psi.max() else ''
        psi_value['max'] = psi_value.psi.apply(flag)
        psi_value = psi_value.append([{'score_range':'>>> summary', 
                                       'expecteds': sum(expected_cnt),
                                       'expected(%)':100, 
                                       'actucals': sum(actual_cnt),
                                       'actucal(%)':100, 
                                       'ac - ex(%)': np.nan,
                                       'ln(ac/ex)': np.nan,
                                       'psi': np.sum(sub_psi_array),
                                       'max':'<<< result'}], ignore_index=True)
        if save_file_path:
            if not isinstance(save_file_path, str):
                raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
            elif save_file_path.endswith('.csv'):
                raise Exception('参数save_file_path不是csv文件后缀，请检查!')
            psi_value.to_csv(save_file_path, encoding='utf-8', index=1)
    else:
        psi_value = np.sum(sub_psi_array)
 
    return psi_value


def psi_grouply_table(input_df, group_var,
                      benchmark_list, compare_list=None, 
                      c_var_list=None, d_var_list=None, 
                      save_file_path=None, progress=False):
    '''
    ----------------------------------------------------------------------
    功能：按样本集[INS/OOS/OOT]或时间窗分组计算变量列表中所有变量的PSI
    1) psi_for_continue_var(expected_arr, actucal_arr, bins=10, bucket_type='bins')
    2) psi_for_discrete_var(expected_arr, actucal_arr, detail=False)
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param group_var: string, 分组依据，如'month'，'week'，'group'
    :param benchmark_list: list，基准组名，如['201709', '201710']，['INS', 'OOT']
    :param compare_list: list，比较组，如['201709', '201710']。默认值=None，则取除了benchmark_list外的所有取值。
    :param c_var_list: list, 需要统计psi的<连续型变量>列表，默认值=None，对所有数值型变量(int、float)统计。必须要在input_df中，否则抛异常。
    :param d_var_list: list, 需要统计psi的<离散型变量>列表，默认值=None，则不做统计。必须要在input_df中，否则抛异常。
    :param save_file_path: string, csv文件保存路径. 默认值=None
    :param progress: bool, 取值为False时不显示计算进度, 否则显示进度, 便于debug.
    ----------------------------------------------------------------------
    :return output_df: pd.DataFrame, 返回结果
    ----------------------------------------------------------------------
    示例：
    >>> psi_grouply_table(input_df=df, 
                  group_var='group',
                  benchmark_list=['seg1'], 
                  compare_list=None, 
                  c_var_list=['Age','Fare', 'score'],
                  d_var_list=['Age', 'Cabin'])
    >>> 
    	seg2	mean	max	benchmark
    Fare	0.0349	0.0349	0.0349	seg1
    score	0.3857	0.3857	0.3857	seg1
    Age	4.7407	4.7407	4.7407	seg1
    ----------------------------------------------------------------------
    '''
    input_df_copy = input_df.copy()
    cols = list(input_df_copy.columns)
    if group_var not in set(cols):
        raise Exception('参数group_var取值包含不属于input_df的变量，请检查!')
        
    """防御性处理: 取值范围和类型检查"""  
    cols_type_dict = dict(input_df_copy.dtypes)
    str_cols = [x for x in cols if cols_type_dict[x] == object]
        
    if d_var_list is not None:
        if not set(d_var_list).issubset(set(cols)):
            raise Exception('参数d_var_list取值包含不属于input_df的变量，请检查!')
    else:
        d_var_list = []
        
    if c_var_list is not None:
        if not set(c_var_list).issubset(set(cols)):
            raise Exception('参数c_var_list取值包含不属于input_df的变量，请检查!')
        c_var_list = list(set(c_var_list) - set(str_cols + d_var_list)) # 自动过滤string型和离散型变量
    else:
        c_var_list = list(set(cols) - set(str_cols + d_var_list))
         
    if compare_list is None:
        compare_list = sorted(list(set(input_df_copy[group_var]) - set(benchmark_list)))
        
    """分组计算psi"""    
    output_df = pd.DataFrame()
    calculated_var_list = c_var_list + d_var_list
    expected_df = input_df_copy[input_df_copy[group_var].isin(benchmark_list)]
    for cp in compare_list:
        sub_output_df = pd.DataFrame(index=calculated_var_list, columns=[str(cp)])
        actucal_df = input_df_copy[input_df_copy[group_var] == cp]
        # 连续型变量
        for var in c_var_list:
            if progress: print('> ' + var)
            expected_arr = expected_df[var].dropna(axis=0)
            actucal_arr  = actucal_df[var].dropna(axis=0)
            psi_score = psi_for_continue_var(expected_arr, actucal_arr, bins=10, bucket_type='bins')
            sub_output_df.loc[var, str(cp)] = round(psi_score, 4)
        # 离散型变量   
        for var in d_var_list:
            if progress: print('> ' + var)
            expected_arr = expected_df[var].dropna(axis=0)
            actucal_arr  = actucal_df[var].dropna(axis=0)
            psi_score = psi_for_discrete_var(expected_arr, actucal_arr, max_bins=100, detail=False)
            sub_output_df.loc[var, str(cp)] = round(psi_score, 4)
            
        output_df = pd.concat([output_df, sub_output_df], axis=1, sort=False)
        
    compare_list = [str(x) for x in compare_list]
    benchmark_list = [str(x) for x in benchmark_list]
    output_df['mean'] = output_df.apply(lambda row: round(row[compare_list].mean(), 4), axis=1)
    output_df['max']  = output_df.apply(lambda row: round(row[compare_list].max(), 4), axis=1)
    output_df['benchmark'] = '+'.join(benchmark_list)
    output_df = output_df.sort_values(by='mean', ascending=1)
    
    """文件保存"""
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.csv'):
            raise Exception('参数save_file_path不是csv文件后缀，请检查!')
        output_df.to_csv(save_file_path, encoding='utf-8', index=1)
         
    return output_df


def coefficent_stability_test(input_df, var_list, target_var, group_var=None, group_list=None, save_file_path=None):
    """
    ----------------------------------------------------------------------
    功能：多次训练LR模型, 得到各变量权重系数估计结果, 分析点估计稳定性。
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param var_list: list, 变量列表
    :param target_var: str, 目标变量
    :param group_var: string, 分组依据，如按[组，月，周]。默认值=None, 对整个数据集统计。示例: 'apply_month'
    :param group_list: list, 分组元素取值, 若为None，则默认为所有取值。示例: [201701, 201702]
    :param save_file_path: string, csv文件保存路径. 默认值=None
    ----------------------------------------------------------------------
    :return output_df: pd.DataFrame, 返回结果
    ----------------------------------------------------------------------
    示例：
    >>> coefficent_stability_df = coefficent_stability_test(develop_data, final_cols, target_var)
    >>> 
    	参数估计	标准误	z值	wald卡方	p值	置信下界	置信上界	group	p值>0.05	系数反转
    const	-3.780529	0.037782	-100.060479	10012.099518	0.000000e+00	-3.854582	-3.706477	INS		
    tongdun_score_woe	-0.878744	0.053468	-16.435077	270.111760	1.072881e-60	-0.983539	-0.773949	INS		
    shangjia_score_woe	-0.732425	0.069958	-10.469494	109.610306	1.192788e-25	-0.869541	-0.595310	INS		
    taobao_score_woe	-0.721556	0.102378	-7.047974	49.673940	1.815414e-12	-0.922213	-0.520899	INS		
    creditcard_score_woe	-0.687103	0.064645	-10.628925	112.974043	2.186182e-26	-0.813804	-0.560402	INS		
    carrier_score_woe	-0.544060	0.080692	-6.742469	45.460892	1.557172e-11	-0.702212	-0.385908	INS		
    ----------------------------------------------------------------------
    """
    input_df_copy = input_df.copy()
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
    
    output_df = None
    for group_i in group_list:
        print('-' * 50)
        gp_df = input_df_copy [input_df_copy ['group'] == group_i]
        logit_instance, logit_model, logit_result, logit_result_0, varlist \
            = py_lr.logistic_reg(X=gp_df[var_list], y=gp_df[target_var], stepwise='BS', sls=0.05)
        desc, params, evaluate, quality = py_lr.logit_output(logit_instance, logit_model, logit_result, logit_result_0)
        params = params.sort_values(by=['参数估计'], ascending=1)
        params['group'] = group_i
        params['p值>0.05'] = params['p值'].apply(lambda x: '<---' if x > 0.05 else "")
        params['系数反转'] = params['参数估计'].apply(lambda x: '<<<<' if x > 0 else "")
        if output_df is None:
            output_df = params
        else:
            output_df = pd.concat([output_df, params], axis=0, sort=False)
            
    """文件保存"""
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.csv'):
            raise Exception('参数save_file_path不是csv文件后缀，请检查!')
        output_df.to_csv(save_file_path, encoding='utf-8', index=1)
            
    return output_df


def coefficent_stability_analysis(input_df, group_var='group', show_data=0, save_file_path=None):
    """
    ----------------------------------------------------------------------
    功能：多次训练LR模型。根据coefficent_stability_test()函数结果可视化
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据. 由coefficent_stability_test()函数生成
    :param group_var: str, 分组变量
    :param target_var: str, 目标变量
    ----------------------------------------------------------------------
    :return output_df: pd.DataFrame, 返回结果
    ----------------------------------------------------------------------
    示例：
    >>> coefficent_stability_analysis(coefficent_stability_df, group_var='group')
    >>>
    ----------------------------------------------------------------------
    """
    group_list = list(set(input_df[group_var]) - set(['OOT1+OOT2', 'INS+OOS']))
    var_list = sorted(list(set(input_df.index) - set(['const'])))
    
    plt.figure(figsize=[16, 6])
    plt.title('LR Coefficent Stability') 
    plt.xlabel('Feature')
    plt.ylabel('Param Evaluate(%)')
    
    for g in group_list:
        df = input_df[input_df[group_var] == g]
        bin_num = len(var_list)
        x = [x[:-4] for x in var_list]
        y = [list(df[df.index == x]['参数估计'])[0] for x in var_list]
        plt.plot(x, y, label=g, linewidth=3, marker='o')
        if show_data:
            for a, b in zip(x, y):
                plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    plt.legend(groups)
    
    """step4: 保存输出"""
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.jpg'):
            raise Exception('参数save_file_path不是jpg文件后缀，请检查!')
        plt.savefig(save_file_path)
        print('成功保存至:{}'.format(save_file_path))
        
    return plt
