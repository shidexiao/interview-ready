# -*- coding:utf-8 -*-
__author__ = 'fenghaijie'

import math
import numpy as np
import pandas as pd

"""
模块描述：样本分群（Segment）
功能包括：
1.percentile_segment()
2.
"""


def percentile_segment(input_df, target_var, var, percentile=50, seg_point=None):
    '''
    功能：适用于分类问题，根据某个变量的分位数来分成2个群体，并查看正负样本率等指标
    ----------
    :param input_df: dataframe
    :param target_var: string, 目标变量
    :param var: string, 分群变量
    :param percentile: int, [0, 100]，分位数
    :param seg_point: float, 分群点
    ----------     
    :return output_output_df: 两个分群，以及正负样本数，正样本率等信息
    ----------
    示例：
    >>> percentile_segment(df, target, var='Fare', percentile=50)
    >>> 
    
    '''
    input_df = input_df.loc[input_df[target_var].isin([0,1]), :]
    if seg_point is None:
        breakpoint = np.percentile(input_df[var], percentile)
    else:
        breakpoint = seg_point
    input_df.loc[(input_df[var] >  breakpoint), 'seg'] = '高'
    input_df.loc[(input_df[var] <= breakpoint), 'seg'] = '低'
    temp = input_df.groupby(['seg', target_var]).size()
    temp = pd.DataFrame(temp, columns=['cnt'])
    cnt_lst = temp['cnt'].values
    
    output_df = pd.DataFrame()
    output_df['seg'] = ['<= ' + str(breakpoint), '> ' + str(breakpoint)]
    output_df['dummy'] = 100
    output_df['bads']  = [cnt_lst[1], cnt_lst[3]]
    output_df['goods'] = [cnt_lst[0], cnt_lst[2]]
    output_df['total'] = output_df['bads'] + output_df['goods']
    output_df['samples(%)'] = output_df['total'].apply(lambda x: x * 100 / len(input_df))  
    output_df['bads(%)'] = output_df['bads'] * output_df['dummy'] / (output_df['bads'] + output_df['goods'])
    output_df['all_bads(%)'] = input_df[target_var].mean()
    output_df['all_bads(%)'] = output_df['all_bads(%)'].apply(lambda x: x * 100)
    output_df['samples(%)'] = output_df['samples(%)'].apply('{0:.2f}'.format)
    output_df['bads(%)'] = output_df['bads(%)'].apply('{0:.2f}'.format)
    output_df['all_bads(%)'] = output_df['all_bads(%)'].apply('{0:.2f}'.format) 
    del output_df['dummy']
    
    return output_df

