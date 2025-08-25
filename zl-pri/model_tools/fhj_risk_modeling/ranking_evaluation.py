# -*- coding:utf-8 -*-
from __future__ import division
__author__ = 'fenghaijie / hjfeng0630@qq.com'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
模块描述：模型排序性指标（Ranking）
功能包括：
1.model_ranking_eval  :根据ks_table，可视化观察模型排序性
1) 放款层bad_rate
2) 放款层lift
3) 放款层log(odds)
4) 申请层reject_rate
"""


def model_ranking_eval(ks_table, ranking_index, group_var=None, show_data=0, save_file_path=None):
    """
    ----------------------------------------------------------------------
    功能：根据ks_table，可视化观察模型排序性
    ----------------------------------------------------------------------
    :param ks_table: dataframe, 至少包含字段['bad_rate', 'lift', 'reject_rate', 'group']
    :param ranking_index: str, 排序性指标，取值'bad_rate', 'lift','reject_rate', 'ln_odds'
    :param group_var: str, 分组变量，取值'group', 'apply_month'
    :param show_data: int, 是否展示数据
    :param save_file_path: string, jpg文件保存路径. 默认值=None
    ----------------------------------------------------------------------
    :return plt图像
    ----------------------------------------------------------------------
    内置默认颜色, 按顺序为: ['r'(red), 'b'(blue), 'g'(green), 'y'(yellow), 'c'(cyan), 
                          'm'(magenta), 'k'(black), 'orange', 'gray', 'peru']
    ----------------------------------------------------------------------
    示例：
    >>> model_ranking_eval(ks_table_df, ranking_index='ln_odds', show_data=1)
    ----------------------------------------------------------------------
    """
    ks_table_copy = ks_table.copy()
    ks_table_copy = ks_table_copy.fillna(0.01)
    ks_table_copy['ln_odds'] = ks_table_copy['odds'].apply( \
        lambda x: np.log(float(x)) if x != 'inf' else np.log(1000))
    if group_var is None:
        group_var = 'group'
        ks_table_copy[group_var] = 'all'
    group_list = sorted(list(set(ks_table_copy[group_var]) - set(['OOT1+OOT2', 'INS+OOS'])))
    group_cnt = len(group_list)
    
    if ranking_index not in set(ks_table_copy.columns):
        raise Exception('参数ranking_index取值包含不属于ks_table的变量，请检查!')
    
    plt.figure(figsize=[16, 6])
    min_score_list = list(ks_table_copy['min_score'])
    asc_flag = 0 if min_score_list[0] >= min_score_list[1] else 1 # 升序降序标识
    ks_sign = list(ks_table_copy['ks_sign'])[0]
    
    if asc_flag == 1:
        if ks_sign == '-':
            print('> score类似<欺诈分>, 取值越大，P(bad)越大.')
            plt.xlabel('bucket(Fraud Score, score low -> high)')
        else:
            print('> score类似<信用分>, 取值越大，P(bad)越小.')
            plt.xlabel('bucket(Credit Score, score low -> high)')
    else:
        if ks_sign == '-':
            print('> score类似<信用分>, 取值越大，P(bad)越小.')
            plt.xlabel('bucket(Credit Score, score high -> low)')
        else: 
            print('> score类似<欺诈分>, 取值越大，P(bad)越大.')
            plt.xlabel('bucket(Fraud Score, score high -> low)')
            
    if ranking_index == 'bad_rate':
        plt.title('Bad Rate')
        plt.ylabel('bad rate(%)')
    elif ranking_index == 'lift':
        plt.title('Lift Chart')
        plt.ylabel('lift')
    elif ranking_index == 'reject_rate':
        plt.title('Reject Rate')
        plt.ylabel('reject rate(%)')
    elif ranking_index == 'ln_odds':
        plt.title('Log Odds')
        plt.ylabel('ln_odds(bad/good)')
    else:
        raise Exception('参数ranking_index取值包含不属于ks_table的变量，请检查!')
     
    color_list = ['r', 'b', 'g', 'y', 'c', 'm', 'y', 'k', 'orange', 'gray', 'peru']
    linestyle_list = ['-', ':', '-.']
        
    valid_group_list = []
    for i in range(group_cnt):
        g = group_list[i]
        group_ks_table = ks_table_copy[ks_table_copy[group_var] == g]
        bucket_num = group_ks_table.shape[0]
        x = [str(x) for x in range(1, bucket_num + 1)]
        
        rank_list = list(group_ks_table[ranking_index])
        valid = False
        y = None
        if rank_list[0] != 'nan%':
            valid = True
            valid_group_list.append(g)
            if ranking_index != 'ln_odds':
                y = [float(x[:-1]) for x in rank_list]
            else:
                y = [round(x, 4) for x in rank_list]
        
        color = color_list[int(i % len(color_list))]
        linestyle = linestyle_list[0] if i < len(color_list) else linestyle_list[1]
            
        if valid:
            plt.plot(x, y, label=g, linewidth=3, marker='o', color=color, linestyle=linestyle)
        
        if show_data and valid:
            for a, b in zip(x, y):
                plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
                
    plt.legend(labels=valid_group_list)
    
    """step4: 保存输出"""
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.jpg'):
            raise Exception('参数save_file_path不是jpg文件后缀，请检查!')
        plt.savefig(save_file_path)
        print('成功保存至:{}'.format(save_file_path))
    
    return plt