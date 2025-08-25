# -*- coding:utf-8 -*-
from __future__ import division
__author__ = 'fenghaijie'

import pandas as pd

"""策略评估模块"""


def lift_group_stat(input_df, target_var, group_var=None, hit_rule="status == 'D'"):
    """
    功能：评估命中规则部分的lift
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param target_var: string, 目标变量
    :param group_var: string, 分组变量
    :param hit_rule: string, 圈中规则，格式为"x1 > 1 and x2 > 2"
    ----------------------------------------------------------------------
    :return stat_df: pd.DataFrame, 统计数据
    ----------------------------------------------------------------------
    用法：
    >>> stat = lift_group_stat(input_df=out, 
                               target_var='s1d10',
                               group_var='appl_month',
                               hit_rule="status == 'D'") 
    >>> 
    	group	tot_size	tot_bad_size	tot_bad_rate	hit_size	hit_bad_size	hit_bad_rate	lift
    0	2019-05	16449	1010	0.0614	727	163	0.2242	3.65
    1	2019-06	16500	997	0.0604	709	166	0.2341	3.88
    2	2019-07	16946	910	0.0537	621	118	0.1900	3.54
    3	2019-08	17238	1188	0.0689	740	146	0.1973	2.86
    4	2019-09	17457	1365	0.0782	784	185	0.2360	3.02
    """
    df = input_df.copy()
    df['bad'] = df[target_var]
    
    if group_var is None:
        df['group'] = 'all'
    else:
        df['group'] = df[group_var]
    
    # 总体样本
    tot_grouped = df[['group', 'bad']].groupby(['group'])
    tot_table = pd.DataFrame(tot_grouped.count())
    tot_table.columns = ['tot_size']
    tot_table['tot_bad_size'] = tot_grouped.sum().bad
    tot_table['tot_bad_rate'] = round(tot_table['tot_bad_size'] / tot_table['tot_size'], 4)
    
    # 圈中样本
    hit_df = df.query(hit_rule)
    hit_grouped = hit_df[['group', 'bad']].groupby(['group'])
    hit_table = pd.DataFrame(hit_grouped.count())
    hit_table.columns = ['hit_size']
    hit_table['hit_bad_size'] = hit_grouped.sum().bad
    hit_table['hit_bad_rate'] = round(hit_table['hit_bad_size'] / hit_table['hit_size'], 4)
    
    # 合并结果
    table = pd.concat([tot_table, hit_table], axis=1)
    table['hit_rate'] = round(table['hit_size'] / table['tot_size'], 4)
    table['hit_lift'] = round(table['hit_bad_rate'] / table['tot_bad_rate'], 2)
    table['group'] = list(table.index)
    table = table[['group',
                   'tot_size', 'tot_bad_size', 'tot_bad_rate', 'hit_rate',
                   'hit_size', 'hit_bad_size', 'hit_bad_rate', 'hit_lift']]
    table = table.reset_index(drop=1)
    
    table['rst_size'] = table['tot_size'] - table['hit_size']
    table['rst_bad_size'] = table['tot_bad_size'] - table['hit_bad_size']
    table['rst_bad_rate'] = round(table['rst_bad_size'] / table['rst_size'], 4)
    table['rst_lift'] = round(table['rst_bad_rate'] / table['tot_bad_rate'], 2)
    
    return table


def swapset_table(input_df, score_var1, score_var2, target_var, bins, amount_var=None, weight_var=None):
    """
    功能：新旧模型分交叉后，统计样本量和bad量，进而swap set分析
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param score_var1: string, 模型分数变量1
    :param score_var2: string, 模型分数变量2
    :param target_var: string, 目标变量
    :param bins: int, 分箱数, 典型取值为10或20
    :param amount_var: string, 金额变量, 每笔样本对应金额
    :param weight_var: string, 权重变量, 每个样本代表的原始样本数。
    ----------------------------------------------------------------------
    :return stat_df: pd.DataFrame, 统计数据
    ----------------------------------------------------------------------
    用法：
    >>> stat = swapset_table(input_df=test_df, 
                             score_var1='score1', 
                             score_var2='score2',
                             target_var='bad',
                             bins=10,
                             weight_var='weight')
    >>> stat[['tot']].unstack().fillna(0)
    >>> stat[['bad']].unstack().fillna(0)
    """
    df = input_df.copy()
    
    def get_bucket(score_list, bins, weight_list=None):

        final_score_list = [] # 得到总体样本上的分数

        if weight_list is None:
            final_score_list = score_list
        else:
            for i in range(len(score_list)):
                final_score_list += [score_list[i]] * round(weight_list[i])
    
        bucket_lst = pd.qcut(final_score_list, bins, duplicates='drop') # 等频分箱
        bucket_map = {}
        for x in range(len(bucket_lst)):
            bucket_map[final_score_list[x]] = bucket_lst[x]
            
        return bucket_map
    
    score1_list = list(df.loc[:, score_var1].dropna())
    score2_list = list(df.loc[:, score_var2].dropna())

    if weight_var is None:
        weight_list = None
    else:
        weight_list = list(df.loc[:, weight_var])
    
    # 分箱映射
    bucket1_map = get_bucket(score1_list, bins, weight_list)
    bucket2_map = get_bucket(score2_list, bins, weight_list)
    
    new_score1 = score_var1 + '_bin'
    new_score2 = score_var2 + '_bin'
    df[new_score1] = df[score_var1].apply(lambda x: bucket1_map.get(x) if pd.notna(x) else "missing")
    df[new_score2] = df[score_var2].apply(lambda x: bucket2_map.get(x) if pd.notna(x) else "missing")
    
    if weight_var is None:
        if amount_var is None:
            # 订单口径
            df['tot'] = 1
            df['bad'] = df[target_var]
        else:
            # 金额口径
            df['tot'] = df[amount_var]
            df['bad'] = df[target_var] * df[amount_var]
    else:
        if amount_var is None:
            # 订单口径
            df['tot'] = df[weight_var]
            df['bad'] = df[target_var] * df[weight_var]
        else:
            # 金额口径
            df['tot'] = df[weight_var] * df[amount_var]
            df['bad'] = df[target_var] * df[weight_var] * df[amount_var]

    stat_df = df.groupby([new_score1, new_score2]).sum()[['tot', 'bad']]

    if amount_var is None:
        # 订单口径
        stat_df['tot'] = stat_df['tot'].apply(lambda x: round(x))
        stat_df['bad'] = stat_df['bad'].apply(lambda x: round(x))
    
    return stat_df



def lift_eval(input_df, score='score', cutoff=600, target='bad', sample_rate=0.2):
    """
    ----------------------------------------------------------------------
    功能：计算抽样和原始样本的lift
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param score: string, 信用分
    :param cutoff: float, 分数阈值，大于阈值则pass，否则reject
    :param target: string, 目标变量
    :param sample_rate: float, 好样本欠采样比例，比如0.1
    ----------------------------------------------------------------------
    :return lift_sam: float, 抽样样本上拒绝人群的lift
            lift_ori: float, 原始样本上拒绝人群的lift
    ----------------------------------------------------------------------
    推导过程：
    假设：抽样过程为从原始样本中保留所有bad样本，但good样本采取随机抽样，抽样比例为sample_rate
    抽样样本上拒绝人群的lift为：
    lift_sam = (bad_rj / rj) / ((bad_rj + bad_ps) / (rj + ps))
             = (bad_rj / rj) * ((rj + ps) / (bad_rj + bad_ps))
             = (bad_rj / (bad_rj + bad_ps)) * ((rj + ps) / rj)
             = (bad_rj / (bad_rj + bad_ps)) * (1 + ps / rj)
    其中，
    bad_rj：在相同cutoff下保持不变
    bad_rj + bad_ps：由于保留所有bad样本，保持不变
    
    ps = bad_ps + good_ps = bad_ps + good_ps_origin * sample_rate 
    rj = bad_rj + good_rj = bad_rj + good_rj_origin * sample_rate
    其中，bad_ps和bad_rj在抽样前后不变
    
    因此，原始样本上拒绝人群的lift为：
    lift_ori = (bad_rj / (bad_rj + bad_ps)) * (1 + ps_ori / rj_ori)
    其中,
    ps_ori = bad_ps + good_ps / sample_rate
    rj_ori = bad_rj + good_rj / sample_rate
    
    在抽样样本上，我们可以统计出: ps, rj, bad_ps, bad_rj, good_ps, good_rj
    因此，可以计算出lift_ori
    ----------------------------------------------------------------------
    """
    df = input_df.copy()
    df['result'] = df['score'].apply(lambda x: 'RJ' if x >= cutoff else 'PS')
    rj_df = df[df['result'] == 'RJ']
    ps_df = df[df['result'] == 'PS']
    
    # 拒绝样本好坏样本数
    rj = len(rj_df)
    bad_rj = rj_df[target].sum()
    good_rj = rj - bad_rj
    
    # 通过样本好坏样本数
    ps = len(ps_df)
    bad_ps = ps_df[target].sum()
    good_ps = ps - bad_ps
    
    # 抽样样本上的lift
    lift_sam = (bad_rj / rj) / ((bad_rj + bad_ps) / (rj + ps))
    
    # 原始样本上的lift
    lift_ori = bad_rj / (bad_rj + bad_ps) * \
          (1 + (int(sample_rate * bad_ps) + good_ps) / (int(sample_rate * bad_rj) + good_rj))
        
    return lift_sam, lift_ori