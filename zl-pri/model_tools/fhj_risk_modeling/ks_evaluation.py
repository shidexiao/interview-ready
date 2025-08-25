# -*- coding:utf-8 -*-
from __future__ import division

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import ks_2samp

"""
模块描述：模型区分度KS评价指标（Kolmogorov-Smirnov, KS）
功能包括：
1. ks_compute              :利用scipy库函数计算ks指标
2. ks_grouply_calculate    :利用scipy库函数计算每组中各变量的KS指标
3. ks_table                :生成ks_table, 可观察每个bin内的正负样本数，几率odds，lift和ks
4. ks_grouply_table        :分组计算ks_table
5. ks_plot                 :ks计算可视化图
6. ks_table_plot           :读取ks_table()生成的ks_table可视化绘制KS曲线
7. var_marginal_ks         :单变量边际ks
"""

def ks_compute(proba_arr, target_arr):
    '''
    ----------------------------------------------------------------------
    功能：利用scipy库函数计算ks指标
    ----------------------------------------------------------------------
    :param proba_arr:  numpy array of shape (1,), predicted probability of the sample being positive
    :param target_arr: numpy array of shape (1,), 取值为0或1
    ----------------------------------------------------------------------
    :return ks_value: float, ks score estimation
    ----------------------------------------------------------------------
    示例：
    >>> ks_compute(proba_arr=df['score'], target_arr=df[target])
    >>> 0.5262199213881699
    ----------------------------------------------------------------------
    '''
    get_ks = lambda proba_arr, target_arr: ks_2samp(proba_arr[target_arr == 1], proba_arr[target_arr == 0]).statistic
    ks_value = get_ks(proba_arr, target_arr)
    return ks_value


def ks_grouply_calculate(input_df, target_var, group_var, group_list=None, var_list=None, save_file_path=None):
    '''
    ----------------------------------------------------------------------
    功能: 利用scipy库函数计算每组中各变量的KS指标
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param target_var: str, 目标变量,会自动过滤出0和1。示例: 's1d30'。
    :param group_var: str, 分组依据，如按组（ins/oos/oot），月，周。示例: ['apply_month']
    :param group_list: list, 分组元素取值, 若为None，则默认为所有取值。示例: [201701, 201702]
    :param var_list: list, 需要统计缺失率的变量列表，默认值=None，对所有数值型变量统计。必须要在input_df中，否则抛异常。
    :param save_file_path: string, csv文件保存路径. 默认值=None
    ----------------------------------------------------------------------
    :return output_df: pd.DataFrame, 每组每个变量的ks值，并按ks_mean降序排列显示
    ----------------------------------------------------------------------
    示例:
    >>> ks_grouply_calculate(input_df=df, target_var=target, 
                     group_var='group', group_list=None, var_list=None, save_file_path=None)
    >>>
    	seg1	seg2	mean	std	cv	target
    Survived	1.0000	1.0000	1.00000	0.000000	0.000000	Survived
    score	0.5534	0.5459	0.54965	0.005303	0.009476	Survived
    Pclass	0.3500	0.3069	0.32845	0.030476	0.090047	Survived
    ----------------------------------------------------------------------
    '''
    """step1: 防御性处理"""
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
            
    cols_type_dict = dict(input_df_copy.dtypes)
    str_cols = [x for x in cols if cols_type_dict[x] == object]

    if var_list is None:
        var_list = list(set(cols) - set(str_cols))
    else:
        if not set(var_list).issubset(set(cols)):
            raise Exception('参数var_list取值包含不属于input_df_copy的变量，请检查!')
        var_list = list(set(var_list) - set(str_cols))
            
    """step2: 分组计算ks"""           
    output_df = pd.DataFrame()
    for gp in group_list: 
        gp_input_df  = input_df_copy[(input_df_copy[group_var] == gp) & (input_df_copy[target_var].isin([0,1]))]
        pos_input_df = gp_input_df[gp_input_df[target_var] == 1]
        neg_input_df = gp_input_df[gp_input_df[target_var] == 0]
        
        sub_output_df = pd.DataFrame(columns=[str(gp)], index=[var_list])
        ks_value_lst = []
        for var in var_list:
            ks_value = ks_2samp(pos_input_df[var], neg_input_df[var]).statistic
            ks_value_lst.append(round(ks_value, 4))
        sub_output_df[str(gp)] = ks_value_lst
        output_df = pd.concat([output_df, sub_output_df], axis=1, sort=False)
        
    output_cols = [str(x) for x in group_list]
    output_df.loc[:, 'mean'] = output_df.apply(lambda row: row[output_cols].mean(), axis=1)
    output_df.loc[:, 'std'] = output_df.apply(lambda x: x[output_cols].std(), axis=1)
    output_df.loc[:, 'temp'] = 0.01
    output_df.loc[:, 'cv'] = output_df['std'] / (output_df['temp'] + output_df['mean'])
    output_df = output_df.drop(['temp'], axis=1)
    output_df = output_df.sort_values(by=['mean'], ascending=False)
    output_df['target'] = target_var
    
    """step3: 保存输出"""
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.csv'):
            raise Exception('参数save_file_path不是csv文件后缀，请检查!')
        output_df.to_csv(save_file_path, encoding='utf-8', index=False)
    
    return output_df


def ks_table(input_df, score_var, target_var, 
             loan_var=None, eff_var=None, 
             bins=20, bin_mode=1,
             score_bmk_list=None, bin_break_list=None, ascending=False, 
             save_file_path=None):
    '''
    ----------------------------------------------------------------------
    功能：生成ks_table, 可观察每个bin内的正负样本数，几率odds，lift和ks
    包括：['bucket','min_score','max_score','delta_score',
          'total','total_rate','rejects','reject_rate','loans','loan_rate','cum_loan_rate',
          'effects','goods','bads','bad_rate','cum_bad_rate','cum_good_rate',
          'odds','lift','cum_lift','ks','max_ks']
    含义：
    bucket                  INT      COMMENT '分箱序号，取值为1，2，3,...',
    min_score               DOUBLE   COMMENT '分箱内订单评分(score)的最小值',
    max_score               DOUBLE   COMMENT '分箱内订单评分(score)的最大值',
    delta_score             DOUBLE   COMMENT '分箱内订单评分(score)的间隔',
    total                   INT      COMMENT '分箱内申贷订单数',
    total_rate              DOUBLE   COMMENT '分箱内订单占比（分箱内订单数 / 所有分箱的总订单数)',
    rejects                 BIGINT   COMMENT '分箱内拒绝订单数',
    reject_rate             DOUBLE   COMMENT '分箱内拒绝率（分箱内拒绝订单数 / 分箱内申贷订单数)',
    loans                   BIGINT   COMMENT '分箱内放贷订单数',
    loan_rate               DOUBLE   COMMENT '分箱内放贷率（分箱内放贷订单数 / 分箱内申贷订单数)',
    cum_loan_rate           DOUBLE   COMMENT '分箱间累积放贷率 (分箱间累积放贷订单数 / 所有分箱总放贷订单数)',
    effects                 INT      COMMENT '分箱内进入表现期的订单数',
    goods                   INT      COMMENT '分箱内状态为good的订单数',
    good_rate               DOUBLE   COMMENT '分箱内good率 (状态为good的订单数 / 状态为effect订单数)',
    cum_good_rate           DOUBLE   COMMENT '分箱间累积good率 (分箱间累积good订单数 / 所有分箱总good订单数)',
    bads                    INT      COMMENT '分箱内状态为bad的订单数',
    bad_rate                DOUBLE   COMMENT '分箱内bad率 (状态为bad的订单数 / 状态为effect订单数)',
    cum_bad_rate            DOUBLE   COMMENT '分箱间累积bad率 (分箱间累积bad逾期订单数 / 所有分箱总bad订单数)',
    odds                    DOUBLE   COMMENT '分箱内坏好比odds (分箱内状态为bad的订单数 / 分箱内状态为good的订单数)',
    lift                    DOUBLE   COMMENT '分箱内提升度lift (分箱内bad率 / 所有分箱总bad率)',
    cum_lift                DOUBLE   COMMENT '分箱间累积提升度lift(分箱间累积bad率 / 所有分箱总bad率)',
    ks                      DOUBLE   COMMENT '分箱间ks (分箱间累积bad率 - 分箱间累积good率)',
    max_ks                  DOUBLE   COMMENT '分箱间最大的ks'

    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param score_var: str, ks曲线横坐标变量。
    :param target_var: str, 目标变量，取值为1或0。取值示例: 's1d30'
    :param loan_var: str, 是否放贷的标识变量。默认值=None, 代表全部是放贷样本。取值示例：'is_loan'
    :param eff_var: str, 是否进入表现期的标识变量。默认值=None, 代表全部进入表现期。取值示例：'is_eff_s1d30'
    :param bins: int, 分箱数，默认20
    :param bin_mode: int, 分箱模式. 
                     当前支持:
                     bin_mode=1: 等频分箱(区间左开右闭)
                     bin_mode=2: 等距分箱(区间左开右闭)
                     bin_mode=3: 指定分布分箱(区间左开右闭, 以给定分数的等距分位数为基准 )
                     bin_mode=4: 指定边界分箱分箱(区间左开右闭, 给定分箱边界, bins=边界数-1)
    :param score_bmk_list: list, 指定的分数变量取值，一般会选择INS上的score_var作为基准。
                     score_bmk_list = input_df[(input_df[loan_var] == 1) & (input_df['group'] == 'INS')]
    :param bin_break_list: list, 指定的分箱取值，一般会选择INS上的score_var分箱间隔。
    :param ascending: bool, ks曲线横坐标score_var升序或降序。
                      默认值False, 按score_var降序，高分段在头部（分箱编号小），低分段在尾部，符合策略分析习惯（高分段制定cutoff）。
                      1）若score_var类似<欺诈分>（取值越大, P(bad)越高）, 则高分段bad人群集中，低分段good人群集中。
                      cum_bad_rate曲线在cum_good_rate上方
                      2）若score_var类似<信用分>（取值越大, P(bad)越低）, 则高分段good人群集中，低分段bad人群集中。
                      cum_bad_rate曲线在cum_good_rate下方
                      此时若希望ks > 0, 可令ascending=True; 
                      不影响max_ks取值，仅影响顺序。               
    :param save_file_path: string, csv文件保存路径. 默认值=None
    ----------------------------------------------------------------------
    :return output_df: pd.DataFram, 返回结果
    ----------------------------------------------------------------------
    示例：
    >>> ks_t1 = ks_table(input_df=df, score_var='score', target_var='Survived', loan_var=None, eff_var=None, 
                         bins=10, bin_mode=1, score_bmk_list=None, bin_break_list=None, ascending=1)
    ------------------------
    >>> ks_t2 = ks_table(input_df=df, score_var='score', target_var='Survived', loan_var=None, eff_var=None, 
                         bins=10, bin_mode=2,score_bmk_list=None, bin_break_list=None, ascending=0)
    ------------------------
    >>> ks_t3 = ks_table(input_df=df[df['group'] == 'seg1'], score_var='score', target_var='Survived', 
                         loan_var=None, eff_var=None, bins=10, bin_mode=3,
                         score_bmk_list=df[df['group'] == 'seg1']['score'], bin_break_list=None, ascending=0)
    ------------------------
    >>> ks_t4 = ks_table(input_df=df, score_var='score', target_var='Survived', loan_var=None, eff_var=None, 
                         bins=10, bin_mode=4, score_bmk_list=None, bin_break_list=[0.2, 0.6, 0.7], ascending=0)
    ----------------------------------------------------------------------
    知识：
    KS曲线中, 如果横坐标变量是P(bad), 则升序排列时，沿着横坐标方向P(bad)增大，此时cum_bad_rate一定会在cum_good_rate曲线上方。
    但评分卡模型分数转换中，会将P(bad)转换为信用分，因此取值升序排列时，沿着横坐标方向P(bad)减小，此时cum_bad_rate会在cum_good_rate曲线下方。
    ----------------------------------------------------------------------
    '''
    """step0: 防御性处理"""
    input_df_copy = input_df.copy()
    cols = set(input_df_copy.columns)
    if score_var not in cols:
        raise Exception('参数score_var取值包含不属于input_df的变量，请检查!')
    if target_var not in cols:
        raise Exception('参数target_var取值包含不属于input_df的变量，请检查!')
    if loan_var is not None:
        if loan_var not in cols:
            raise Exception('参数loan_var取值包含不属于input_df的变量，请检查!')
    if eff_var is not None:
        if eff_var not in cols:
            raise Exception('参数eff_var取值包含不属于input_df的变量，请检查!')
    
    
    """step1: 样本标签定义"""
    if loan_var is None:
        input_df_copy.loc[:, 'reject'] = 0
        input_df_copy.loc[:, 'loan'] = 1
    else:
        input_df_copy[[loan_var]] = input_df_copy[[loan_var]].fillna(0)
        input_df_copy.loc[:, 'reject'] = 1 - input_df_copy[loan_var]
        input_df_copy.loc[:, 'loan'] = input_df_copy[loan_var]
        
    if eff_var is None:
        # 默认全部进入表现期
        input_df_copy.loc[:, 'eff'] = 1
        input_df_copy.loc[:, 'bad'] = input_df_copy.loc[:, target_var]
        input_df_copy.loc[:, 'good'] = (input_df_copy.loc[:, 'loan'] - input_df_copy.loc[:, target_var])
    else:
        input_df_copy.loc[:, 'eff'] = input_df_copy.loc[:, eff_var]
        input_df_copy.loc[:, 'bad'] = input_df_copy.loc[:, target_var] * input_df_copy.loc[:, 'eff']
        input_df_copy.loc[:, 'good'] = (input_df_copy.loc[:, loan_var] - input_df_copy.loc[:, target_var]) \
                                      * input_df_copy.loc[:, 'eff']
    # 排除不定样本进入统计
    input_df_copy.loc[:, 'bad'] = input_df_copy.loc[:, 'bad'].apply(lambda x: np.nan if x not in (0,1) else x)
    input_df_copy.loc[:, 'good'] = input_df_copy.loc[:, 'good'].apply(lambda x: np.nan if x not in (0,1) else x)
    input_df_copy.loc[:, 'score'] = input_df_copy.loc[:, score_var]

    """step2: 确定分箱模式"""
    if bin_mode == 1:
        """模式1: 等频分箱(区间左开右闭)"""
        input_df_copy['bucket'] = pd.qcut(input_df_copy.score, bins, duplicates='drop')
    elif bin_mode == 2:
        """模式3: 等距分箱(区间左开右闭)"""
        input_df_copy['bucket'] = pd.cut(input_df_copy.score, bins, include_lowest=0, duplicates='drop')
    elif bin_mode == 3:
        """模式2: 给定分数分箱(基于给定分数基准, 区间左开右闭)"""
        if score_bmk_list is None:
            raise Exception('参数score_bmk_list取值为None，请检查!')
        breakpoints = np.arange(0, bins + 1) / (bins) * 100
        bin_break_list = np.stack([np.percentile(score_bmk_list, b) for b in breakpoints])
        labels = range(len(bin_break_list) - 1)
        input_df_copy['bucket'] = pd.cut(input_df_copy.score, bins=bin_break_list, labels=labels, include_lowest=0)
    elif bin_mode == 4:
        """模式4: 指定间隔分箱分箱(基于给定分箱间隔, 区间左开右闭)"""
        if bin_break_list is None:
            raise Exception('参数bin_break_list取值为None，请检查!')
        labels = range(len(bin_break_list) - 1)
        input_df_copy['bucket'] = pd.cut(input_df_copy.score, bins=bin_break_list, labels=labels, include_lowest=0)
    else:
        raise Exception('不支持当前分箱模式, 仅支持:\n' \
                        + 'bin_mode=1: 等频分箱(区间左开右闭)\n' \
                        + 'bin_mode=2: 等距分箱(区间左开右闭)\n' \
                        + 'bin_mode=3: 指定分布分箱(区间左开右闭, 以给定分数的等距分位数为基准)\n' \
                        + 'bin_mode=4: 指定边界分箱分箱(区间左开右闭, 给定分箱边界, bins=边界数-1)\n')
        
    """step3: 计算ks-table"""
    grouped = input_df_copy.groupby('bucket', as_index=False)
    table = pd.DataFrame(grouped.min().score, columns = ['min_score'])
    table['max_score'] = grouped.max().score
    table['min_score'] = grouped.min().score
    table['delta_score'] = table.max_score - table.min_score
    # 申请层统计
    table['rejects'] = grouped.sum().reject
    table['loans'] = grouped.sum().loan
    table['total'] = table.rejects + table.loans
    # 放款层统计
    table['bads'] = grouped.sum().bad
    table['goods'] = grouped.sum().good 
    table['effects'] = table.goods + table.bads
    
    table['reject_rate'] = (table.rejects / table.total).apply('{0:.2%}'.format)
    table['loan_rate'] = (table.loans / table.total).apply('{0:.2%}'.format)
    table['total_rate'] = (table.total / table.total.sum()).apply('{0:.2%}'.format)
    
    table = (table.sort_values(by='min_score', ascending=ascending)).reset_index(drop=True)
    table['odds'] = (table.bads / table.goods).apply('{0:.4f}'.format)
    table['bad_rate'] = (table.bads / table.effects).apply('{0:.2%}'.format)
    table['lift'] = ((table.bads / table.effects) \
                    /(table.bads.sum() / table.effects.sum())).apply('{0:.4f}'.format)
    
    table['cum_loan_rate'] = ((table.loans / input_df_copy.loan.sum()).cumsum()).apply('{0:.2%}'.format)
    table['cum_bad_rate'] = ((table.bads / input_df_copy.bad.sum()).cumsum()).apply('{0:.2%}'.format)
    table['cum_good_rate'] = ((table.goods / input_df_copy.good.sum()).cumsum()).apply('{0:.2%}'.format)
  
    table['cum_lift'] = ((table.bads.cumsum() / table.effects.cumsum()) \
                        /(table.bads.sum() / table.effects.sum())).apply('{0:.4f}'.format)
    
    table['ks'] = np.round(((table.bads / input_df_copy.bad.sum()).cumsum() \
                          - (table.goods / input_df_copy.good.sum()).cumsum()), 4) * 100
    table['ks_sign'] = '-' if table.ks.max() <= 0 else '+'
    # KS取绝对值
    table['ks'] = table['ks'].apply(lambda x: abs(x))
    flag = lambda x: '<<<<<<' if x == table.ks.max() else ''
    table['max_ks'] = table.ks.apply(flag)
    table['bucket'] = [x for x in range(1, table.shape[0]+1)]
    output_df = table[['bucket','min_score','max_score','delta_score',
                       'total','total_rate','rejects','reject_rate','loans','loan_rate',
                       'cum_loan_rate','effects',
                       'goods','bads','bad_rate','cum_bad_rate','cum_good_rate',
                       'odds','lift','cum_lift','ks_sign','ks','max_ks']]
    
    """step4: 保存输出"""
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.csv'):
            raise Exception('参数save_file_path不是csv文件后缀，请检查!')
        output_df.to_csv(save_file_path, encoding='utf-8', index=False)
 
    return output_df  


def ks_grouply_table(input_df, score_var, target_var, 
                     loan_var=None, eff_var=None, 
                     bins=20, bin_mode=1, group_var=None, group_list=None,
                     score_bmk_list=None, bin_break_list=None, ascending=False,
                     save_file_path=None):
    '''
    ----------------------------------------------------------------------
    功能：生成ks_table, 可观察每个bin内的正负样本数，几率odds，lift和ks等指标
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param score_var: str, ks曲线横坐标变量。
    :param target_var: str, 目标变量，取值为1或0。取值示例: 's1d30'
    :param loan_var: str, 是否放贷的标识变量。默认值=None, 代表全部是放贷样本。取值示例：'is_loan'
    :param eff_var: str, 是否进入表现期的标识变量。默认值=None,代表全部进入表现期。取值示例：'is_eff_s1d30'
    :param bins: int, 分箱数，默认20
    :param bin_mode: int, 分箱模式. 
                     当前支持:
                     bin_mode=1: 等频分箱(区间左开右闭)
                     bin_mode=2: 等距分箱(区间左开右闭)
                     bin_mode=3: 指定分布分箱(区间左开右闭, 以给定分数的等距分位数为基准 )
                     bin_mode=4: 指定边界分箱分箱(区间左开右闭, 给定分箱边界, bins=边界数-1)
    :param group_var: string, 分组依据，如按[组，月，周]。默认值=None, 对整个数据集统计。示例: 'apply_month'
    :param group_list: list, 分组元素取值, 若为None，则默认为所有取值。示例: [201701, 201702]
    :param score_bmk_list: list, 指定的分数取值，一般会选择放贷样本INS上的score_var分布作为基准。
                     score_bmk_list = input_df[(input_df[loan_var] == 1) & (input_df['group'] == 'INS')]
    :param bin_break_list: list, 指定的分箱取值，一般会选择INS上的score_var分箱间隔。
    :param ascending: bool, ks曲线横坐标score_var升序或降序。
                      默认值False, 按score_var降序，高分段在头部（分箱编号小），低分段在尾部，符合策略分析习惯（高分段制定cutoff）。
                      1）若score_var类似<欺诈分>（取值越大, P(bad)越高）, 则高分段bad人群集中，低分段good人群集中。
                      cum_bad_rate曲线在cum_good_rate上方
                      2）若score_var类似<信用分>（取值越大, P(bad)越低）, 则高分段good人群集中，低分段bad人群集中。
                      cum_bad_rate曲线在cum_good_rate下方
                      此时若希望ks > 0, 可令ascending=True; 
                      不影响max_ks取值，仅影响顺序。
    :param save_file_path: string, csv文件保存路径. 默认值=None
    ----------------------------------------------------------------------
    :return output_df: pd.DataFram, 返回结果
    ----------------------------------------------------------------------
    示例：
    >>> ks_group_table = \
        ks_grouply_table(input_df=df, score_var='score', target_var='Survived', loan_var=None, eff_var=None, 
                         bins=10, bin_mode=3, group_var=None, group_list=None,
                         score_bmk_list=df[df['group'] == 'seg1']['score'], bin_break_list=None, ascending=False,
                         save_file_path=None)
    ----------------------------------------------------------------------
    '''
    """step1: 防御性处理"""
    input_df_copy = input_df.copy()
    cols = list(input_df_copy.columns)
    if group_var is not None:
        if group_var not in set(input_df_copy.columns):
            raise Exception('参数group_var取值包含不属于input_df_copy的变量，请检查!')
    else:
        group_var = 'group'
        input_df_copy[group_var] = 'all'
    
    if group_list is None: 
        group_list = sorted(list(set(input_df_copy[group_var])))
    else:
        group_list = sorted(list(set(group_list)))
        
    """step2: 参数确定"""
    if bin_mode == 1:
        mode_flag = 'equal_freq'  # 等频
    elif bin_mode == 2:
        mode_flag = 'equal_dist'  # 等距
    elif bin_mode == 3:
        if score_bmk_list is None:
            raise Exception('该模式下，参数score_bmk_list取值不能为None，必须指定数值，请检查!')
        mode_flag = 'ins_ef_base' # ins等距
    elif bin_mode == 4:
        if bin_break_list is None:
            raise Exception('该模式下，参数bin_break_list取值不能为None，必须指定数值，请检查!')
        mode_flag = 'given_dist' # 给定间隔
    else:
        raise Exception('不支持当前分箱模式, 仅支持:\n' \
                        + 'bin_mode=1: 等频分箱(区间左开右闭)\n' \
                        + 'bin_mode=2: 等距分箱(区间左开右闭)\n' \
                        + 'bin_mode=3: 指定分布分箱(区间左开右闭, 以给定分数的等距分位数为基准)\n' \
                        + 'bin_mode=4: 指定边界分箱分箱(区间左开右闭, 给定分箱边界, bins=边界数-1)\n')
    
    """step3: 计算KS"""
    output_df = pd.DataFrame()
    for gp in group_list:
        gp_input_df = input_df_copy[input_df_copy[group_var] == gp]
        sub_output_df = ks_table(gp_input_df, score_var, target_var, loan_var, eff_var, 
                                 bins, bin_mode, score_bmk_list, bin_break_list, ascending)
        sub_output_df['group'] = gp
        output_df = pd.concat([output_df, sub_output_df], axis=0, sort=False)
    output_df['bin_mode'] = mode_flag
      
    """step4: 保存输出"""
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.csv'):
            raise Exception('参数save_file_path不是csv文件后缀，请检查!')
        output_df.to_csv(save_file_path, encoding='utf-8', index=False)
    
    return output_df


def ks_plot(score_arr, target_arr, asc=1):
    '''
    ----------------------------------------------------------------------
    功能: 计算单变量KS并可视化显示，主要用于模型评估
    ----------------------------------------------------------------------
    :param score_arr:  numpy array of shape (1,), 自变量
    :param target_arr: numpy array of shape (1,), 目标变量数列, 取值为0或1
    :param asc: int, 默认=1，表示对score_arr升序，0为降序
    ----------------------------------------------------------------------
    return None：ks值，以及ks曲线，见示例
    ----------------------------------------------------------------------
    '''
    """step1: 放缩"""
    min_score = np.min(score_arr)
    max_score = np.max(score_arr)
    
    """step2: 计算"""
    ks_df = pd.DataFrame({'score': score_arr, 'bad': target_arr})
    ks_df = ks_df[ks_df['bad'].isin([0,1])]
    ks_df.loc[:, 'good'] = 1 - ks_df.bad  
    
    ks_df = ks_df.sort_values(by=['score'], ascending=asc)
    ks_df.loc[:, 'cumsum_good_rate'] = 1.0 * ks_df.good.cumsum() / sum(ks_df.good)
    ks_df.loc[:, 'cumsum_bad_rate'] = 1.0 * ks_df.bad.cumsum() / sum(ks_df.bad)
    ks_df.loc[:, 'ks'] = ks_df['cumsum_bad_rate'] - ks_df['cumsum_good_rate']
    ks_sign = '-' if ks_df.ks.max() <= 0 else '+'
    ks_df['ks'] = ks_df['ks'].apply(lambda x: abs(x))
    ks_df['ks_sign'] = ks_sign
    
    ks_df.loc[:, 'bucket'] = range(1, len(ks_df.ks) + 1) # 分箱编号
    ks_df = ks_df.reset_index(drop=1)
    
    qe = list(np.arange(0, 1, 1.0 / ks_df.shape[0]))
    qe.append(1)
    qe = qe[1:]
    ks_index = pd.Series(ks_df.index)
    ks_index = ks_index.quantile(q = qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)
     
    ks_df = ks_df.loc[ks_index]
    ks_df = ks_df[['score','cumsum_good_rate','cumsum_bad_rate','ks_sign','ks']]
    
    if asc == 1:
        ks_df0 = np.array([[min_score, 0, 0, ks_sign,0]])
    else:
        ks_df0 = np.array([[max_score, 0, 0, ks_sign,0]])
    
    ks_df = np.concatenate([ks_df0, ks_df], axis=0)
    ks_df = pd.DataFrame(ks_df, columns=['score', 'cumsum_good_rate', 'cumsum_bad_rate', 'ks_sign', 'ks'])
    ks_df[['score', 'cumsum_good_rate', 'cumsum_bad_rate', 'ks']] = \
    ks_df[['score', 'cumsum_good_rate', 'cumsum_bad_rate', 'ks']].astype(float)
    
    if asc == 1:
        ks_df['score'] = sorted(ks_df['score']) # 画图总是按升序
        if ks_sign == '-':
            print('> score类似<欺诈分>, 取值越大，P(bad)越大.')
            print('> 升序模式, cum_bad_rate在cum_good_rate下方.')
            plt.xlabel('Fraud Score')
        else:
            print('> score类似<信用分>, 取值越大，P(bad)越小.')
            print('> 升序模式, cum_bad_rate在cum_good_rate上方.')
            plt.xlabel('Credit Score')
    else:
        ks_df['score'] = sorted(ks_df['score']) # 画图总是按升序
        if ks_sign == '-':
            print('> score类似<信用分>, 取值越大，P(bad)越小.')
            print('> 降序模式, cum_bad_rate在cum_good_rate下方.')
            plt.xlabel('Credit Score')
        else: 
            print('> score类似<欺诈分>, 取值越大，P(bad)越大.')
            print('> 降序模式, cum_bad_rate在cum_good_rate上方.')
            plt.xlabel('Fraud Score')
    ks_value_max = ks_df.ks.max()
    ks_pop_max = ks_df.score[ks_df.ks.idxmax()]
    ks_value = ks_value_max
    ks_pop = ks_pop_max

      
    print('ks_value is ' + str(np.round(ks_value, 4)) + ' at score = ' + str(np.round(ks_pop, 4)))
    
    """step3: 绘图"""
    plt.plot(ks_df.score, ks_df.cumsum_good_rate, label='cum_good_rate', color='green', linestyle='-', linewidth=2)               
    plt.plot(ks_df.score, ks_df.cumsum_bad_rate, label='cum_bad_rate', color='red', linestyle='-', linewidth=2)                   
    plt.plot(ks_df.score, ks_df.ks, label='ks_curve', color='blue', linestyle='-', linewidth=2)
                     
    plt.axvline(ks_pop, color='gray', linestyle='--')
    plt.axhline(ks_value, color='blue', linestyle='--')
    plt.axhline(ks_df.loc[ks_df.ks.idxmax(), 'cumsum_good_rate'], color='green', linestyle='--')
    plt.axhline(ks_df.loc[ks_df.ks.idxmax(), 'cumsum_bad_rate'], color='red', linestyle='--')
    plt.title('KS=%s ' % np.round(ks_value, 4) +  'at Pop=%s' % np.round(ks_pop, 4), fontsize=15)
    plt.legend(labels=['cumsum_good_rate', 'cumsum_bad_rate', 'ks_curve'])
    plt.show()
    
    return plt


def ks_table_plot(input_df, group_var=None, group_value=None, save_file_path=None):
    '''
    ----------------------------------------------------------------------
    功能: 读取ks_table()生成的ks_table可视化绘制KS曲线
    ----------------------------------------------------------------------
    :param input_df:  pd.DataFrame, 输入数据
    :param group_var: str, 分组依据，如按[组，月，周]。默认值=None, 对整个数据集统计。示例: 'apply_month'
    :param group_value: str, 分组元素取值, 若为None，则默认为所有取值。示例: [201701, 201702]
    :param save_file_path: string, jpg文件保存路径. 默认值=None
    ----------------------------------------------------------------------
    return None：ks曲线
    ----------------------------------------------------------------------
    '''
    input_df_copy = input_df.copy() 
    cols = list(input_df_copy.columns)
    if group_var is not None:
        if group_var not in set(input_df_copy.columns):
            raise Exception('参数group_var取值包含不属于input_df_copy的变量，请检查!')
        input_df_copy[group_var] = input_df_copy[group_var].apply(lambda x: str(x))
    else:
        group_var = 'group'
        input_df_copy[group_var] = 'all'
     
    if group_value is None:
        group_value = sorted(list(set(input_df_copy[group_var])))[0] # 若存在多个group, 任取一个group
    input_df_copy = input_df_copy[input_df_copy[group_var] == group_value]
        
    input_df_copy['cum_bad_rate'] = input_df_copy['cum_bad_rate'].apply(lambda x: float(x[:-1]))
    input_df_copy['cum_good_rate'] = input_df_copy['cum_good_rate'].apply(lambda x: float(x[:-1]))
    
    plt.figure(figsize=[8, 6])
    
    bucket_num = input_df_copy.shape[0]
    input_df_copy.index = range(1, bucket_num+1)
    bucket_list = [str(x) for x in range(1, bucket_num+1)]
    cum_bad_rate_lst =  [round(x,2) for x in list(input_df_copy['cum_bad_rate'])]
    cum_good_rate_lst = [round(x,2) for x in list(input_df_copy['cum_good_rate'])]
    ks_value_lst = [round(x,2) for x in list(input_df_copy['ks'])]
    
    min_score_list = list(input_df_copy['min_score'])
    asc_flag = 0 if min_score_list[0] >= min_score_list[1] else 1 # 升序降序标识
    ks_sign = list(input_df_copy['ks_sign'])[0]
    
    if asc_flag == 1:
        if ks_sign == '-':
            print('> score类似<欺诈分>, 取值越大，P(bad)越大.')
            print('> 升序模式, cum_bad_rate在cum_good_rate下方.')
            plt.xlabel('bucket(Fraud Score, score low -> high)')
        else:
            print('> score类似<信用分>, 取值越大，P(bad)越小.')
            print('> 升序模式, cum_bad_rate在cum_good_rate上方.')
            plt.xlabel('bucket(Credit Score, score low -> high)')
    else:
        if ks_sign == '-':
            print('> score类似<信用分>, 取值越大，P(bad)越小.')
            print('> 降序模式, cum_bad_rate在cum_good_rate下方.')
            plt.xlabel('bucket(Credit Score, score high -> low)')
        else: 
            print('> score类似<欺诈分>, 取值越大，P(bad)越大.')
            print('> 降序模式, cum_bad_rate在cum_good_rate上方.')
            plt.xlabel('bucket(Fraud Score, score high -> low)')
            
    ks_value = input_df_copy.ks.max()
    ks_pop = input_df_copy.bucket[input_df_copy.ks.idxmax()]
    plt.ylabel('rate(%)')

    plt.plot(bucket_list, cum_bad_rate_lst, linewidth=3, marker='o', color='red')
    plt.plot(bucket_list, cum_good_rate_lst, linewidth=3, marker='o', color='green')
    plt.plot(bucket_list, ks_value_lst, linewidth=3, marker='o', color='blue')
    for a, b in zip(bucket_list, cum_bad_rate_lst):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(bucket_list, cum_good_rate_lst):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(bucket_list, ks_value_lst):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

    plt.axvline(ks_pop-1, color='gray', linestyle='--')
    plt.axhline(input_df_copy.loc[input_df_copy.ks.idxmax(), 'cum_bad_rate'] , color='red', linestyle='--')
    plt.axhline(input_df_copy.loc[input_df_copy.ks.idxmax(), 'cum_good_rate'], color='green', linestyle='--')
    plt.axhline(ks_value, color='blue', linestyle='--')
    
    plt.title('KS=%s ' % np.round(ks_value, 4) +  'at bucket=%s' % np.round(ks_pop, 4), fontsize=15)
    plt.legend(labels=['cum_bad_rate','cum_good_rate','ks_curve'])
    plt.show()
    
    """step4: 保存输出"""
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.jpg'):
            raise Exception('参数save_file_path不是jpg文件后缀，请检查!')
        plt.savefig(save_file_path)
        print('成功保存至:{}'.format(save_file_path))
    
    return plt


def marginal_ks(input_df, target_var, group_var, group_list=None, var_list=None, save_file_path=None):
    pass


def roc_plot(input_df, group_var=None, group_value=None, save_file_path=None):
    '''
    ----------------------------------------------------------------------
    功能: 读取ks_table()生成的ks_table可视化绘制ROC曲线
    ----------------------------------------------------------------------
    :param input_df:  pd.DataFrame, 输入数据
    :param group_var: str, 分组依据，如按[组，月，周]。默认值=None, 对整个数据集统计。示例: 'apply_month'
    :param group_value: str, 分组元素取值, 若为None，则默认为所有取值。示例: [201701, 201702]
    :param save_file_path: string, jpg文件保存路径. 默认值=None
    ----------------------------------------------------------------------
    return None：ROC曲线
    ----------------------------------------------------------------------
    使用示例：
    >> ks_t1 = ks_table(input_df=df, score_var='score2', target_var='Survived', 
                        loan_var=None, eff_var=None, 
                        bins=10, bin_mode=1,
                        score_bmk_list=None, bin_break_list=None, ascending=1)
    >> roc_plot(input_df=ks_t1)
    ----------------------------------------------------------------------
    '''
    input_df_copy = input_df.copy() 
    cols = list(input_df_copy.columns)
    if group_var is not None:
        if group_var not in set(input_df_copy.columns):
            raise Exception('参数group_var取值包含不属于input_df_copy的变量，请检查!')
        input_df_copy[group_var] = input_df_copy[group_var].apply(lambda x: str(x))
    else:
        group_var = 'group'
        input_df_copy[group_var] = 'all'
     
    if group_value is None:
        group_value = sorted(list(set(input_df_copy[group_var])))[0] # 若存在多个group, 任取一个group
    input_df_copy = input_df_copy[input_df_copy[group_var] == group_value]
        
    input_df_copy['cum_bad_rate'] = input_df_copy['cum_bad_rate'].apply(lambda x: float(x[:-1]))
    input_df_copy['cum_good_rate'] = input_df_copy['cum_good_rate'].apply(lambda x: float(x[:-1]))
    
    plt.figure(figsize=[8, 6])
    
    bucket_num = input_df_copy.shape[0]
    input_df_copy.index = range(1, bucket_num+1)
    bucket_list = [str(x) for x in range(1, bucket_num+1)]
    cum_bad_rate_lst =  [round(x,2) for x in list(input_df_copy['cum_bad_rate'])]
    cum_good_rate_lst = [round(x,2) for x in list(input_df_copy['cum_good_rate'])]
    
    plt.ylabel('True Positive Rate(%)')
    plt.xlabel('False Positive Rate(%)')
    plt.plot(cum_good_rate_lst, cum_bad_rate_lst, linewidth=3, marker='o', color='blue')
    plt.title('Receiver Operating Characteristic')
    plt.show()
    
    return plt

