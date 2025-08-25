# -*- coding:utf-8 -*-
__author__ = 'fenghaijie / hjfeng0630@qq.com'

import math
import numpy as np

"""
模块描述：模型分数变化(Model Scaling)
功能包括：
1. scorecard_scaling: 评分卡分数变换，将LR输出结果进行尺度变换（按Odds校准）
"""


def scorecard_scaling(score_type='credit', PDO=25, basescore=600, odds=1.0/15):
    """
    ----------------------------------------------------------------------
    功能：评分卡分数变换，将LR输出结果进行尺度变换（按Odds校准）
    ----------------------------------------------------------------------
    :param score_type: str, 代表评分卡分数含义. 'credit'=信用分, 'fraud'=欺诈分
    :param PDO: int, 默认值=25, 为正数
                1) 若score为信用分, 则代表odds翻倍(odds -> 2 * odds)所【减少】的分数
                2) 若score为欺诈分, 则代表odds翻倍(odds -> 2 * odds)所【增加】的分数
                注意：odds = bad_rate / good_rate
    :param basescore: float, 基准分，默认值=600
    :param odds: float, 基准分对应的坏好比, 默认值=1/15
    ----------------------------------------------------------------------
    :return alpha: float. score = alpha + beta * ln(odds)
    :return beta: float.  score = alpha + beta * ln(odds)
    ----------------------------------------------------------------------
    知识：
    对于LR模型而言, 公式为：
    log(odds) = w0 + w1 * x1 + w2 * x2 + ... + wn * xn
    其中, w0为截距, odds = p / (1-p) = bad_rate / good_rate = bads / goods
    业务含义为: 自变量[x1, x2, ..., xn]取值越大, bad_rate越高
    因此一般要求系数[w0, w1, w2, ..., wn]同号，因此在变量筛选阶段需剔除业务含义相反的变量。例如这里就需要剔除具有正面的变量x
    
    若自变量[x1, x2, ..., xn]进行WOE变换, 对应的变量列表为[x1_woe, x2_woe, ..., xn_woe]
    xn_woe = ln(odds_good) = ln(good_rate / bad_rate)
    且变换方向为：woe值越大, good_rate越高（本脚本woe变换就是属于这种）,
    => 那么所估计的系数[w0, w1, w2, ..., wn]必然是负数!
    ---------------------
    评分卡尺度变换过程为: 
    score = alpha + beta * ln(odds)
          = alpha + beta * [w0 + w1 * x1_woe + w2 * x2_woe + ... + wn * xn_woe]
          = alpha + (beta * wo) + (beta * w1) * x1_woe + (beta * w2) * x2_woe + ... + (beta * wn) * xn_woe 
          = [alpha + (beta * wo)] + (beta * w1) * x1_woe + (beta * w2) * x2_woe + ... + (beta * wn) * xn_woe
          = base_score_0 + sub_score_1 + sub_score_2 + ... + sub_score_n
          
    由此可见, 变量列表为[x1_woe, x2_woe, ..., xn_woe]与最终score存在线性关系, 具有良好的可解释性.
    
    1) 若score的含义是信用分, 则代表score越大, good_rate越高.
    [x1_woe, x2_woe, ..., xn_woe]已经经过woe变换, 具有该业务含义——woe值越大, good_rate越高
    而前面说到, 系数[w0, w1, w2, ..., wn]必然是负数
    => beta只可能是负数, 才能"负负得正"
    
    1) 若score的含义是欺诈分, 则代表score越大, bad_rate越高.
    [x1_woe, x2_woe, ..., xn_woe]已经经过woe变换, 具有该业务含义——woe值越大, bad_rate越高
    而前面说到, 系数[w0, w1, w2, ..., wn]必然是负数
    => beta只可能是正数, 才能"负正得负"
    ---------------------
    接下来定义变换尺度, 也就是确定alpha和beta
    注意：这里的ln(odds)仍是LR等式左边部分，odds的含义仍是坏好比！
    
    1) 若score = 信用分: 
    basescore = alpha + beta * ln(odds_0) 
    basescore - PDO = alpha + beta * ln(odds_1)
    odds_1 = 2 * odds_0
    其中, 已知的3个参数为odds_0、PDO、basescore
    => -PDO = beta * [ln(odds_1) - ln(odds_0)]
           = beta * [ln(odds_1 / odds_0)]
           = beta * ln(2)
    => beta = -PDO / ln2
    => alpha = basescore - beta * ln(odds_0) 
    
    2) 若score = 欺诈分: 
    basescore = alpha + beta * ln(odds_0) 
    basescore + PDO = alpha + beta * ln(odds_1)
    odds_1 = 2 * odds_0
    其中, 已知的3个参数为odds_0、PDO、basescore
    => PDO = beta * [ln(odds_1) - ln(odds_0)]
           = beta * [ln(odds_1 / odds_0)]
           = beta * ln(2)
    => beta = PDO / ln2
    => alpha = basescore - beta * ln(odds_0) 
    ---------------------
    在默认参数(一般都是信用分)下：
    beta = -PDO / ln2 = -36.06737602222409
    alpha = basescore - beta * ln(odds_0) = 697.672264890213 
    ----------------------------------------------------------------------
    """
    if score_type == 'credit': 
        beta = -1 * PDO / np.log(2)  
        alpha = basescore + beta * np.log(odds)
        print('评分卡类型为信用评分卡, 分数转换公式为：score = {} + {} * ln(odds)'.format(alpha, beta))
    else:
        beta = 1 * PDO / np.log(2)  
        alpha = basescore + beta * np.log(odds)
        print('评分卡类型为欺诈评分卡, 分数转换公式为：score = {} + {} * ln(odds)'.format(alpha, beta))
    
    return alpha, beta