import random
import numpy as np
import math
import pandas as pd

"""
模块描述：拒绝推断(Reject Inference, RI)
功能包括：
1. proportional_assignment: 简单随机推断(分散打包法)
2. simple_augmentation:  拒绝样本按欺诈评分降序排列，取头部样本赋予1，其余赋予0。
3. parcelling: 分箱随机赋值。

信贷行业常见的风控流程为：申贷 > 准入拒绝 > 反欺诈拒绝 > 信用审核(自动+人工) > 放贷 > 贷后管理（催收）。

因此，经过层层风控环节过滤之后，最终放贷订单可能只占申贷订单的10%（不同公司不同产品线各异）。

在准入评分卡（A卡）用到的建模样本一般是针对于放款的样本，然而在实际风控流程中，A卡需要预测的样本要比最终准入的样本要多，这就导致预测的目标样本与建模样本之间是有一定差异的（以部分样本去估计总体），那么如何衡量这种差异，如果差异很大，如何缩小。这就是信用模型中拒绝推断（也被称为拒绝演绎，Reject Inference）的内容。

参考资料：
【1】SAS中的拒绝推断模块介绍  Reject Inference Techniques Implemented in Credit Scoring for SAS.pdf
【2】拒绝推断（RI）之基础篇 https://zhuanlan.zhihu.com/p/55020918
【3】风险控制-如何理解信用模型中的拒绝推断-20180207 https://zhuanlan.zhihu.com/p/33655673
"""

def proportional_assignment(accept_df, reject_df, target_var, times=5, frac=0.2):
    '''
    ----------------------------------------------------------------------
    功能: 简单随机推断(分散打包法)，对拒绝样本随机赋予0/1标签。建议拒绝正样本率约为放款样本的2～5倍
    ----------------------------------------------------------------------
    :param accept_df:  dataframe, 放款订单样本，有标签
    :param reject_df:  dataframe, 拒绝订单样本，无标签
    :param target_var: str, 目标变量。示例: 's1d30'
    :param times:      float, 拒绝订单正样本率 = times * 放款订单正样本率
    :param frac:       float, 拒绝订单采样率，0～1
    ----------------------------------------------------------------------
    :return reject_df: 有标签的拒绝订单样本
    ----------------------------------------------------------------------
    知识：
    1. Proportional Assignment（简单随机推断）
    Proportional assignment is random partitioning of the rejects into "good" and "bad" accounts with a "bad" rate 2~5 times greater than in the accepted population.
    ----------------------------------------------------------------------
    '''
    accept_df = accept_df.loc[accept_df[target_var].isin([0, 1]), :]
    reject_bad_ratio = accept_df[target_var].mean() * times
    reject_df = reject_df.sample(frac=frac).reset_index(drop=1) # 随机抽样
    reject_df.loc[:, 'randNum'] = [random.random() for i in range(len(reject_df))]
    reject_df.loc[:, target_var] = reject_df.loc[:, 'randNum'].apply(lambda x: 1 if x <= reject_bad_ratio else 0)
    del reject_df.loc['randNum']
    
    return reject_df


def simple_augmentation(accept_df, reject_df, target_var, score_var, cutoff=None, times=5, frac=0.2):
    '''
    ----------------------------------------------------------------------
    功能: 对拒绝样本按欺诈评分降序排列，取头部样本赋予1，其余赋予0。建议拒绝正样本率约为放款样本的2～5倍
    ----------------------------------------------------------------------
    :param accept_df:  dataframe, 放款订单样本，有标签
    :param reject_df:  dataframe, 拒绝订单样本，无标签
    :param target_var: str, 目标变量。示例: 's1d30'
    :param score_var:  str, 欺诈/信用分数变量
    :param cutoff:     截断分数，默认为None
    :param times:      float, 拒绝订单正样本率 = times * 放款订单正样本率
    :param frac:       float, 拒绝订单采样率，0～1
    ----------------------------------------------------------------------
    :return reject_df: 有标签的拒绝订单样本
    ----------------------------------------------------------------------
    知识：
    2. Simple Augmentation（硬截止法）
    Simple augmentation assumes scoring the rejects using the base_logit_model and partitioning it into "good" and "bad" accounts based on a cut-off value. 
    The cut-off value is selected so the "bad" rate in the rejects is 2~5 times greater than in the accepts.
    扩张法(Augmentation)又称加权法(Re-Weighting)
    硬截止法首先利用接受样本创建信用评分模型，并据此给拒绝样本中的申请人打分。
    该方法假设得分高于某个临界值的为好客户，低于临界值的为坏客户，这里的临界值也需要业务人员给出坏客户率的先验估计。
    ----------------------------------------------------------------------
    '''
    accept_df = accept_df.loc[accept_df[target_var].isin([0, 1]), :]
    reject_bad_ratio = accept_df[target_var].mean() * times
    if cutoff is None:
        cut_index = int(reject_bad_ratio * len(reject_df))
        cut_off_score = sorted(list(reject_df[score_var]), reverse=1)[cut_index]
    else:
        cut_off_score = cutoff
    reject_df = reject_df.sample(frac=frac).reset_index(drop=1)  # 随机抽样
    reject_df.loc[:, target_var] = reject_df.loc[:, score_var].apply(lambda x: 1 if x >= cut_off_score else 0)
     
    return reject_df


def parcelling(accept_df, reject_df, target_var, score_var, score_range_mode='A', bins=10, times=5, frac=0.2):
    '''
    ----------------------------------------------------------------------
    功能: 分箱随机赋值。对拒绝样本按欺诈评分降序排列后进行分箱, 再对每个分箱里操作Proportional_Assignment。拒绝正样本率约为放款样本的2～5倍
    ----------------------------------------------------------------------
    :param accept_df:  dataframe, 放款订单样本，有标签
    :param reject_df:  dataframe, 拒绝订单样本，无标签
    :param target_var: str, 目标变量。示例: 's1d30'
    :param score_var:  str, 欺诈/信用分数变量
    :param score_range_mode: str, 用以生成score分箱区间的基准，'A'代表accept_df, 'R'代表reject_df, 'S'代表所有样本
    :param bins:       int, 分箱数
    :param times:      float, 拒绝订单正样本率 = times * 放款订单正样本率
    :param frac:       float, 拒绝订单采样率，0～1
    ----------------------------------------------------------------------
    :return reject_df: 有标签的拒绝订单样本
    ----------------------------------------------------------------------
    知识：
    4. Parcelling（打包法）
    Parcelling is a hybrid method encompassing simple augmentation and proportional assignment. Parcels are created by binning the rejects’ scores, generated using the base_logit_model, into the score bands. Proportional assignment is applied on each parcel with a "bad" rate 2~5 times greater than the "bad" rate in the equivalent score band of the accepted population.

    打包方法首先利用接受样本创建初步的信用评分模型，并把预测概率排序分组，然后给拒绝样本中的申请人打分，并对打分得到的预测概率按照接受样本中的预测概率分组规则进行分组。该方法假设在同一概率组中，拒绝样本中的坏客户比例是相对应的接受样本中坏客户比例的若干倍，这个倍数就叫做事件增长率。事件增长率需要业务人员根据经验给出估计，是一种先验信息。
----------------------------------------------------------------------
    '''
    # 放贷样本分箱
    accept_df = accept_df.loc[accept_df[target_var].isin([0, 1]), :]
    if score_range_mode == 'A':
        benchmark = accept_df.loc[:, score_var]
    elif score_range_mode == 'R':
        benchmark = reject_df.loc[:, score_var]
    else:
        benchmark = list(accept_df.loc[:, score_var]) + list(reject_df.loc[:, score_var])
    break_points = np.stack([np.percentile(benchmark, b) for b in np.arange(0, bins + 1) / (bins) * 100])
    labels = range(len(break_points) - 1)
    accept_df.loc[:, 'bucket'] = pd.cut(accept_df[score_var], bins=break_points, labels=labels, include_lowest=True)
     
    # 各分箱样本统计
    accept_stat = pd.DataFrame()
    accept_stat.loc[:, 'total'] = accept_df.groupby('bucket')[target_var].count()
    accept_stat.loc[:, 'bads'] = accept_df.groupby('bucket')[target_var].sum()
    accept_stat.loc[:, 'bads'] = accept_stat.loc[:, 'bads'].apply(lambda x: int(x))
    accept_stat.loc[:, 'goods'] = accept_stat.total - accept_stat.bads
    accept_stat.loc[:, 'bad_rate'] = accept_stat.bads * 1.0 / accept_stat.total
    accept_bad_ratio_lst = list(accept_stat['bad_rate'])
     
    # 拒绝正样本率约为放款样本的2～5倍
    reject_bad_ratio_lst = [times * x for x in accept_bad_ratio_lst]
     
    # 拒绝样本随机打标签
    def set_bad_prob(score):
        if score < break_points[0]:
            return 0
        if score > break_points[-1]:
            return 1
        for i in range(bins - 1):
            if score >= break_points[i] and score < break_points[i+1]:
                return reject_bad_ratio_lst[i]
    reject_df = reject_df.sample(frac=frac).reset_index(drop=1)   # 随机抽样
    reject_df.loc[:, 'randNum'] = [random.random() for i in range(len(reject_df))]
    reject_df.loc[:, 'bad_prob'] = reject_df.loc[:, score_var].apply(lambda x: set_bad_prob(x))
    reject_df.loc[:, target_var] = reject_df.apply(lambda row: 1 if row['bad_prob'] >= row['randNum'] else 0, axis=1)
    del reject_df['randNum']
    del reject_df['bad_prob']
     
    return reject_df


def reclassification():
    """
    5. Reclassification(重新分类法)
    重新分类的核心思想对被拒绝用户做好坏属性的重新划分，主要依赖外部数据。

    举个最简单的例子，当一个被拒绝的申请者具有一些负面特征，比如通过人行征信，我们发现其在过去3个月内有逾期行为，则可以把他划分成坏人。
    """
    pass

