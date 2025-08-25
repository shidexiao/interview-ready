#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   third_evalutor.py
@Time    :   2023/06/15 16:14:46
@Author  :   tangyangyang
@Contact :   tangyangyang@staff.sinaft.com
@Desc    :   三方数据自动评估模块
'''

# import use library
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("/Users/wj/新浪数科/模型开发/")


from model_tools.metrics import roc_auc_score, ks
import multiprocessing as mp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import tree
import math
from math import *
import os
import pickle
import scorecardpy as sc
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb


def data_report(in_data, numeric_missing_value=np.nan, string_missing_value=np.nan):
    numeric_missing_value = np.nan
    string_missing_value = np.nan

    desc = in_data.describe(percentiles=[
                            0.25, 0.5, 0.75, 0.95, 0.99], include='all').transpose().reset_index()

    if 'unique' in desc.columns:
        desc = desc.drop(['unique', 'top', 'freq'], axis=1)

    desc = desc.rename(columns={'index': 'column'})

    num_missing_to_1 = in_data.select_dtypes(include=['number']).applymap(
        lambda x: 1 if np.isnan(x) or x == numeric_missing_value else 0)
    count_num_missing = num_missing_to_1.apply(np.sum, axis=0).reset_index().rename(
        columns={'index': 'column', 0: 'missing'})
    zero_count_1 = in_data.select_dtypes(include=['number']).applymap(
        lambda x: 1 if x == 0 else 0)
    zero_count = zero_count_1.apply(np.sum, axis=0).reset_index().rename(
        columns={'index': 'column', 0: 'zero_count'})

    str_missing_to_1 = in_data.select_dtypes(include=['object']).applymap(
        lambda x: 1 if pd.isnull(x) or x == string_missing_value else 0)
    count_str_missing = str_missing_to_1.apply(np.sum, axis=0).reset_index().rename(
        columns={'index': 'column', 0: 'missing'})

    datetime_missing_to_1 = in_data.select_dtypes(
        include=['datetime']).applymap(lambda x: 1 if x == None else 0)
    count_datetime_missing = datetime_missing_to_1.apply(np.sum, axis=0).reset_index().rename(
        columns={'index': 'column', 0: 'missing'})

    count_missing = pd.concat(
        [count_num_missing, count_str_missing, count_datetime_missing])

    column_unique = in_data.apply(pd.Series.nunique, axis=0).reset_index().rename(
        columns={'index': 'column', 0: 'unique'})
    desc = desc.merge(column_unique, how='left', on='column')

    column_types = pd.DataFrame(in_data.dtypes).reset_index().rename(
        columns={'index': 'column', 0: 'dtype'})
    desc = desc.merge(column_types, how='left', on='column').merge(
        count_missing, how='left', on='column')
    desc = desc.merge(zero_count, how='left', on='column')

    desc['count'] = in_data.index.size
    desc['missing_rate'] = desc['missing'] / desc['count']
    desc['zero_rate'] = desc['zero_count'] / desc['count']

    if 'first' in desc.columns:
        desc['first'] = desc['first'].fillna('')
        desc['last'] = desc['last'].fillna('')

    desc['count'] = desc['count'].astype(np.int)

    if 'mean' in desc.columns:
        desc['mean'] = desc['mean'].astype(np.float)
        desc['std'] = desc['std'].astype(np.float)
        desc['min'] = desc['min'].astype(np.float)
        desc['25%'] = desc['25%'].astype(np.float)
        desc['50%'] = desc['50%'].astype(np.float)
        desc['75%'] = desc['75%'].astype(np.float)
        desc['95%'] = desc['95%'].astype(np.float)
        desc['99%'] = desc['99%'].astype(np.float)
        desc['max'] = desc['max'].astype(np.float)
    desc['missing_rate'] = desc['missing_rate'].astype(np.float)
    desc['dtype'] = desc['dtype'].astype(str)

    if 'first' in desc.columns:
        desc['first'] = desc['first'].astype(str)
        desc['last'] = desc['last'].astype(str)

    if 'first' in desc.columns:
        desc = desc[['column', 'dtype', 'count', 'missing', 'missing_rate', 'zero_count', 'zero_rate', 'unique', 'mean',
                    'std', 'min', '25%', '50%', '75%', '95%', '99%', 'max', 'first', 'last']]
    elif 'mean' in desc.columns:
        desc = desc[['column', 'dtype', 'count', 'missing', 'missing_rate', 'zero_count', 'zero_rate', 'unique', 'mean',
                    'std', 'min', '25%', '50%', '75%', '95%', '99%', 'max']]
    else:
        desc = desc[['column', 'dtype', 'count', 'missing',
                     'missing_rate', 'zero_count', 'zero_rate', 'unique']]
    return desc


def capture_topk(y_ture, y_pred, top=20):
    sort_index = np.argsort(-y_pred)
    y_ture_sort = y_ture[sort_index]
    topk = int(len(y_pred)*top/100)
    return np.sum(y_ture_sort[:topk])/np.sum(y_ture_sort)


def lift_topk(y_ture, y_pred, top=20):
    sort_index = np.argsort(-y_pred)
    y_ture_sort = y_ture[sort_index]
    topk = int(len(y_pred)*top/100)
    return (np.sum(y_ture_sort[:topk])/topk)/(np.sum(y_ture_sort)/len(y_ture))

# def lift_repair(data, cut_off, target, score, adjust_bad_rate, adjust_good_rate):
#     rj_sample_1 = data.query(f"{score} < @cut_off and {target}==1").shape[0]
#     rj_sample_0 = data.query(f"{score} < @cut_off and {target}==0").shape[0]
#     ps_sample_1 = data.query(f"{score} >= @cut_off and {target}==1").shape[0]
#     ps_sample_0 = data.query(f"{score} >= @cut_off and {target}==0").shape[0]
#     l = (rj_sample_1/adjust_bad_rate)/(rj_sample_1/adjust_bad_rate+rj_sample_0/adjust_good_rate)*\
#     (1+(rj_sample_0/adjust_good_rate +ps_sample_0/adjust_good_rate)/(rj_sample_1/adjust_bad_rate+ps_sample_1/adjust_bad_rate))
#     return l


def lift_repair_topk(y_ture, y_pred, adjust_bad_rate, adjust_good_rate, top=20):
    sort_index = np.argsort(-y_pred)
    y_ture_sort = y_ture[sort_index]
    topk = int(len(y_pred)*top/100)
    rj_sample_1 = np.sum(y_ture_sort[:topk])
    rj_sample_0 = topk - rj_sample_1
    ps_sample_1 = np.sum(y_ture_sort[topk:])
    ps_sample_0 = len(y_ture_sort[topk:]) - ps_sample_1
    return (rj_sample_1/adjust_bad_rate)/(rj_sample_1/adjust_bad_rate+rj_sample_0/adjust_good_rate) *\
        (1+(rj_sample_0/adjust_good_rate + ps_sample_0/adjust_good_rate) /
         (rj_sample_1/adjust_bad_rate+ps_sample_1/adjust_bad_rate))


def f_evalutor(x, proba_name, target):
    d = []
    d.append(x['cnt'].sum())
    d.append(x.query(f"{target}==[0, 1]")[target].sum())
    d.append(x.query(f"{target}==[0, 1]")[target].mean())
    d.append(x[target].replace(-1, 0).mean())
    d.append(round(roc_auc_score(
        x.query(f"{target}!=-1")[target], x.query(f"{target}!=-1")[proba_name]), 3))
    d.append(round(
        ks(x.query(f"{target}!=-1")[target], x.query(f"{target}!=-1")[proba_name]), 3))
    d.append(round(roc_auc_score(x[target].replace(-1, 0), x[proba_name]), 3))
    d.append(round(ks(x[target].replace(-1, 0), x[proba_name]), 3))
    # capture@top10%
    d.append(round(capture_topk(x.query(
        f"{target}!=-1")[target].values, x.query(f"{target}!=-1")[proba_name].values, 10), 3))
    # lift@top10%
    d.append(round(lift_topk(x.query(
        f"{target}!=-1")[target].values, x.query(f"{target}!=-1")[proba_name].values, 10), 3))
    # lift_repair@top10%
    d.append(round(lift_repair_topk(x.query(f"{target}!=-1")[target].values, x.query(
        f"{target}!=-1")[proba_name].values, 0.44, 0.16, 10), 3))
    return pd.Series(d, index=['#Count', '#Bad', '%Bad', '%Bad(灰当白)', 'AUC', 'KS', 'AUC(灰当白)', 'KS(灰当白)', 'Capture@top10%', 'Lift@top10%', 'Lift_Repair@top10%'])


class woe_bin(object):
    def __init__(self, indata, target, min_group_rate=0.1, max_bin=6, bin_method='mono', alg_method='iv'):
        self.indata = indata
        self.target = target
        self.min_group_rate = min_group_rate
        self.max_bin = max_bin
        self.bin_method = bin_method
        self.alg_method = alg_method

        self.min_num = int(len(self.indata) * self.min_group_rate)  # 限定最小分组数

    def read_data(self, tabname, tabkey):
        return pd.read_csv(tabname, index_col=tabkey)

    def check_y(self, dt, y):
        if dt[y].isnull().sum() > 0:
            raise Exception('目标变量中含有%s个空置' % str(dt[y].isnull().sum()))

        ambigous_dt = dt.loc[dt[y].isin([0, 1]) == 0]
        if len(ambigous_dt) > 0:
            raise Exception('目标变量中含有非01变量'.format(
                str(ambigous_dt[y].value_counts())))

    def bin_trans(self, var, sp_list):
        # sp_list = [1, 3.5]
        bin_no = 1
        for i, vi in enumerate(sp_list):
            if var <= vi:
                bin_no = i + 1
                break
            else:
                bin_no = len(sp_list) + 1
        return bin_no

    # bin_trans(12, [1, 3.5, 11])

    def woe_trans(self, var, sp_list, woe_list):

        woe = 0.0
        if np.isnan(sp_list).any():
            if pd.isna(var):
                woe = woe_list[np.where(np.isnan(sp_list))][0]
            else:
                for i, vi in enumerate(sp_list):
                    if var <= vi:
                        woe = woe_list[i]
                        break
                    else:
                        woe = woe_list[len(woe_list) - 1]
        else:
            for i, vi in enumerate(sp_list):
                if var <= vi:
                    woe = woe_list[i]
                    break
                else:
                    woe = woe_list[len(woe_list) - 1]
        return woe

    def get_bin(self, tabname, varname, sp_list):
        tab1 = tabname.copy()
        kwds = {"sp_list": sp_list}
        tab1['bin'] = tab1[varname].apply(self.bin_trans, **kwds)

        return tab1[['target', 'bin']]
    # test = get_bin(data1, 'td_id_3m', [1, 3.5])

    def get_bound(self, sp_list):
        # sp_list = [1, 3.5]
        ul = sp_list.copy()
        ll = sp_list.copy()
        ul.append(float("inf"))
        ll.insert(0, float("-inf"))

        sp_dict = {
            'bin': [i + 1 for i in list(range(len(sp_list) + 1))], 'll': ll, 'ul': ul}
        return pd.DataFrame(sp_dict)

    def get_dist(self, df, t0, t1):
        '''
        :param df:
        :param t0: t0和t1以全部数据来计算woe和iv
        :param t1:
        :return:
        '''
        # t_sum = pd.pivot_table(df, index='bin', columns='target', values='one', aggfunc=[np.sum])
        # t1 = df.target.sum()
        # t0 = len(df) - t1
        t_sum = df.groupby(['bin'])['target', 'one'].sum()
        t_sum.rename(columns={'target': 'p1'}, inplace=True)
        t_sum['p0'] = t_sum['one'] - t_sum['p1']
        t_sum.reset_index(level=0, inplace=True)

        t_sum['p1_r'] = t_sum['p1'] / t1 + 1e-6
        t_sum['p0_r'] = t_sum['p0'] / t0 + 1e-6
        t_sum['woe'] = np.log(t_sum['p1_r'] / t_sum['p0_r'])
        t_sum['iv0'] = (t_sum['p1_r'] - t_sum['p0_r']) * t_sum['woe']

        t_sum.drop(['one', 'p0_r', 'p1_r'], axis=1, inplace=True)
        return t_sum

    def get_mapiv_result(self, tabname, varname, sp_list, t0, t1):
        boundry = self.get_bound(sp_list)
        bin_no = self.get_bin(tabname, varname, sp_list)

        bin_no['one'] = 1
        mapiv1 = self.get_dist(bin_no, t0, t1)

        return boundry.merge(mapiv1, on='bin')

    # test = pd.DataFrame({'bin':[0,1,1,2,3,0,0,3,2],'target':[0,0,1,0,0,1,1,1,1]})
    # test['one'] = 1
    # test1 = get_dist(test)
    #
    # test1.index

    def get_iv(self, intab, varname, split_i):
        data_l = intab[intab[varname] <= split_i]
        data_u = intab[intab[varname] > split_i]

        p1 = intab['target'].sum()
        p0 = len(intab) - p1
        total_value = 0
        if p1 > 0 and p0 > 0:  # 分割后的数据满足最小分组数要求
            p1_l = data_l['target'].sum()
            p0_l = len(data_l) - p1_l

            p1_u, p0_u = p1 - p1_l, p0 - p0_l
            p1_u, p1_l, p0_u, p0_l = p1_u + 1e-6, p1_l + 1e-6, p0_u + 1e-6, p0_l + 1e-6
            if p0_l > 0 and p0_u > 0 and p1_l > 0 and p1_u > 0 and (p0_l + p1_l) >= self.min_num and (
                    p0_u + p1_u) >= self.min_num:

                woe_l = np.log((p1_l / p1) / (p0_l / p0) + 1e-6)
                woe_u = np.log((p1_u / p1) / (p0_u / p0) + 1e-6)

                if self.alg_method == 'iv':
                    iv_l = (p1_l / p1 - p0_l / p0) * \
                        np.log((p1_l / p1) / (p0_l / p0) + 1e-6)
                    iv_u = (p1_u / p1 - p0_u / p0) * \
                        np.log((p1_u / p1) / (p0_u / p0) + 1e-6)

                    total_value = iv_l + iv_u

                elif self.alg_method == 'gini':
                    gini_l = 1 - (p0_l**2 + p1_l**2) / len(data_l) ** 2
                    gini_u = 1 - (p0_u ** 2 + p1_u ** 2) / len(data_u) ** 2
                    gini_a = 1 - (p0**2 + p1**2) / len(intab) ** 2

                    total_value = 1 - (gini_l*len(data_l) +
                                       gini_u*len(data_u)) / (gini_a*len(intab))

                elif self.alg_method == 'entropy':
                    entropy_l = -(p0_l/len(data_l)) * log2(p0_l/len(data_l)) - \
                        (p1_l/len(data_l)) * log2(p1_l/len(data_l))
                    entropy_u = -(p0_u/len(data_u)) * log2(p0_u/len(data_u)) - \
                        (p1_u/len(data_u)) * log2(p1_u/len(data_u))
                    entropy_a = -(p0/len(intab)) * log2(p0/len(intab)) - \
                        (p1/len(intab)) * log2(p1/len(intab))

                    total_value = 1 - \
                        (entropy_l*len(data_l) + entropy_u *
                         len(data_u)) / (entropy_a*len(intab))

            else:
                return (0, -1)
                # iv = iv_l + iv_u
        else:
            return (0, -1)

        return (total_value, np.float(woe_l < woe_u))

    def split_var_bin(self, tabname, varname, woe_direct):
        # t1 = np.unique(tabname[varname])
        t1 = np.unique(np.percentile(tabname[varname], np.arange(1, 100, 2)))
        if len(t1) > 1:
            t2 = [round((t1[i] + t1[i + 1]) / 2.0, 4)
                  for i in range(len(t1) - 1)]  # 切割点平均值
            t3 = [(i, self.get_iv(tabname, varname, i)) for i in t2]
            t3_1 = [j for j in t3 if j[1][1] ==
                    woe_direct and j[1][0] >= 0.001]  # 与首次切割方向相同
            if len(t3_1) > 0:

                t3_max = [i[1][0] for i in t3_1]
                max_index = t3_max.index(max(t3_max))

                split_value = t3_1[max_index][0]
                gain_iv = max(t3_max)
                tab_l = tabname[tabname[varname] <= split_value]
                tab_u = tabname[tabname[varname] > split_value]

                split_value_i, max_iv_i = self.split_var_bin(
                    tab_l, varname, woe_direct)
                split_value_j, max_iv_j = self.split_var_bin(
                    tab_u, varname, woe_direct)
            else:
                return [], []
        else:
            return [], []
        # return split_value.append(split_value_i.append(split_value_j))
        return [split_value] + split_value_i + split_value_j, [gain_iv] + max_iv_i + max_iv_j

    def first_split(self, tabname, varname):
        # 第一次决定分割的woe方向
        t1 = np.unique(tabname[varname])
        t2 = [round((t1[i] + t1[i + 1]) / 2.0, 4)
              for i in range(len(t1) - 1)]  # 切割点平均值
        t3 = [(i, self.get_iv(tabname, varname, i)) for i in t2]
        # t3_1 = [j for j in t3 if j[1][0] >= 0.001]
        t3_max = [i[1][0] for i in t3]
        max_index = t3_max.index(max(t3_max))

        return t3[max_index][0], t3[max_index][1][1]

    def get_nulldata_mapiv(self, tab, t0, t1):
        null_t1 = tab.target.sum()
        null_t0 = len(tab) - null_t1
        null_p1r = null_t1 / t1 + 1e-6
        null_p0r = null_t0 / t0 + 1e-6
        null_woe = np.log(null_p1r / null_p0r)
        null_iv = (null_p1r - null_p0r) * null_woe
        nullmapiv = pd.DataFrame(
            {'bin': 0, 'll': np.nan, 'ul': np.nan, 'p1': null_t1,
                'p0': null_t0, 'woe': null_woe, 'iv0': null_iv},
            index=[0])
        # nullmapiv = pd.DataFrame({'bin':[0], 'll':[np.nan], 'ul':[np.nan], 'p1':[null_t1], 'p0':[null_t0], 'woe':[null_woe], 'iv0':[null_iv]})

        return nullmapiv

    def get_firstnull_mapiv(self, tab, t0, t1):
        n_null_t1 = tab.target.sum()
        n_null_t0 = len(tab) - n_null_t1
        n_null_p1r = n_null_t1 / t1 + 1e-6
        n_null_p0r = n_null_t0 / t0 + 1e-6
        n_null_woe = np.log(n_null_p1r / n_null_p0r)
        n_null_iv = (n_null_p1r - n_null_p0r) * n_null_woe
        n_nullmapiv = pd.DataFrame(
            {'bin': 1, 'll': -np.inf, 'ul': np.inf, 'p1': n_null_t1,
                'p0': n_null_t0, 'woe': n_null_woe, 'iv0': n_null_iv},
            index=[1])
        # nullmapiv = pd.DataFrame({'bin':[0], 'll':[np.nan], 'ul':[np.nan], 'p1':[null_t1], 'p0':[null_t0], 'woe':[null_woe], 'iv0':[null_iv]})

        return n_nullmapiv

    def decession_tree_bin(self, X, y):

        clf = tree.DecisionTreeClassifier(criterion='entropy',
                                          max_leaf_nodes=self.max_bin,
                                          min_samples_leaf=self.min_num).fit(X, y)

        # basic output
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        threshold = clf.tree_.threshold
        boundary = []
        for i in range(n_nodes):
            if children_left[i] != children_right[i]:
                boundary.append(threshold[i])
        sort_boundary = sorted(boundary)
        return sort_boundary

    def split_onevar(self, tabname, varname):

        t1 = tabname.target.sum()
        t0 = len(tabname) - t1

        nulltab = tabname[pd.isnull(tabname[varname])]  # 缺失值单独一箱
        n_nulltab = tabname[pd.isnull(tabname[varname]) == 0]  # 非缺失
        if self.bin_method == 'mono':
            if len(np.unique(n_nulltab[varname])) > 1:
                split_value_1, woe_direct = self.first_split(
                    n_nulltab, varname)
                if woe_direct == -1:
                    n_nullmapiv = self.get_firstnull_mapiv(
                        n_nulltab, t0, t1)  # 非缺失值不能再成功分组
                else:
                    tab_l = n_nulltab[n_nulltab[varname] <= split_value_1]
                    tab_u = n_nulltab[n_nulltab[varname] > split_value_1]

                    split_value = [split_value_1]
                    split_value_l, max_iv_l = self.split_var_bin(
                        tab_l, varname, woe_direct)
                    split_value_u, max_iv_u = self.split_var_bin(
                        tab_u, varname, woe_direct)

                    sp_result = split_value + split_value_l + split_value_u
                    gain_iv_resulit = [1] + max_iv_l + max_iv_u

                    # 取gain_iv_resulit前max_bin-1个索引

                    # print(split_value, split_value_l, split_value_u)
                    max_bin = min(self.max_bin - 1, len(gain_iv_resulit))
                    sp_result1 = [sp_result[i] for i in np.argpartition(
                        gain_iv_resulit, -max_bin)[-max_bin:]]

                    sp_result1.sort()

                    n_nullmapiv = self.get_mapiv_result(
                        n_nulltab, varname, sp_result1, t0, t1)  # 非缺失值的分布，计算时传入t0,t1
            else:
                n_nullmapiv = pd.DataFrame()
        else:
            if len(np.unique(n_nulltab[varname])) > 1:
                sp_result = self.decession_tree_bin(
                    n_nulltab[[varname]], n_nulltab['target'])
                n_nullmapiv = self.get_mapiv_result(
                    n_nulltab, varname, sp_result, t0, t1)  # 非缺失值的分布，计算时传入t0,t1
            else:
                n_nullmapiv = pd.DataFrame()

        if len(nulltab) > 0:
            nullmapiv = self.get_nulldata_mapiv(nulltab, t0, t1)
        else:
            nullmapiv = pd.DataFrame()

        all_mapiv = pd.concat([nullmapiv, n_nullmapiv], axis=0)
        all_mapiv['varname'] = varname
        # print(sp_result)
        return all_mapiv

    def split_data(self):
        if self.target != 'target':
            self.indata.rename(columns={self.target: 'target'}, inplace=True)
            self.target = 'target'
        data_df = self.indata.copy()

        assert self.bin_method in ['mono', 'tree']
        assert self.alg_method in ['iv', 'gini', 'entropy']
        self.check_y(data_df, 'target')

        order = ['varname', 'bin', 'll', 'ul', 'p0',
                 'p1', 'total', 'woe', 'iv0', 'iv']  # 指定输出列名
        feature_list = [c for c in data_df if (
            data_df[c].dtype.kind in ('i', 'f')) & ('target' not in c)]

        mapiv = pd.DataFrame()
        no_cores = mp.cpu_count() - 1
        pool = mp.Pool(processes=no_cores)
        # for var_i in feature_list:
        #     indata = self.indata[[var_i, 'target']]
        #     indata[var_i] = indata[var_i].apply(lambda x : round(x, 3))
        #     mapiv1 = self.split_onevar(indata, var_i)
        #     mapiv = pd.concat([mapiv1, mapiv], axis=0)

        args = zip(
            [data_df[[var_i, 'target']].apply(
                lambda x: round(x, 3)) for var_i in feature_list],
            feature_list,
        )
        bins = list(pool.starmap(self.split_onevar, args))

        for mi in bins:
            mapiv = mapiv.append(mi)
        mapiv['woe'] = round(mapiv['woe'], 6)
        mapiv['iv0'] = round(mapiv['iv0'], 6)
        m1 = mapiv.groupby(['varname'])[['iv0']].sum()
        m1.rename(columns={'iv0': 'iv'}, inplace=True)
        m1.reset_index(level=0, inplace=True)
        mapiv_t = mapiv.merge(m1, on='varname')

        mapiv_t['total'] = mapiv_t['p0'] + mapiv_t['p1']

        mapiv_t = mapiv_t[order]

        # 仅保留最小分箱数满足要求的分组(缺失那箱可以不用满足该条件)

        min_per = mapiv_t[mapiv_t['bin'] > 0].groupby(['varname'])[
            ['total']].min()
        min_per.reset_index(level=0, inplace=True)
        min_per['flag'] = min_per['total'] >= self.min_num

        mapiv_select = mapiv_t.merge(
            min_per[['varname', 'flag']], how='left', on='varname')
        mapiv_select1 = mapiv_select[mapiv_select['flag'] == 1]

        return mapiv_select1.drop('flag', axis=1)

    def apply_woetab(self, indata, mapiv):
        outdata = indata.copy()
        var_list = np.unique(mapiv['varname'])
        for vi in var_list:
            if vi in outdata.columns:
                ul_list = mapiv[mapiv['varname'] == vi]['ul'].values
                woe_list = mapiv[mapiv['varname'] == vi]['woe'].values
                kwds = {"sp_list": ul_list, "woe_list": woe_list}
                outdata['W_{}'.format(vi)] = outdata[vi].apply(
                    self.woe_trans, **kwds)
            else:
                continue

        outdata_col = list(outdata)
        outdata_col_woe = [i for i in outdata_col if i.startswith('W_')]
        # outdata_woe = outdata[['target'] + outdata_col_woe]
        outdata_woe = outdata[outdata_col_woe]  # 不再提供‘target’
        return outdata_woe

    def cal_cate_woe(self, var):

        data_df = self.indata.copy()
        t1 = data_df[self.target].sum()
        t0 = len(data_df) - t1
        cnt0 = data_df.groupby(var)[[self.target]].count()
        cnt0.rename(columns={'target': 'total'}, inplace=True)
        sum0 = data_df.groupby(var)[[self.target]].sum()
        sum0.rename(columns={'target': 'bad'}, inplace=True)

        map0 = pd.concat([cnt0, sum0], axis=1)
        map0['good'] = map0['total'] - map0['bad']
        map0['p1_r'] = map0['bad'] / t1 + 1e-6
        map0['p0_r'] = map0['good'] / t0 + 1e-6
        map0['woe'] = np.log(map0['p1_r'] / map0['p0_r'])
        map0.reset_index(inplace=True)

        return map0[[var, 'woe']]

    def split_data_cate(self):

        if self.target != 'target':
            self.indata.rename(columns={self.target: 'target'}, inplace=True)
            self.target = 'target'
        data_df = self.indata.copy()
        order = ['varname', 'bin', 'll', 'ul', 'p0',
                 'p1', 'total', 'woe', 'iv0', 'iv']  # 指定输出列名
        feature_list = [c for c in data_df if (
            data_df[c].dtype.kind not in ('i', 'f'))]
        mapiv = pd.DataFrame()

        all_map_order = ['varname', 'val', 'woe', 'new_woe']
        all_map = pd.DataFrame()
        for var_i in feature_list:
            # print(var_i)
            map_i = self.cal_cate_woe(var_i)
            map_dict = dict(map_i.values)
            map_i.rename(columns={var_i: 'val'}, inplace=True)
            map_i['varname'] = var_i
            data_df['tempW_{}'.format(var_i)] = data_df[var_i].map(map_dict)

            indata = data_df[['tempW_{}'.format(var_i), 'target']]
            indata['tempW_{}'.format(var_i)] = indata['tempW_{}'.format(
                var_i)].apply(lambda x: round(x, 4))
            mapiv1 = self.split_onevar(indata, 'tempW_{}'.format(var_i))
            mapiv1['varname'] = var_i
            mapiv = pd.concat([mapiv1, mapiv], axis=0)

            ul_list = mapiv['ul'].values
            woe_list = mapiv['woe'].values
            kwds = {"sp_list": ul_list, "woe_list": woe_list}
            map_i['new_woe'] = map_i['woe'].apply(self.woe_trans, **kwds)
            all_map = all_map.append(map_i)

        mapiv['woe'] = round(mapiv['woe'], 6)
        mapiv['iv0'] = round(mapiv['iv0'], 6)
        m1 = mapiv.groupby(['varname'])[['iv0']].sum()
        m1.rename(columns={'iv0': 'iv'}, inplace=True)
        m1.reset_index(level=0, inplace=True)
        mapiv_t = mapiv.merge(m1, on='varname')

        mapiv_t['total'] = mapiv_t['p0'] + mapiv_t['p1']
        mapiv_t = mapiv_t[order]

        # 仅保留最小分箱数满足要求的分组(缺失那箱可以不用满足该条件)

        min_per = mapiv_t[mapiv_t['bin'] > 0].groupby(['varname'])[
            ['total']].min()
        min_per.reset_index(level=0, inplace=True)
        min_per['flag'] = min_per['total'] >= self.min_num

        mapiv_select = mapiv_t.merge(
            min_per[['varname', 'flag']], how='left', on='varname')
        mapiv_select1 = mapiv_select[mapiv_select['flag'] == 1]

        return mapiv_select1.drop('flag', axis=1), all_map[all_map_order]

    def apply_woetab_cate(self, indata, all_mapdict):

        outdata = indata.copy()
        var_list = np.unique(all_mapdict['varname'])
        for vi in var_list:
            if vi in outdata.columns:
                map_dict = dict(
                    all_mapdict[all_mapdict['varname'] == vi][['val', 'new_woe']].values)
                outdata['W_{}'.format(vi)] = outdata[vi].map(map_dict)
            else:
                continue

        outdata_col = list(outdata)
        outdata_col_woe = [i for i in outdata_col if i.startswith('W_')]
        # outdata_woe = outdata[['target'] + outdata_col_woe]
        outdata_woe = outdata[outdata_col_woe]  # 不再提供‘target’
        return outdata_woe


def coverage_static(desc):
    desc['cnt'] = 1
    desc['coverage'] = desc['missing_rate'].apply(lambda x: 1-x)
    desc['coverage_seg'] = pd.cut(desc['coverage'], bins=[
                                  0, 0.2, 0.4, 0.8, 1], right=True, include_lowest=True)
    return desc


def iv_static(mono_woe):
    iv_result = mono_woe[['varname', 'iv', 'target']].drop_duplicates()
    iv_result['cnt'] = 1
    iv_result['iv_seg'] = pd.cut(iv_result['iv'], bins=[
                                  0, 0.02, 0.1, 0.3, 0.5, 99], right=True, include_lowest=True)
    iv_result.rename(columns={'varname': 'column'}, inplace=True)
    return iv_result


def score_card(prob):
    base_score = 600
    base_odds = 1/50
    PDO = 50
    B = PDO*1.0/math.log(2)
    A = base_score + B*math.log(base_odds)
    return round(A - B*math.log(prob / (1-prob+1e-20)), 0)


def model_comparison_effect_cross(df, base_model_name, compare_model_name, target, bins):
    bins_label = ["(%d%%, %d%%]" % (i*100/bins, 100/bins*(i+1)) for i in range(bins)]
    df_temp = df[[target, base_model_name, compare_model_name]].query("{0}>=0".format(target))
    df_temp[base_model_name] = pd.qcut(df_temp[base_model_name], q=bins, duplicates='drop')
    df_temp[compare_model_name] = pd.qcut(df_temp[compare_model_name], q=bins, duplicates='drop')

    crosstab_1 = pd.crosstab(
        df_temp[base_model_name],
        df_temp[compare_model_name],
        df_temp[target],
        aggfunc='mean',
        margins=True
    )
    crosstab_2 = pd.crosstab(
        df_temp[base_model_name],
        df_temp[compare_model_name],
        df_temp[target],
        aggfunc='count',
        margins=True
    )
    crosstab = pd.concat([crosstab_1.reset_index(), crosstab_2.reset_index()])
    return crosstab


def step_model_build(dev_sample, oot_sample, train, valid, input_variables, target, parameters_1, parameters_2, importance_cumsum_threshold=0.9):
    xgb_model = XGBClassifier(**parameters_1)
    xgb_model.fit(train[input_variables], train[target], eval_set=[(train[input_variables], train[target]), (valid[input_variables], valid[target])],
                  eval_metric='auc', early_stopping_rounds=20, verbose=False)

    shap_values = xgb_model.get_booster().predict(
        xgb.DMatrix(dev_sample[input_variables]), pred_contribs=True)
    shap_df = pd.DataFrame(
        np.abs(shap_values[:, :-1]), columns=input_variables)

    shap_imp = shap_df.mean().sort_values(ascending=False).reset_index()
    shap_imp.columns = ['Feature', 'Shap_Importance']
    shap_imp = shap_imp[shap_imp['Shap_Importance'] > 0]

    shap_imp['Importance_Cumsum'] = (
        shap_imp['Shap_Importance']/shap_imp['Shap_Importance'].sum()).cumsum()
    sel_num = shap_imp.query(f"Importance_Cumsum >= {importance_cumsum_threshold}").index.min()
    sel_cols = [x for x in shap_imp.head(sel_num)['Feature'].tolist()]
    xgb_model = XGBClassifier(**parameters_2)

    xgb_model.fit(train[sel_cols], train[target], eval_set=[(train[sel_cols], train[target]), (oot_sample[sel_cols], oot_sample[target])],
                  eval_metric='auc', early_stopping_rounds=20, verbose=False)
    shap_values = xgb_model.get_booster().predict(
        xgb.DMatrix(dev_sample[sel_cols]), pred_contribs=True)
    shap_df = pd.DataFrame(
        np.abs(shap_values[:, :-1]), columns=sel_cols)

    shap_imp = shap_df.mean().sort_values(ascending=False).reset_index()
    shap_imp.columns = ['Feature', 'Shap_Importance']
    return xgb_model, sel_cols, shap_imp


class ThirdEvalutor(object):
    def __init__(self, data, base_variables, evaluate_variables, evaluate_score, target_list, project_name, save_directory):
        self.data = data
        self.base_variables = base_variables
        self.evaluate_variables = evaluate_variables
        self.evaluate_score = evaluate_score
        self.target_list = target_list
        self.project_name = project_name
        self.save_directory = save_directory

    def sample_distribution(self, target):

        def f_mi_1(x, proba_name, target):
            d = []
            d.append(x['cnt'].sum())
            d.append(x.query(f"{target}==0")['cnt'].sum())
            d.append(x.query(f"{target}==-1")['cnt'].sum())
            d.append(x.query(f"{target}==1")['cnt'].sum())
            d.append(x.query(f"{target}==-1")['cnt'].sum()/x['cnt'].sum())
            d.append(x.query(f"{target}==1")['cnt'].sum()/x.query(f"{target}!=-1")['cnt'].sum())
            d.append(x[target].replace(-1, 0).sum()/x['cnt'].sum())
            return pd.Series(d, index=['总样本量', '白样本量', '灰样本量', '黑样本量', '灰样本率', '黑样本率_不含灰', '黑样本率_灰当白'])

        if not os.path.exists(self.save_directory + "{0}_Sample_Distribution.csv".format(self.project_name)):
            self.data['cnt'] = 1
            if "apply_month" in self.data.columns:
                sample_dist = self.data.query(f"{target} == [-1,0,1]").groupby(['source', 'apply_month']).apply(lambda x: f_mi_1(x, None, target)).reset_index()
            else:
                sample_dist = self.data.query(f"{target} == [-1,0,1]").groupby(['source']).apply(lambda x: f_mi_1(x, None, target)).reset_index()

            self.sample_dist = sample_dist
            sample_dist.to_csv(self.save_directory + "{0}_Sample_Distribution.csv".format(self.project_name), index=False)
        else:
            self.sample_dist = pd.read_csv(self.save_directory + "{0}_Sample_Distribution.csv".format(self.project_name))
        return self.sample_dist

    def coverage(self, product_label=None):
        assert isinstance(product_label, pd.DataFrame) or product_label is None, "product label must be a dataframe if it's not none."
        if not os.path.exists(self.save_directory + "{0}_Data_Report.csv".format(self.project_name)):
            report = data_report(self.data)
            report.to_csv(self.save_directory + "{0}_Data_Report.csv".format(self.project_name), index=False)
            self.data_report = report
        else:
            self.data_report = pd.read_csv(self.save_directory + "{0}_Data_Report.csv".format(self.project_name))
        report = coverage_static(self.data_report)
        evalute_features = self.evaluate_variables + self.evaluate_score
        if product_label is None:
            coverage_disribution = report.query("column==@evalute_features").groupby(
                ['coverage_seg']).agg({'cnt': 'sum'}).reset_index()
        else:
            report = report.merge(product_label, on='column', how='left')
            coverage_disribution = report.query("column==@evalute_features").groupby(['product', 'coverage_seg']).agg({'cnt': 'sum'}).reset_index()
        self.coverage_disribution = coverage_disribution
        return self.coverage_disribution 

    def iv_calculator(self, bin_method='mono', product_label=None):
        assert isinstance(product_label, pd.DataFrame) or product_label is None, "product label must be a dataframe if it's not none."
        if not os.path.exists(self.save_directory + "{0}_WoE_Distribution.csv".format(self.project_name)):
            woe_detail_result = pd.DataFrame()
            for target in self.target_list:
                temp = self.data.query(f"{target} == [0, 1]")[
                    self.evaluate_variables + self.evaluate_score+[target]]
                mono_woe = woe_bin(indata=temp, target=target, min_group_rate=0.05, max_bin=6, bin_method=bin_method, alg_method='iv')
                mapiv_temp = mono_woe.split_data()
                mapiv_temp.sort_values(['iv', 'varname', 'bin'], ascending=[False, False, True], inplace=True)
                mapiv_temp.insert(7, '%bad', mapiv_temp['p1']/mapiv_temp['total'])
                # mapiv_temp['%bad'] = mapiv_temp['p1']/mapiv_temp['total']
                mapiv_temp['target'] = target
                woe_detail_result = pd.concat([woe_detail_result, mapiv_temp])
            self.woe_detail_result = woe_detail_result
            woe_detail_result.to_csv(self.save_directory + "{0}_WoE_Distribution.csv".format(self.project_name), index=False)
        else:
            self.woe_detail_result = pd.read_csv(self.save_directory + "{0}_WoE_Distribution.csv".format(self.project_name))
        iv_result = iv_static(self.woe_detail_result)
        if product_label is None:
            iv_distribution = iv_result.groupby(
                ['iv_seg', 'target']).agg({'cnt': 'sum'}).reset_index()
        else:
            iv_result = iv_result.merge(product_label, on='column', how='left')
            iv_distribution = iv_result.groupby(['product', 'iv_seg', 'target']).agg({'cnt': 'sum'}).reset_index()
        self.iv_distribution = iv_distribution

        return self.woe_detail_result, iv_distribution

    def model_gain_evalutor(self, target, model_params_1, model_params_2, main_score, bins=5):
        assert 'source' in self.data.columns, "model data must contain source column"
        self.data['cnt'] = 1
        dev_sample = self.data.query(
            f"source=='0.Train' and {target} == [0, 1]").reset_index(drop=True)
        oot_sample = self.data.query(
            f"source=='1.OOT' and {target} == [0, 1]").reset_index(drop=True)
        print("DEV SAMPLE SIZE: %d, BAD: %d, BAD RATIO: %.4f" %
              (dev_sample.shape[0], dev_sample[target].sum(), dev_sample[target].mean()))
        print("OOT SAMPLE SIZE: %d, BAD: %d, BAD RATIO: %.4f" %
              (oot_sample.shape[0], oot_sample[target].sum(), oot_sample[target].mean()))
        train, validation = train_test_split(
            dev_sample, test_size=0.3, random_state=42)

        variable_filter = self.data_report[self.data_report['unique'] > 1]
        variable_filter = variable_filter[~((variable_filter['dtype'] == 'object'))]
        variable_filter = variable_filter[variable_filter['missing_rate'] <= 0.85]
        variable_filter = variable_filter[variable_filter['dtype'] != 'datetime64[ns]']

        condicate_base_variables = [x for x in self.base_variables if x in variable_filter['column'].tolist()]
        condicate_evalute_variables = [x for x in self.evaluate_variables if x in variable_filter['column'].tolist()]
        condicate_evalute_score = [x for x in self.evaluate_score if x in variable_filter['column'].tolist()]

        # base model
        xgb_base_model, base_sel_cols, _ = step_model_build(dev_sample, oot_sample, train, validation, condicate_base_variables, target, model_params_1, model_params_2)
        self.data['base_proba'] = xgb_base_model.predict_proba(self.data[base_sel_cols])[:, 1]

        # sub model
        xgb_sub_model, sub_sel_cols, _ = step_model_build(dev_sample, oot_sample, train, validation, condicate_evalute_variables + condicate_evalute_score, target, model_params_1, model_params_2)
        self.data['sub_proba'] = xgb_sub_model.predict_proba(self.data[sub_sel_cols])[:, 1]

        # 增益评估
        gain_condicate_variables = condicate_base_variables + condicate_evalute_variables + condicate_evalute_score
        xgb_gain_model, gain_sel_cols, shap_imp = step_model_build(dev_sample, oot_sample, train, validation, gain_condicate_variables, target, model_params_1, model_params_2)
        self.data['gain_proba'] = xgb_gain_model.predict_proba(self.data[gain_sel_cols])[:, 1]
        self.gain_sel_cols = gain_sel_cols
        self.shap_imp = shap_imp

        # 交叉增益
        self.data['sub_score'] = self.data['sub_proba'].apply(score_card)
        self.crosstab = model_comparison_effect_cross(df=self.data.query(f"{target}==[0, 1] and source == '1.OOT' and {main_score}=={main_score}"), base_model_name=main_score, compare_model_name='sub_score', target=target, bins=bins)

        # evalute doc
        gain_evalute_result = pd.DataFrame()
        for target_ in self.target_list:
            for score in ['sub_proba', 'base_proba', 'gain_proba']:
                temp = self.data.query(f"{target_}==[-1,0,1]").groupby(["source"]).apply(
                    lambda x: f_evalutor(x, score, target_)).reset_index()
                temp['Label'] = target_
                temp['Model'] = score[:4]
                gain_evalute_result = pd.concat([gain_evalute_result, temp])

        self.gain_evalute_result = gain_evalute_result

    def correlator(self, compare_list):
        third_corr = self.data.query("source == '1.OOT'")[compare_list].corr(method='spearman')
        top_corr = self.data.query("source == '1.OOT'")[self.gain_sel_cols[:15]].corr(method='spearman')
        self.third_corr = third_corr.reset_index()
        self.top_corr = top_corr.reset_index()

    def doc_format(self):
        writer = pd.ExcelWriter(self.save_directory + "{0}_Model_Report.xlsx".format(self.project_name))
        workbook = writer.book
        # cell_format = workbook.add_format({'font_name': '微软雅黑', 'font_size': 9, 'bold': True, 'font_color': 'red'})
        # Formater
        text_fmt = workbook.add_format({'font_name': '微软雅黑', 'font_size': 9})
        percent_fmt = workbook.add_format(
            {'font_name': '微软雅黑', 'font_size': 9, 'num_format': '0.00%'})
        int_fmt = workbook.add_format(
            {'font_name': '微软雅黑', 'font_size': 9, 'num_format': '#,##0'})
        float_fmt = workbook.add_format(
            {'font_name': '微软雅黑', 'font_size': 9, 'num_format': '#,##0.00'})
        float_fmt_1 = workbook.add_format(
            {'font_name': '微软雅黑', 'font_size': 9, 'num_format': '#,##0.0000'})
        fill_fmt = workbook.add_format({'bg_color': '#FCE4D6'})

        header_1 = {
            'bold': True,  # 粗体
            'font_name': '微软雅黑',
            'font_size': 9,
            'font_color': '#FFFFFF',
            'border': False,  # 边框线
            'align': 'center',  # 水平居中
            'valign': 'vcenter',  # 垂直居中
            'bg_color': '#1F4E78'  # 背景颜色
        }
        header_fmt = workbook.add_format(header_1)

        header_2 = {
            'bold': True,  # 粗体
            'font_name': '微软雅黑',
            'border': False,
            'font_size': 9,
            'align': 'center',  # 水平居中
            'valign': 'vcenter'  # 垂直居中
        }
        header_fmt_2 = workbook.add_format(header_2)
        header_fmt_2.set_bottom()
        date_fmt = workbook.add_format(
            {'bold': True, 'font_size': 9, 'font_name': u'微软雅黑', 'num_format': 'yyyy-mm',
                'valign': 'vcenter', 'align': 'center'})

        # 1.Sample Distribution
        self.sample_dist.to_excel(writer, sheet_name=u'1.Sample_Distribution', encoding='utf8',
                             header=False, index=False, startcol=1, startrow=1, freeze_panes=(1, 0))

        worksheet1 = writer.sheets[u'1.Sample_Distribution']
        for col_num, value in enumerate(self.sample_dist.columns.values):
            worksheet1.write(1, col_num+1, value, header_fmt)
            
        worksheet1.set_column('A:A', 1)
        for col in 'B':
            worksheet1.set_column(f'{col}:{col}', 9, text_fmt)
        for col in 'C':
            worksheet1.set_column(f'{col}:{col}', 9, date_fmt)
        worksheet1.set_column('D:G', 9, int_fmt)
        worksheet1.set_column('H:J', 9, percent_fmt)
        # 2.Data Report
        self.data_report.to_excel(writer, sheet_name=u'2.Data_Report', encoding='utf8',
                                    header=False, index=False, startcol=1, startrow=1, freeze_panes=(1, 0))
        worksheet2 = writer.sheets[u'2.Data_Report']
        for col_num, value in enumerate(self.data_report.columns.values):
            worksheet2.write(0, col_num+1, value, header_fmt)
            
        worksheet2.set_column('A:A', 1)
        for col in 'BCU':
            worksheet2.set_column(f'{col}:{col}', 9, text_fmt)
        for col in 'DEGIS':
            worksheet2.set_column(f'{col}:{col}', 9, int_fmt)
        for col in 'FHT':
            worksheet2.set_column(f'{col}:{col}', 9, percent_fmt)
        worksheet2.set_column('J:R', 9, float_fmt)
        d_len = self.data_report.shape[0]+1
        worksheet2.conditional_format(f'F2:F{d_len}', {'type': 'data_bar',
                                                    'bar_solid': True,
                                                    })
        # 2.1 Coverage Distribution
        self.coverage_disribution.to_excel(writer, sheet_name=u'2.1.Coverage_Distribution', encoding='utf8', index=False, startcol=1, startrow=1, freeze_panes=(1, 0))
        worksheet2_1 = writer.sheets[u'2.1.Coverage_Distribution']
        for col_num, value in enumerate(self.coverage_disribution.columns.values):
            worksheet2_1.write(1, col_num+1, value, header_fmt)
            
        worksheet2_1.set_column('A:A', 1)
        for col in 'BC':
            worksheet2_1.set_column(f'{col}:{col}', 9, text_fmt)
        for col in 'D':
            worksheet2_1.set_column(f'{col}:{col}', 9, int_fmt)
        # 3.WOE
        self.woe_detail_result.to_excel(writer, sheet_name='3.BIN_WoE', encoding='utf8',
                                    header=False, index=False, startcol=1, startrow=1, freeze_panes=(1, 0))
        worksheet3 = writer.sheets['3.BIN_WoE']
        for col_num, value in enumerate(self.woe_detail_result.columns.values):
            worksheet3.write(0, col_num+1, value, header_fmt_2)

        worksheet3.set_column('A:A', 1)
        for col in 'BM':
            worksheet3.set_column(f'{col}:{col}', 9, text_fmt)
        for col in 'CFGH':
            worksheet3.set_column(f'{col}:{col}', 9, int_fmt)
        for col in 'I':
            worksheet3.set_column(f'{col}:{col}', 9, percent_fmt)
        for col in 'DEJKL':
            worksheet3.set_column(f'{col}:{col}', 9, float_fmt_1)

        for target_ in self.target_list:
            temp_ = self.woe_detail_result.query(f"target == '{target_}'")
            for i, col in enumerate(temp_['varname'].drop_duplicates().tolist()):
                temp = temp_.query(f"varname == '{col}'")
                row_min, row_max = min(temp.index)+2, max(temp.index)+2
                if i % 2 == 0:
                    worksheet3.conditional_format(f"B{row_min}:M{row_max}", {
                                                'type': 'formula', 'criteria': '1>0', "format": fill_fmt})
                    worksheet3.conditional_format(f'I{row_min}:I{row_max}', {'type': 'data_bar',
                                                                            'bar_solid': True, 'bar_color': '#FFB628'
                                                                            })
                else:
                    worksheet3.conditional_format(f'I{row_min}:I{row_max}', {'type': 'data_bar',
                                                                            'bar_solid': True, 'bar_color': '#FFB628'
                                                                            })

        # 3.1 IV Distribution
        self.iv_distribution.to_excel(writer, sheet_name=u'3.1.IV_Distribution', encoding='utf8', index=False, startcol=1, startrow=1, freeze_panes=(1, 0))
        worksheet3_1 = writer.sheets[u'3.1.IV_Distribution']
        for col_num, value in enumerate(self.iv_distribution.columns.values):
            worksheet3_1.write(1, col_num+1, value, header_fmt)
            
        worksheet3_1.set_column('A:A', 1)
        for col in 'BCD':
            worksheet3_1.set_column(f'{col}:{col}', 9, text_fmt)
        for col in 'E':
            worksheet3_1.set_column(f'{col}:{col}', 9, int_fmt)

        # 4.Gain Result
        self.gain_evalute_result.to_excel(writer, sheet_name=u'4.Gain_Result', encoding='utf8', index=False, startcol=1, startrow=1, freeze_panes=(1, 0))
        worksheet4 = writer.sheets[u'4.Gain_Result']
        for col_num, value in enumerate(self.gain_evalute_result.columns.values):
            worksheet4.write(1, col_num+1, value, header_fmt)
            
        worksheet4.set_column('A:A', 1)
        for col in 'BNO':
            worksheet4.set_column(f'{col}:{col}', 9, text_fmt)
        for col in 'CD':
            worksheet4.set_column(f'{col}:{col}', 9, int_fmt)
        for col in 'GHIJKLM':
            worksheet4.set_column(f'{col}:{col}', 9, float_fmt)
        for col in 'EF':
            worksheet4.set_column(f'{col}:{col}', 9, percent_fmt)
        # 4.1 Cross Analysis
        try:
            self.crosstab.to_excel(writer, sheet_name=u'4.1.Cross_Analysis', encoding='utf8', index=False, startcol=1, startrow=1, freeze_panes=(1, 0))

            worksheet4_1 = writer.sheets[u'4.1.Cross_Analysis']
            for col_num, value in enumerate(self.crosstab.columns.values):
                worksheet4_1.write(1, col_num+1, value, header_fmt)
                
            worksheet4_1.set_column('A:A', 1)
        except:
            pass

        # 4.2 Shap Importance
        self.shap_imp.to_excel(writer, sheet_name=u'4.2.Shap_Importance', encoding='utf8', index=False, startcol=1, startrow=1, freeze_panes=(1, 0))

        worksheet4_2 = writer.sheets[u'4.2.Shap_Importance']
        for col_num, value in enumerate(self.shap_imp.columns.values):
            worksheet4_2.write(1, col_num+1, value, header_fmt)
            
        worksheet4_2.set_column('A:A', 1)
        for col in 'BC':
            worksheet4_2.set_column(f'{col}:{col}', 9, text_fmt)
        for col in 'D':
            worksheet4_2.set_column(f'{col}:{col}', 9, float_fmt)

        # 5.Correlation Analysis
        self.third_corr.to_excel(writer, sheet_name=u'5.Correlation', encoding='utf8', index=False, startcol=1, startrow=1, freeze_panes=(1, 0))

        worksheet5 = writer.sheets[u'5.Correlation']
        for col_num, value in enumerate(self.third_corr.columns.values):
            worksheet5.write(1, col_num+1, value, header_fmt)

        chars = ''
        for i in range(65, 91):
            chars += chr(i)
        end_column = chars[1+self.third_corr.shape[1]]
        end_row = 1+self.third_corr.shape[1]

        worksheet5.set_column('A:A', 1)
        worksheet5.set_column('B:B', 9, text_fmt)
        worksheet5.set_column('C:Z', 9, float_fmt)
        worksheet5.conditional_format(f"C2:{end_column}{end_row}", {'type': '3_color_scale'})

        # 5.1.Top Analysis
        self.top_corr.to_excel(writer, sheet_name=u'5.1.TOP_Correlation', encoding='utf8', index=False, startcol=1, startrow=1, freeze_panes=(1, 0))

        worksheet5_1 = writer.sheets[u'5.1.TOP_Correlation']
        for col_num, value in enumerate(self.top_corr.columns.values):
            worksheet5_1.write(1, col_num+1, value, header_fmt)

        end_column = chars[1+self.top_corr.shape[1]]
        end_row = 1+self.top_corr.shape[1]

        worksheet5_1.set_column('A:A', 1)
        worksheet5_1.set_column('B:B', 9, text_fmt)
        worksheet5_1.set_column('C:Z', 9, float_fmt)
        worksheet5_1.conditional_format(f"C2:{end_column}{end_row}", {'type': '3_color_scale'})

        writer.save()
        writer.close()
