#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   rule_finder.py
@Time    :   2020/09/15 16:18:44
@Author  :   tangyangyang
@Version :   1.0
@Contact :   tangyangyang-jk@360jinrong.net
@Desc    :   rule finder
'''

# import use library
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import cpu_count


def rule_evalution(query_data, data, target):
    hit_size = query_data.shape[0]
    hit_bad_rate = query_data[target].mean()
    globle_bad_rate = data[target].mean()
    hit_rate = hit_size / data.shape[0]
    lift = hit_bad_rate / globle_bad_rate
    return hit_size, hit_rate, hit_bad_rate, lift


def rule_finder_single(in_data, variable, target):
    result = []
    sub_data = in_data[in_data[variable].notnull()].reset_index(drop=True)
    quantile_list = [0.005, 0.01, 0.02, 0.05, 0.95, 0.98, 0.99, 0.995]
    for q in quantile_list:
        threshold = sub_data[variable].quantile(q)
        rule = ""
        if q < 0.5:
            temp = sub_data.query("{0} <= @threshold".format(variable))
            rule += "<= {0}".format(threshold)
        else:
            temp = sub_data.query("{0} >= @threshold".format(variable))
            rule += ">= {0}".format(threshold)
        hit_size, hit_rate, hit_bad_rate, lift = rule_evalution(
            temp, in_data, target)
        result.append([variable, rule, hit_size, hit_rate, hit_bad_rate, lift])
    result_df = pd.DataFrame(result,
                             columns=[
                                 'variable', 'rule', 'hit_size', 'hit_rate',
                                 'hit_bad_rate', 'lift'
                             ])
    return result_df


def data_generater(in_data, variables, target):
    for col in variables:
        yield col, in_data[[col, target]]


def rule_finder_parallel(in_data, variables, target):
    n_jobs = cpu_count()
    df_list = Parallel(n_jobs=n_jobs)(
        delayed(rule_finder_single)(
            in_data=in_df, variable=variable, target=target)
        for variable, in_df in data_generater(in_data, variables, target))
    return pd.concat(df_list)


if __name__ == "__main__":
    model_data = pd.read_csv('./input/model_data.csv')
    dtypes = model_data.dtypes
    numeric_variables = [
        x for x in dtypes[dtypes != 'object'].index.tolist()
        if x not in ['fpd10', '3pd30']
    ]
    rule_result = rule_finder_parallel(model_data,
                                       numeric_variables,
                                       target='fpd10')
    result = rule_result.drop_duplicates().query(
        "hit_size>150").sort_values(by=['lift'],
                                    ascending=False)
