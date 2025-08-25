#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   eda.py
@Time    :   2020/09/27 15:06:47
@Author  :   tangyangyang
@Version :   1.0
@Contact :   tangyangyang-jk@360jinrong.net
@Desc    :   exploratory data analysis
'''

# import use library
import pandas as pd


class eda:
    """exploratory data analysis
    Parameters
    ----------
    data : dataframe
        analysis data object
    date_variable : str or date
        aggregate object to exploratory data variable distribution
    feature_list : list
        analysis features list
    target_list : list
        analysis target
    """
    def __init__(self, data, date_variable, feature_list, target_list):
        self.data = data
        self.date_variable = date_variable
        self.feature_list = feature_list
        self.target_list = target_list
        assert type(target_list) == list, "target list parameter must be a list."

    def target_distribution(self):
        """target distribution analysis. """
        data_copy = self.data.copy()
        data_copy['sample'] = 1
        agg_func = {}
        agg_func['sample'] = 'count'
        for target in self.target_list:
            agg_func[target] = ['sum', 'mean']
        if self.date_variable is None:
            data_copy[self.date_variable] = "all"
        distribution = data_copy.groupby(self.date_variable).agg(agg_func)
        distribution.columns = pd.Index([e[0] + "_" + e[1] for e in distribution.columns.tolist()])
        return distribution.reset_index()

    def missvalue_distribution(self):
        """miss value distribution. """
        date_list = self.data[self.date_variable].drop_duplicates().sort_values().tolist()

        missvalue_result = pd.DataFrame()
        for col in self.feature_list:
            result = self.data[[self.date_variable, col]].groupby(self.date_variable)[col].apply(lambda x: sum(x.isnull())/x.shape[0]).reset_index()
            result.rename(columns={col: 'miss_rate'}, inplace=True)
            result['miss_rate_avg'] = result['miss_rate'].mean()
            result['miss_rate_var'] = result['miss_rate'].var()
            result.insert(0, 'variable', col)
            missvalue_result = pd.concat([missvalue_result, result])
        return missvalue_result

    def psi_distribution(self):
        """single variable psi distribution. """

    def iv_distribution(self):
        """single variable iv and ks distribution. """
