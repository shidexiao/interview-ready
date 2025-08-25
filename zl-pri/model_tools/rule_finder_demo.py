#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   rule_finder_demo.py
@Time    :   2022/02/21 19:01:50
@Author  :   tangyangyang
@Contact :   tangyangyang@staff.sinaft.com
@Desc    :   RuleFinder Demo
'''

# import use library
import sys
sys.path.append("/Users/wj/xinlang_sk/模型开发/")

import pandas as pd
import numpy as np
from rule_finder import rule_extract
from model_tools.ScoreCard import model_helper

data = pd.read_csv("./data/huisu_data_all.csv").iloc[:, 1:]
label = pd.read_csv("./data/y_labels_v4.csv")
label.drop(['credit_id', 'apply_dt'], axis=1, inplace=True)
data = data.merge(label, on='user_id', how='left')
data['apply_month'] = data['apply_dt'].apply(lambda x: x[:7])
data['dpd7_mob1_fz'].value_counts(dropna=False)

feat_cols = data.columns[4:-9]
dtypes = data[feat_cols].dtypes
category_variables = dtypes[dtypes == 'object'].index.tolist()

data[category_variables] = data[category_variables].replace("none", np.nan)

category_variables_new = []
for col in category_variables:
    try:
        data[col] = data[col].astype(float)
    except:
        print(col, data[col].head())
        category_variables_new.append(col)
        pass

condicate_variables = [x for x in feat_cols if x not in category_variables_new]
data_report = model_helper.data_report(data[condicate_variables])
variable_filter = data_report[data_report['unique'] > 1]
variable_filter = variable_filter[~(
    (variable_filter['dtype'] == 'object') & (variable_filter['unique'] >= 20))]

variable_filter = variable_filter[variable_filter['missing_rate'] < 1]
variable_filter = variable_filter[variable_filter['dtype'] != 'datetime64[ns]']
condicate_variables = list(variable_filter['column'])

rule_result = rule_extract(data, condicate_variables, "dpd7_mob1_fz", comb_list=[1, 2])
rule_result.drop_duplicates().query("hit_size>=150 and lift>=2").sort_values("lift", ascending=False).drop_duplicates(["variable"], keep="first").head(20)
rule_result.to_excel("./doc/rule_result.xlsx", index=False)
