import sys
sys.path.append("/Users/yang")
sys.path.append("/home/tangyy/")

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pymysql
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import scorecardpy as sc
import pickle
from sklearn.model_selection import train_test_split
from model_tools.metrics import roc_auc_score, ks
from model_tools.ScoreCard import model_helper
from model_tools.Model import model_utils
from model_tools.Evalutor import third_evalutor
from model_tools.AutoModel import AutoXGBoost

import warnings
warnings.filterwarnings("ignore")


model_data = pd.read_csv("../data/贷中回溯行为变量6期10D变量明细.csv")
model_data = model_data.query("(current_due_amt<=0 or current_due_amt!=current_due_amt) and (his_maxdueday<15 or his_maxdueday!=his_maxdueday)")
print(model_data.shape)
model_data['his_maxdueday'].describe()

model_data.head()
model_data['id_no'].value_counts()

model_data.columns[:30]

view_cols = [
    'tx_date', 'tx_time', 'channel_key', 'loan_no', 'id_no', 'dn_seq',
    'apply_amt', 'distr_amt', 'credit_amt', 'first_loan_date', 'mob_i',
    'is_loan', 'is_risk_pass', 'is_settled', 'is_wholesale_fund_cust',
]

model_data.query("id_no=='371102198703258120'")[view_cols]


model_data['mob_i'].value_counts()
model_data = model_data.query("mob_i>=0").reset_index(drop=True)
model_data.shape

condicate_variables = [i for i in model_data.columns if model_data[i].dtypes !='object' and i.find('is_stage')<0 and i not in ['apply_amt','distr_amt','is_risk_pass','is_loan']]
condicate_variables = [x for x in condicate_variables if x not in ['credit_amt','mob_i', 'asc_row']]

model_data['apply_month'] = model_data['tx_date'].apply(lambda x: x[:7])
# model_data = model_data.query("mob_i >= 3").reset_index(drop=True)

target = 'is_stage6_due10'

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

model_data['cnt'] = 1
model_data.query(f"{target}==[0, 1, -1]")[['apply_month', 'cnt', target]].groupby(['apply_month']).apply(lambda x: f_mi_1(x, None, target))

# data_report = model_helper.data_report(model_data[condicate_variables])
# data_report.to_csv("../doc/data_report_trade_1.csv", index=False)
data_report = pd.read_csv("../doc/data_report_trade_1.csv")

input_variables = data_report.query("dtype!='object' and missing_rate<=0.85 and unique>1")['column'].tolist()

import joblib
import math

def score_card(prob):
    base_score = 600
    base_odds = 1/50
    PDO = 50
    B = PDO*1.0/math.log(2)
    A = base_score + B*math.log(base_odds)
    return round(A - B*math.log( prob/ (1-prob+1e-20)),0)


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
    return (rj_sample_1/adjust_bad_rate)/(rj_sample_1/adjust_bad_rate+rj_sample_0/adjust_good_rate)*\
    (1+(rj_sample_0/adjust_good_rate +ps_sample_0/adjust_good_rate)/(rj_sample_1/adjust_bad_rate+ps_sample_1/adjust_bad_rate))

def f_evalutor_3(x, proba_name, target, input_type='score'):
    d = []
    d.append(x['cnt'].sum())
    d.append(x.query(f"{target}==[0, 1]")[target].sum())
    d.append(x.query(f"{target}==[0, 1]")[target].mean())
    d.append(x[target].replace(-1, 0).mean())
    if input_type=='score':
        d.append(round(roc_auc_score(x.query(f"{target}!=-1")[target], -1*x.query(f"{target}!=-1")[proba_name]), 3))
    else:
        d.append(round(roc_auc_score(x.query(f"{target}!=-1")[target], x.query(f"{target}!=-1")[proba_name]), 3))
    d.append(round(ks(x.query(f"{target}!=-1")[target], x.query(f"{target}!=-1")[proba_name]),3))
    if input_type=='score':
        d.append(round(roc_auc_score(x[target].replace(-1, 0), -1*x[proba_name]),3))
    else:
        d.append(round(roc_auc_score(x[target].replace(-1, 0), x[proba_name]),3))
    d.append(round(ks(x[target].replace(-1, 0), x[proba_name]),3))
    # capture@top10%
    if input_type=='score':
        # d.append(round(capture_topk(x.query(f"{target}!=-1")[target].values, -1*x.query(f"{target}!=-1")[proba_name].values, 10), 3))
        # lift@top10%
        d.append(round(lift_topk(x.query(f"{target}!=-1")[target].values, -1*x.query(f"{target}!=-1")[proba_name].values, 10), 3))
        d.append(round(lift_topk(x.query(f"{target}!=-1")[target].values, -1*x.query(f"{target}!=-1")[proba_name].values, 20), 3))
        d.append(round(lift_topk(x.query(f"{target}!=-1")[target].values, -1*x.query(f"{target}!=-1")[proba_name].values, 30), 3))
        # lift_repair@top10%
        # d.append(round(lift_repair_topk(x.query(f"{target}!=-1")[target].values, -1*x.query(f"{target}!=-1")[proba_name].values, 1, 0.72, 10), 3))
    else:
        # d.append(round(capture_topk(x.query(f"{target}!=-1")[target].values, x.query(f"{target}!=-1")[proba_name].values, 10), 3))
        d.append(round(lift_topk(x.query(f"{target}!=-1")[target].values, x.query(f"{target}!=-1")[proba_name].values, 10), 3))
        d.append(round(lift_topk(x.query(f"{target}!=-1")[target].values, x.query(f"{target}!=-1")[proba_name].values, 20), 3))
        d.append(round(lift_topk(x.query(f"{target}!=-1")[target].values, x.query(f"{target}!=-1")[proba_name].values, 30), 3))
        # lift_repair@top10%
        #d.append(round(lift_repair_topk(x.query(f"{target}!=-1")[target].values, x.query(f"{target}!=-1")[proba_name].values, 1, 0.72, 10), 3))
    return pd.Series(d, index=['#Count', '#Bad', '%Bad', '%Bad(灰当白)', 'AUC', 'KS', 'AUC(灰当白)', 'KS(灰当白)', 'Lift@top10%', 'Lift@top20%', 'Lift@top30%'])

mono_woe = third_evalutor.woe_bin(model_data.query(f"mob_i>=3 and {target}==[0,1]")[input_variables+[target]], target, min_group_rate=0.05, max_bin=6, bin_method='mono', alg_method='iv')        
mapiv = mono_woe.split_data()
mapiv['%bad'] = mapiv['p1']/(mapiv['p1']+mapiv['p0'])
