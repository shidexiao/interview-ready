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


model_data = pd.read_csv("../data/订单层_剔除资金批发的合同回溯还款行为_衍生变量明细.csv")
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
# data_report.to_csv("../doc/data_report_trade_2.csv", index=False)
data_report = pd.read_csv("../doc/data_report_trade_2.csv")

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

model_data = model_data.query("channel_key != ['APPZY', 'R360', 'RP']")
target = 'is_stage6_due10'
dev_sample = model_data.query(f"apply_month >='2023-11' and apply_month <= '2024-04' and mob_i>=3 and {target}==[0, 1]").reset_index(drop=True)
trn_sample, vld_sample = train_test_split(dev_sample, test_size=0.3, random_state=42)
oot_sample = model_data.query(f"apply_month >= '2024-05' and mob_i>=3 and {target}==[0, 1]").reset_index(drop=True)

print("DEV SAMPLE SIZE: %d, BAD: %d, BAD RATIO: %.4f" %
      (dev_sample.shape[0], dev_sample[target].sum(), dev_sample[target].mean()))
print("OOT SAMPLE SIZE: %d, BAD: %d, BAD RATIO: %.4f" %
      (oot_sample.shape[0], oot_sample[target].sum(), oot_sample[target].mean()))


def calculate_importance(model, data, input_variables, importance_type='shap'):
    if importance_type=='shap':
        if type(model) == XGBClassifier:
            shap_values = model.get_booster().predict(xgb.DMatrix(data[input_variables]), pred_contribs=True)
        elif type(model) == LGBMClassifier:
            shap_values = model.predict(data[input_variables], pred_contrib=True)
        shap_df = pd.DataFrame(np.abs(shap_values[:,:-1]), columns=input_variables)

        shap_imp = shap_df.mean().sort_values(ascending=False).reset_index()
    else:
        shap_imp = pd.DataFrame([condicate_base_variables, gbm_model.feature_importances_]).T
    shap_imp.columns = ['Feature', 'Importance']
    shap_imp = shap_imp[shap_imp['Importance']>0]
    shap_imp['type'] = importance_type
    return shap_imp.sort_values('Importance', ascending=False)


from sklearn.model_selection import StratifiedKFold, KFold

target = 'is_stage6_due10'
save_model="../model/new_bcard_lgbm_mobgt3"
n_folds=5
seed=42
use_stratified=True
variables = input_variables
oof_predict = np.zeros(dev_sample.shape[0])
oot_predict = np.zeros(oot_sample.shape[0])
evalute_predict = np.zeros(model_data.shape[0])
feature_importance_df = pd.DataFrame()
score_list = []
kf = StratifiedKFold(
    n_splits=n_folds,
    shuffle=True,
    random_state=seed) if use_stratified else KFold(
    n_splits=n_folds,
    shuffle=True,
    random_state=seed)

for n_fold, (trn_idx, vld_idx) in enumerate(
        kf.split(dev_sample[variables], dev_sample[target])):
    x_trn, x_vld = dev_sample[variables].iloc[trn_idx,:], dev_sample[variables].iloc[vld_idx, :]
    y_trn, y_vld = dev_sample[target].values[trn_idx], dev_sample[target][vld_idx]
    clf = LGBMClassifier(**{'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.95, 'importance_type': 'gain', 'learning_rate': 0.05, 'max_depth': 3, 'min_child_samples': 200, 
                              'min_child_weight': 5, 'min_split_gain': 0.0, 'n_estimators': 200, 'n_jobs': -1, 'num_leaves': 30, 'objective': None, 'random_state': None, 'reg_alpha': 5, 'reg_lambda': 2, 
                              'silent': 'warn', 'subsample': 0.95, 'subsample_for_bin': 1000, 'subsample_freq': 0, 'sample_weight': None, 'seed': 42}
    )
    clf.fit(
        x_trn,
        y_trn,
        eval_set=[
            (x_trn,
             y_trn),
            (x_vld,
             y_vld)],
        eval_metric="auc",
        early_stopping_rounds=20,
        verbose=False)
    
    # shap importance
    shap_imp = calculate_importance(clf, oot_sample, variables, importance_type='shap')
    
    sel_cols = shap_imp['Feature'].tolist()[:30]
    
    clf1 = LGBMClassifier(**{'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.95, 'importance_type': 'gain', 'learning_rate': 0.05, 'max_depth': 4, 'min_child_samples': 200, 
                              'min_child_weight': 5, 'min_split_gain': 0.0, 'n_estimators': 300, 'n_jobs': -1, 'num_leaves': 30, 'objective': None, 'random_state': None, 'reg_alpha': 5, 'reg_lambda': 2, 
                              'silent': 'warn', 'subsample': 0.95, 'subsample_for_bin': 1000, 'subsample_freq': 0, 'sample_weight': None, 'seed': 42}
    )
    
    clf1.fit(
        x_trn[sel_cols],
        y_trn,
        eval_set=[
            (x_trn[sel_cols],
             y_trn),
            (x_vld[sel_cols],
             y_vld)],
        eval_metric="auc",
        early_stopping_rounds=20,
        verbose=False)
    
    if save_model is not None:
        # pickle.dump(clf1, open(f'{save_model}_{n_fold}.pkl', 'wb'))
        clf1.booster_.save_model(f'{save_model}_{n_fold}.txt')
    vld_predict = clf1.predict_proba(x_vld[sel_cols])[:, 1]  # , num_iteration=clf.best_iteration_
    oof_predict[vld_idx] = vld_predict
    vld_score = roc_auc_score(y_vld, vld_predict)
    vld_ks = ks(y_vld, vld_predict)
    oot_predict += clf1.predict_proba(oot_sample[sel_cols])[:, 1] / kf.n_splits
    evalute_predict += clf1.predict_proba(model_data[sel_cols])[:, 1] / kf.n_splits

    shap_imp = calculate_importance(clf1, dev_sample, sel_cols, importance_type='shap')
    feature_importance_df = pd.concat([feature_importance_df, shap_imp], axis=0)
    
    print(
        '- Fold %d AUC : %.4f, KS : %.4f' %
        (n_fold + 1, vld_score, vld_ks))
    score_list.append(vld_score)
oot_score = roc_auc_score(oot_sample[target], oot_predict)
oot_ks = ks(oot_sample[target], oot_predict)
print('- OOT AUC : %.4f, KS : %.4f' % (oot_score, oot_ks))
feature_importance = feature_importance_df[["Feature", "Importance"]].groupby(
    "Feature").mean().sort_values(by="Importance", ascending=False).reset_index()
feature_importance.columns = ['Feature', 'Importance']




model_data_1 = pd.read_csv("../data/贷中回溯行为变量6期10D变量明细.csv")
model_data_1 = model_data_1.query("(current_due_amt<=0 or current_due_amt!=current_due_amt) and (his_maxdueday<15 or his_maxdueday!=his_maxdueday)")

import lightgbm as lgbm
oot_sample['predict_proba'] = 0
model_data_1['predict_proba'] = 0
for i in range(5):
    model_name = f"../model/new_bcard_lgbm_mobgt3_{i}.txt"
    model_txt = lgbm.Booster(model_file=model_name)
    xx = model_txt.feature_name()
    oot_sample['predict_proba'] += model_txt.predict(oot_sample[xx])
    model_data_1['predict_proba'] += model_txt.predict(model_data_1[xx])

oot_sample['predict_proba'] = oot_sample['predict_proba']/5
model_data_1['predict_proba'] = model_data_1['predict_proba']/5

for i in range(5):
    model_name = f"../model/bcard_lgbm_mobgt1_{i}.txt"
    model_txt = lgbm.Booster(model_file=model_name)
    xx = model_txt.feature_name()
    oot_sample['predict_proba'] += model_txt.predict(oot_sample[xx])
    model_data_1['predict_proba'] += model_txt.predict(model_data_1[xx])

oot_sample['predict_proba'] = oot_sample['predict_proba']/10
model_data_1['predict_proba'] = model_data_1['predict_proba']/10

oot_score = roc_auc_score(oot_sample[target], oot_sample['predict_proba'])
oot_ks = ks(oot_sample[target], oot_sample['predict_proba'])
print('- OOT AUC : %.4f, KS : %.4f' % (oot_score, oot_ks))

model_data_1['bcard_score_v1_0'] = model_data_1['predict_proba'].apply(score_card)

target = 'is_stage6_due10'
model_helper.model_group_monitor(
    model_data_1.query(f"tx_date>='2024-05-01' and {target}==[0, 1]"),
    target,
    "bcard_score_v1_0",
    higher_better=True,
    number_of_groups=10,
)

target = 'is_stage6_due10'
model_helper.model_group_monitor(
    model_data_1.query(f"tx_date>='2024-05-01' and mob_i>=1 and {target}==[0, 1]"),
    target,
    "bcard_score_v1_0",
    higher_better=True,
    number_of_groups=20,
)

model_helper.model_group_monitor(
    model_data_1.query(f"tx_date>='2024-05-01' and mob_i>=2 and {target}==[0, 1]"),
    target,
    "bcard_score_v1_0",
    higher_better=True,
    number_of_groups=10,
)

model_helper.model_group_monitor(
    model_data_1.query(f"tx_date>='2024-05-01' and mob_i>=3 and {target}==[0, 1]"),
    target,
    "bcard_score_v1_0",
    higher_better=True,
    number_of_groups=20,
)

model_data_1[['loan_no', 'predict_proba', 'bcard_score_v1_0']].to_csv("../score/bcard_mobgt3_score_v1.csv", index=False)

import lightgbm as lgbm
oot_sample['predict_proba'] = 0
for i in range(5):
    model_name = f"../model/bcard_lgbm_month__{i}.txt"
    model_txt = lgbm.Booster(model_file=model_name)
    xx = model_txt.feature_name()
    oot_sample['predict_proba'] += model_txt.predict(oot_sample[xx])

oot_sample['predict_proba'] = oot_sample['predict_proba']/5

oot_score = roc_auc_score(oot_sample[target], oot_sample['predict_proba'])
oot_ks = ks(oot_sample[target], oot_sample['predict_proba'])
print('- OOT AUC : %.4f, KS : %.4f' % (oot_score, oot_ks))


model_helper.model_group_monitor(
    oot_sample,
    target,
    "predict_proba",
    higher_better=False,
    number_of_groups=10,
)

model_helper.model_group_monitor(
    oot_sample.query("mob_i>=3"),
    target,
    "predict_proba",
    higher_better=False,
    number_of_groups=10,
)


shap_imp_1 = calculate_importance(gbm_model, oot_sample, sel_cols, importance_type='shap')
shap_imp_1.head(10)
sel_cols = shap_imp_1['Feature'].tolist()[:50]


gbm_model = LGBMClassifier(**{'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.95, 'importance_type': 'gain', 'learning_rate': 0.05, 'max_depth': 4, 'min_child_samples': 220, 
                              'min_child_weight': 5, 'min_split_gain': 0.0, 'n_estimators': 250, 'n_jobs': -1, 'num_leaves': 30, 'objective': None, 'random_state': None, 'reg_alpha': 5, 'reg_lambda': 2, 
                              'silent': 'warn', 'subsample': 0.95, 'subsample_for_bin': 1000, 'subsample_freq': 0, 'sample_weight': None, 'seed': 42}
)
gbm_model.fit(dev_sample[sel_cols], dev_sample[target], eval_set=[(dev_sample[sel_cols], dev_sample[target]), (oot_sample[sel_cols], oot_sample[target])],
             early_stopping_rounds=20, verbose=20, eval_metric='auc')

model_data['bcard_xgb_proba'] = gbm_model.predict_proba(model_data[sel_cols])[:, 1]
model_data['bcard_score_v1_0'] = model_data['bcard_xgb_proba'].apply(score_card)

target = 'is_stage6_due10'
model_helper.model_group_monitor(
    model_data.query(f"tx_date>='2024-05-01' and {target}==[0, 1]"),
    target,
    "bcard_score_v1_0",
    higher_better=True,
    number_of_groups=10,
)

model_helper.model_group_monitor(
    model_data.query(f"tx_date>='2024-05-01' and mob_i>=3 and {target}==[0, 1]"),
    target,
    "bcard_score_v1_0",
    higher_better=True,
    number_of_groups=10,
)


                                  Feature  Importance  type
0         last120_advancerepayed2days_rat    0.039848  shap
1                      credit_channel_cnt    0.038425  shap
2               last60_advancerepayed_cnt    0.031802  shap
3                    last_30day_maxdueday    0.028942  shap
4       closeddistance_perd2_tx_maxdueday    0.021561  shap
5                         credit_used_rat    0.020574  shap
6                  last30_due0_perdno_rat    0.020093  shap
7             future_120days_repaying_amt    0.015841  shap
8  last120_advancerepayed2days_loanno_rat    0.015790  shap
9             avg_credit_lastmth2_use_rat    0.015207  shap

                             Feature  Importance  type
0                    credit_used_rat    0.093441  shap
1                 credit_channel_cnt    0.082419  shap
2               last_30day_maxdueday    0.056520  shap
3             last90_due0_perdno_rat    0.020029  shap
4        avg_credit_lastmth1_use_rat    0.015686  shap
5             is_wholesale_fund_cust    0.014408  shap
6         last120_advancerepayed_cnt    0.013889  shap
7                        settled_amt    0.013471  shap
8  closeddistance_perd2_tx_maxdueday    0.013118  shap
9                        settled_cnt    0.012553  shap