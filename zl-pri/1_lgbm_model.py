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
from model_tools.metrics import roc_auc_score, ks
from model_tools.ScoreCard import model_helper
from model_tools.Model import model_utils
from model_tools.Evalutor import third_evalutor
from model_tools.AutoModel import AutoXGBoost

import warnings
warnings.filterwarnings("ignore")

model_data = pd.read_csv("../data/银联智测已接入字段宽表_16w.csv").iloc[:, 1:]
condicate_variables = [x for x in model_data.columns if x.startswith("UP")]
model_data['apply_month'] = model_data['loan_date'].apply(lambda x: x[:7])
model_data.head()

target = 'cob3_30'

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
model_data.query(f"{target}==[0, 1, -1]").groupby(['apply_month']).apply(lambda x: f_mi_1(x, None, target))

data_report = model_helper.data_report(model_data[condicate_variables])
data_report.to_csv("../doc/data_report.csv", index=False)
data_report = pd.read_csv("../doc/data_report.csv")

input_variables = data_report.query("dtype!='object' and missing_rate<=0.8 and unique>1")['column'].tolist()
# model_data[input_variables] = model_data[input_variables].replace([-9999, -99, -999, -88888, -99999, -9999999, -999999, -1111.0, -2, -1], np.nan)

import joblib
import math

def score_card(prob):
    base_score = 600
    base_odds = 1/50
    PDO = 50
    B = PDO*1.0/math.log(2)
    A = base_score + B*math.log(base_odds)
    return round(A - B*math.log( prob/ (1-prob+1e-20)),2)


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


target = 'cob3_30'
dev_sample = model_data.query(f"loan_date >= '2024-03-01' and loan_date < '2024-08-01' and {target}==[0, 1]").reset_index(drop=True)
oot_sample = model_data.query(f"loan_date >= '2024-08-01' and {target}==[0, 1]").reset_index(drop=True)

print("DEV SAMPLE SIZE: %d, BAD: %d, BAD RATIO: %.4f" %
      (dev_sample.shape[0], dev_sample[target].sum(), dev_sample[target].mean()))
print("OOT SAMPLE SIZE: %d, BAD: %d, BAD RATIO: %.4f" %
      (oot_sample.shape[0], oot_sample[target].sum(), oot_sample[target].mean()))


gbm_model = LGBMClassifier(**{'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.95, 'importance_type': 'gain', 'learning_rate': 0.05, 'max_depth': 4, 'min_child_samples': 250, 'min_child_weight': 5, 'min_split_gain': 0.0, 'n_estimators': 300, 'n_jobs': -1, 'num_leaves': 30, 'objective': None, 'random_state': None, 'reg_alpha': 5, 'reg_lambda': 2, 'silent': 'warn', 'subsample': 0.95, 'subsample_for_bin': 1000, 'subsample_freq': 0, 'sample_weight': None, 'seed': 42}
)

gbm_model.fit(dev_sample[input_variables], dev_sample[target], 
              eval_set=[(dev_sample[input_variables], dev_sample[target]), (oot_sample[input_variables], oot_sample[target])],
             early_stopping_rounds=20, verbose=20, eval_metric='auc')

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

shap_imp_1 = calculate_importance(gbm_model, oot_sample, input_variables, importance_type='shap')

sel_cols = shap_imp_1['Feature'].tolist()[:140]

def forward_corr_delete(df, col_list, cut_value=0.7):
    """
    相关性筛选，亮点是当某个变量因为相关性高，需要进行删除时，此变量不再和后续变量进行相关性计算，否则会把后续不应删除的变量由于和已经删除变量相关性高，而进行删除
    param:
        df -- 数据集 Dataframe
        col_list -- 需要筛选的特征集合,需要提前按IV值从大到小排序好 list
        corr_value -- 相关性阈值，高于此阈值的比那里按照iv高低进行删除，默认0.7
    return:
        select_corr_col -- 筛选后的特征集合 list
    """
    corr_list = []
    corr_list.append(col_list[0])
    delete_col = []
    # 根据IV值的大小进行遍历
    for col in col_list[1:]:
        corr_list.append(col)
        #当多个变量存在相关性时，如果前述某个变量已经删除，则不应再和别的变量计算相关性
        if len(delete_col) > 0:  # 判断是否有需要删除的变量
            for i in delete_col:
                if i in corr_list:  # 由于delete_col是一直累加的，已经删除的变量也会出现在delete_col里，因此需要判断变量当前是否还在corr_list里，否则remove会报错
                    corr_list.remove(i)
#         print(delete_col)
#         print(corr_list)
#         print('---------')
        corr = df.loc[:, corr_list].corr()
        corr_tup = [(x, y) for x, y in zip(corr[col].index, corr[col].values)]
        corr_value = [y for x, y in corr_tup if x != col]
        # 若出现相关系数大于0.65，则将该特征剔除
        if len([x for x in corr_value if abs(x) >= cut_value]) > 0:
            delete_col.append(col)
#             print(delete_col)
    select_corr_col = [x for x in col_list if x not in delete_col]
    return select_corr_col, delete_col

select_corr_col, delete_col = forward_corr_delete(oot_sample.sample(50000), xgb_condicate_input, cut_value=0.6)

select_corr_col = input_variables

gbm_model = LGBMClassifier(**{'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.95, 'importance_type': 'gain', 'learning_rate': 0.05, 'max_depth': 4, 'min_child_samples': 220, 
                              'min_child_weight': 5, 'min_split_gain': 0.0, 'n_estimators': 250, 'n_jobs': -1, 'num_leaves': 30, 'objective': None, 'random_state': None, 'reg_alpha': 5, 'reg_lambda': 2, 
                              'silent': 'warn', 'subsample': 0.95, 'subsample_for_bin': 1000, 'subsample_freq': 0, 'sample_weight': None, 'seed': 42}
)
gbm_model.fit(dev_sample[select_corr_col], dev_sample[target], eval_set=[(dev_sample[select_corr_col], dev_sample[target]), (oot_sample[select_corr_col], oot_sample[target])],
             early_stopping_rounds=20, verbose=20, eval_metric='auc')

shap_imp_2 = calculate_importance(gbm_model, oot_sample, select_corr_col, importance_type='shap')


from sklearn.model_selection import StratifiedKFold, KFold

save_model="../model/yinlian_submodel_v2"
n_folds=5
seed=2025
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
    
    sel_cols = shap_imp['Feature'].tolist()[:50]
    
    clf1 = LGBMClassifier(**{'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.95, 'importance_type': 'gain', 'learning_rate': 0.05, 'max_depth': 4, 'min_child_samples': 200, 
                              'min_child_weight': 5, 'min_split_gain': 0.0, 'n_estimators': 120, 'n_jobs': -1, 'num_leaves': 30, 'objective': None, 'random_state': None, 'reg_alpha': 5, 'reg_lambda': 2, 
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

feature_importance.to_csv("../doc/fold_importance.csv", index=False, encoding='gbk')


import lightgbm as lgbm
model_data['predict_proba'] = 0
oot_sample['predict_proba'] = 0
for i in range(n_folds):
    model_name = f"../model/yinlian_submodel_v1_{i}.txt"
    model_txt = lgbm.Booster(model_file=model_name)
    xx = model_txt.feature_name()
    oot_sample['predict_proba'] += model_txt.predict(oot_sample[xx].astype(float))/n_folds
    model_data['predict_proba'] += model_txt.predict(model_data[xx].astype(float))/n_folds

oot_score = roc_auc_score(oot_sample[target], oot_sample['predict_proba'])
oot_ks = ks(oot_sample[target], oot_sample['predict_proba'])
print('- OOT AUC : %.4f, KS : %.4f' % (oot_score, oot_ks))

dev_sample['predict_proba'] = oof_predict
dev_sample['yinlian_submodel_score'] = dev_sample['predict_proba'].apply(score_card)
oot_sample['yinlian_submodel_score'] = oot_sample['predict_proba'].apply(score_card)

model_data['yinlian_submodel_score'] = model_data['predict_proba'].apply(score_card)



model_data[['id_card_no', 'loan_date', 'predict_proba', 'yinlian_submodel_score']].to_csv("../score/ot_v1_2_lgbm_score_batch2_20250105.csv", index=False)
data_1 = pd.concat([dev_sample[['id_card_no', 'loan_date', 'apply_month', 'channel_code', 'predict_proba', 'yinlian_submodel_score', 'cob3_30']], 
                   oot_sample[['id_card_no', 'loan_date', 'apply_month', 'channel_code', 'predict_proba', 'yinlian_submodel_score', 'cob3_30']]])
# dev_sample[['id_card_no', 'loan_date', 'channel_code', 'predict_proba', 'ot_v1_2_gbm_score']].to_csv("../score/ot_v1_2_lgbm_model_score_20250105.csv", index=False)

data_1 = pd.concat([dev_sample[['id_card_no', 'loan_date', 'apply_month', 'channel_code', 'predict_proba', 'yinlian_submodel_score', 'cob3_30']], 
                   oot_sample[['id_card_no', 'loan_date', 'apply_month', 'channel_code', 'predict_proba', 'yinlian_submodel_score', 'cob3_30']]])
score = "yinlian_submodel_score"
data_1['cnt'] = 1
data_1.query(f"{target}==[0, 1] and apply_month>='2024-03'").groupby(['apply_month']).apply(lambda x: f_evalutor_3(x, score, target, "score")).reset_index()

data_1.query(f"{target}==[0, 1] and apply_month>='2024-03' and channel_code=='RS'").groupby(['cnt']).apply(lambda x: f_evalutor_3(x, score, target, "score")).reset_index()
data_1.query(f"{target}==[0, 1] and apply_month>='2024-03' and channel_code=='APPZY'").groupby(['cnt']).apply(lambda x: f_evalutor_3(x, score, target, "score")).reset_index()
data_1.query(f"{target}==[0, 1] and apply_month>='2024-03' and channel_code=='R360'").groupby(['cnt']).apply(lambda x: f_evalutor_3(x, score, target, "score")).reset_index()
data_1.query(f"{target}==[0, 1] and apply_month>='2024-03' and channel_code=='ICE_ZLSK_36'").groupby(['cnt']).apply(lambda x: f_evalutor_3(x, score, target, "score")).reset_index()
data_1.query(f"{target}==[0, 1] and apply_month>='2024-03' and channel_code=='QXL'").groupby(['cnt']).apply(lambda x: f_evalutor_3(x, score, target, "score")).reset_index()
data_1.query(f"{target}==[0, 1] and apply_month>='2024-03'").query('channel_code.str.contains("TYH")').groupby(['cnt']).apply(lambda x: f_evalutor_3(x, score, target, "score")).reset_index()

data_1['channel_code'].value_counts()


data_1.query(f"{target}==[0, 1] and apply_month>='2024-03'").groupby(['cnt']).apply(lambda x: f_evalutor_3(x, score, target, "score")).reset_index()
