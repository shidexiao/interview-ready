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


model_data = pd.read_csv("../data/贷中行为变量月初口径.csv")
model_data = model_data.query("(current_due_amt<=0 or current_due_amt!=current_due_amt) and (his_maxdueday<15 or his_maxdueday!=his_maxdueday)")
print(model_data.shape)
model_data['his_maxdueday'].describe()

model_data.head()
model_data['id_no'].value_counts()

model_data.columns[:30]

view_cols = [
    'tx_date', 'tx_time', 'channel_key', 'loan_no', 'id_no', 'dn_seq',
    'apply_amt', 'distr_amt', 'credit_amt', 'first_loan_date', 'mob_i',
    'is_loan', 'is_risk_pass', 'is_settled', 'is_wholesale_fund_cust', 'tx_mth', 'is_stage6_due10_new'
]

model_data.query("id_no=='371102198703258120'")[view_cols]


model_data['mob_i'].value_counts()
model_data = model_data.query("mob_i>=0").reset_index(drop=True)
model_data.shape

condicate_variables = [i for i in model_data.columns if model_data[i].dtypes !='object' and i.find('is_stage')<0 and i not in ['apply_amt','distr_amt','is_risk_pass','is_loan']]
condicate_variables = [x for x in condicate_variables if x not in ['credit_amt','mob_i', 'asc_row']]

model_data['apply_month'] = model_data['tx_mth']
model_data = model_data.query("mob_i >= 3").reset_index(drop=True)

target = 'is_stage6_due10_new'

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
# data_report.to_csv("../doc/data_report_mob_bt3_monthly.csv", index=False)
data_report = pd.read_csv("../doc/data_report_mob_bt3_monthly.csv")

input_variables = data_report.query("dtype!='object' and missing_rate<=0.8 and unique>1")['column'].tolist()

mono_woe = third_evalutor.woe_bin(model_data.query(f"mob_i >= 3 and {target}==[0,1]")[input_variables+[target]].reset_index(drop=True), target, min_group_rate=0.05, max_bin=6, bin_method='mono', alg_method='iv')        
mapiv = mono_woe.split_data()
mapiv['%bad'] = mapiv['p1']/(mapiv['p1']+mapiv['p0'])
mapiv.to_csv("../doc/mono_mapiv_mob_bt3_monthly.csv", index=False)
    
mapiv['iv'].describe()

candidate_variables = [x for x in mapiv.query("iv>=0.02")['varname'].drop_duplicates().tolist()]
len(candidate_variables)

result = """"""
result += "def process(model_data):\n"
result += "  import numpy as np\n"
for col in candidate_variables:
    bins_info_temp = mapiv.query(f"varname=='{col}'")
    have_null = 1-int(bins_info_temp['bin'].tolist()[0])
    for index, left, right, woe in zip(bins_info_temp['bin'].tolist(), 
                                bins_info_temp['ll'].tolist(),
                                bins_info_temp['ul'].tolist(),
                                bins_info_temp['woe'].tolist()):
#         print(index, left, right, woe)
        if have_null:
            if str(left) == "nan" and str(right) == "nan":
                result += f"  model_data['W_{col}'] = \\\n    np.where(model_data['{col}'].isnull(),     {woe},\n"
            elif str(right) == "inf":
                result += f"  {woe}" + ")"*int(index) + "\n\n"
            else:
                result += f"    np.where(model_data['{col}'] < {right},        {woe},\n"
        else:
            if int(index)==1:
                result += f"  model_data['W_{col}'] = \\\n    np.where(model_data['{col}'] <{right},     {woe},\n"
            elif str(right) == "inf":
                result += f"  {woe}" + ")"*(int(index)-1) + "\n\n"
            else:
                result += f"    np.where(model_data['{col}'] < {right},        {woe},\n"
            
result += "  return model_data"

print(result, file=open("../code/calculator_mob_bt3_monthly.py", "w+"))

import calculator_mob_bt3_monthly
from imp import reload
reload(calculator_mob_bt3_monthly)

model_data = calculator_mob_bt3_monthly.process(model_data)

woe_condicate_variables = ["W_"+x for x in mapiv.query("iv>=0.02")['varname'].drop_duplicates().tolist()]

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


select_corr_col, delete_col = forward_corr_delete(model_data, [x[2:] for x in woe_condicate_variables], cut_value=0.8)

import varclushi

varclus_input = ['W_'+x for x in select_corr_col]
varclus_proc = varclushi.VarClusHi(model_data[varclus_input], maxeigval2=1, maxclus=None)
varclus_proc.varclus()

varclus_info = varclus_proc.info
varclus_result = varclus_proc.rsquare

mapiv['Variable'] = mapiv['varname'].apply(lambda x: 'W_'+x)
varclus_iv = varclus_result.merge(mapiv[['Variable', 'iv']].drop_duplicates(), on='Variable', how='left')
varclus_iv['rank'] = varclus_iv.groupby('Cluster')['iv'].rank(ascending=False)

target = 'is_stage6_due10'
dev_woe = model_data.query(f"apply_month <= '2024-04' and {target}==[0, 1]").reset_index(drop=True)
oot_woe = model_data.query(f"apply_month >= '2024-05' and {target}==[0, 1]").reset_index(drop=True)

from model_tools.ScoreCard import modeler

sw_input = varclus_input
temp = dev_woe
sw = modeler.StepwiseModel(temp[sw_input], temp[target], method='stepwise')
sw_result = sw.stepwise()
input_cols = [x for x in sw_result.keys() if x != 'const']

input_cols = ['W_last_30day_maxdueday',
 'W_advancerepayed_cnt',
 'W_avg_credit_lastmth1_use_rat',
 'W_closeddistance_perd2_tx_maxdueday',
 'W_last60days_pass_rat',
 'W_last60_advancerepayed2days_loanno_cnt']

 model_helper.calculate_vif(dev_woe, input_cols)

 from model_tools.ScoreCard import model_helper

model_params = model_helper.logit_fit(dev_woe, target, input_cols, title = 'Fitting Dev Sample', plot_text = 'Fitting Dev Sample')
model_helper.logit_score(oot_woe, model_params, target, input_cols, 'Scoring OOT Sample', 'Scoring OOT')

dict(model_params)

import math
PDO = 50
P0 = 600
theta0 = 1/50
B = PDO / math.log(2)
A = P0 + B * math.log(theta0)

model_data['trade_model_score_v1'] = A - B * model_params['const']

for i in input_cols:
    model_data['trade_model_score_v1'] = model_data['trade_model_score_v1'] + -B * model_params[i] * model_data[i]

import seaborn as sns

sns.distplot(model_data.query("apply_month >= '2024-05'")['trade_model_score_v1'])

original_variables = [x[2:] for x in input_cols]

mapiv.query("varname==@original_variables").head(20)


mono_woe = third_evalutor.woe_bin(model_data.query(f"mob_i >= 3 and {target}==[0,1]")[input_variables+[target]].reset_index(drop=True), target, min_group_rate=0.05, max_bin=6, bin_method='mono', alg_method='iv')        


bins = sc.woebin(model_data.query(f"mob_i >= 3 and {target}==[0,1]")[input_variables+[target]].reset_index(drop=True), y=target, method="chimerge", bin_num_limit=6)
bins_df = pd.DataFrame()
for k, v in bins.items():
    bins_df = pd.concat([bins_df, v])

bins_df.sort_values(['total_iv', 'breaks'], ascending=[
    False, True], inplace=True)

bins_df.head(20)
model_data['mob_i'].value_counts()

candidate_variables = bins_df.query("total_iv>=0.02")['variable'].drop_duplicates().tolist()

result = """"""
for col in candidate_variables:
#     print(col)
    for index, breaks, woe in zip(bins[col].index.tolist(), bins[col]['breaks'].tolist(), bins[col]['woe'].tolist()):
#         print(index, bin, woe)
        if breaks == "missing":
            result += f"model_data['{col}_woe'] = \\\n    np.where(model_data['{col}'].isnull(),     {woe},\n"
        elif breaks != "missing" and index==0:
            result += f"model_data['{col}_woe'] = \\\n    np.where(model_data['{col}'] < {breaks},     {woe},\n"
        elif breaks == "inf":
            result += f"   {woe}" + ")"*index + "\n\n"
        else:
            result += f"    np.where(model_data['{col}'] < {breaks},        {woe},\n"

exec(result)
candidate_variables = bins_df.query("total_iv>=0.02")['variable'].drop_duplicates().tolist()
woe_condicate_variables = [x+'_woe' for x in candidate_variables]

select_corr_col, delete_col = forward_corr_delete(model_data, candidate_variables, cut_value=0.7)
len(select_corr_col)
varclus_input = [x+'_woe' for x in select_corr_col]


target = 'is_stage6_due10'
dev_woe = model_data.query(f"apply_month <= '2024-04' and {target}==[0, 1]").reset_index(drop=True)
oot_woe = model_data.query(f"apply_month >= '2024-05' and {target}==[0, 1]").reset_index(drop=True)

sw_input = varclus_input
temp = dev_woe
sw_input = varclus_input
sw = modeler.StepwiseModel(temp[sw_input], temp[target], method='stepwise')
sw_result = sw.stepwise()
input_cols = [x for x in sw_result.keys() if x != 'const']

input_cols_1 = ['last30_due0_perdno_rat_woe',
 'last180_advancerepayed2days_cnt_woe',
 'credit_used_rat_woe',
 'last60_due0_up500_loanno_rat_woe',
 'last120days_pass_rat_woe',
 'closeddistance_perd2_tx_maxdueday_woe'
 ]

input_cols = [
 'W_last_30day_maxdueday',
#  'W_advancerepayed_cnt',
 'W_avg_credit_lastmth1_use_rat',
 'W_closeddistance_perd2_tx_maxdueday',
 'W_last60days_pass_rat',
 'W_last60_advancerepayed2days_loanno_cnt',
#  'W_last30_due0_perdno_rat',
 'W_last180_advancerepayed2days_cnt',
 'W_credit_used_rat',
 'W_last60_due0_up500_loanno_rat',
 'W_last120days_pass_rat',
]

original_variables = [x[2:] for x in input_cols]
mapiv.query("varname==@original_variables")

model_helper.calculate_vif(dev_woe, input_cols)

 from model_tools.ScoreCard import model_helper

model_params = model_helper.logit_fit(dev_woe, target, input_cols, title = 'Fitting Dev Sample', plot_text = 'Fitting Dev Sample')

model_helper.logit_score(oot_woe, model_params, target, input_cols, 'Scoring OOT Sample', 'Scoring OOT')

import math
PDO = 40
P0 = 600
theta0 = 0.08
B = PDO / math.log(2)
A = P0 + B * math.log(theta0)

bin_score = mapiv.query("Variable==@input_cols")
bin_score['coef'] = bin_score['Variable'].map(model_params)
bin_score['partial_score'] = -B*bin_score['coef']*bin_score['woe']
init_score = A - B * model_params['const']
# 578.4483859234265

result = """"""
result += "def bcard_mobgt3_month_v1(model_data):\n"
result += "  import numpy as np\n"
for col in [x[2:] for x in input_cols]:
    bins_info_temp = bin_score.query(f"varname=='{col}'")
    have_null = 1-int(bins_info_temp['bin'].tolist()[0])
    for index, left, right, p_score in zip(bins_info_temp['bin'].tolist(), 
                                bins_info_temp['ll'].tolist(),
                                bins_info_temp['ul'].tolist(),
                                bins_info_temp['partial_score'].tolist()):
#         print(index, left, right, woe)
        if have_null:
            if str(left) == "nan" and str(right) == "nan":
                result += f"  model_data['Score_{col}'] = \\\n    np.where(model_data['{col}'].isnull(),     {p_score},\n"
            elif str(right) == "inf":
                result += f"  {p_score}" + ")"*int(index) + "\n\n"
            else:
                result += f"    np.where(model_data['{col}'] < {right},        {p_score},\n"
        else:
            if int(index)==1:
                result += f"  model_data['Score_{col}'] = \\\n    np.where(model_data['{col}'] <{right},     {p_score},\n"
            elif str(right) == "inf":
                result += f"  {p_score}" + ")"*(int(index)-1) + "\n\n"
            else:
                result += f"    np.where(model_data['{col}'] < {right},        {p_score},\n"
            
result += "  return model_data"

print(result, file=open("bcard_mobgt3_month_v1.py", "w+"))

import bcard_mobgt3_month_v1
model_data = bcard_mobgt3_month_v1.bcard_mobgt3_month_v1(model_data)
model_data['Score_Init'] = 578.4483859234265

score_variables = [x for x in model_data.columns if x.startswith("Score")]
model_data['bcard_score_v1_0'] = model_data[score_variables].apply(lambda x: round(x.sum(), 2), axis=1)

['is_stage4_expire10',
 'is_stage4_due10',
 'is_stage5_expire10',
 'is_stage5_due10',
 'is_stage6_expire10',
 'is_stage6_due10',
 'is_stage6_due10_new']

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