# -*- coding:utf-8 -*-
__author__ = 'fenghaijie'

import time
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_validate
from ks_evaluation import ks_compute

"""
模块描述：模型拟合(Model Fittiing)
功能包括：
1) train_model: 模型训练
2) test_model:  模型预测
3) feat_importance: 根据训练完毕的模型得到特征重要性
"""


def train_model(model, X_train, y_train, X_valid=None, y_valid=None, W_train=None):
    '''
    ----------------------------------------------------------------------
    功能：训练机器学习模型。支持加入权重列。
    ----------------------------------------------------------------------
    :param model:   待训练的机器学习模型, 如xgboost或rf
    :param X_train: pd.DataFrame, 训练集特征
    :param y_train: pd.DataFrame, 训练集标签
    :param X_valid: pd.DataFrame, 验证集特征
    :param y_valid: pd.DataFrame, 验证集标签
    :param W_train: pd.DataFrame, 训练权重列, 默认值=None, 代表不加权重列
    ----------------------------------------------------------------------
    :return trained_model: 训练完成的模型
    ----------------------------------------------------------------------
    示例1：
    from sklearn.ensemble import RandomForestClassifier
    >>> rf = RandomForestClassifier(
         n_estimators=200
        ,max_features='sqrt'
        ,max_depth=3
        ,min_samples_leaf=400
    )
    >>> trained_rf = train_model(model=rf, 
                         X_train=develop_data_ins[feats].fillna(999999), 
                         y_train=develop_data_ins[target_var], 
                         X_valid=develop_data_oos[feats].fillna(999999), 
                         y_valid=develop_data_oos[target_var])
    >>>
    模型开始训练中...
    Running time: 0.0003788 Minutes
    输出模型训练结果...
    KS Value  (INS): 0.5412
    Accuracy  (INS): 0.7722
    AUC Score (INS): 0.854296
    ----------------------------------------------------------------------
    示例2：
    import xgboost as xgb
    xgbm = xgb.XGBClassifier(
    max_depth=2,         # 含义：树的深度，默认值为6，典型值3-10。调参：值越大，越容易过拟合；值越小，越容易欠拟合。
    learning_rate=0.1,   # 含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
    n_estimators=70,     # 含义：总共迭代的次数，即决策树的个数
    silent=True, 
    objective='binary:logistic',
    booster='gbtree',    # gbtree 树模型做为基分类器（默认）, gbliner 线性模型做为基分类器
    n_jobs=-1,           # 多线程
    nthread=None,
    gamma=0,             # 惩罚项系数，指定节点分裂所需的最小损失函数下降值。
    min_child_weight=3,  # 含义：默认值为1.调参：值越大，越容易欠拟合；值越小，越容易过拟合（值较大时，避免模型学习到局部的特殊样本）
    max_delta_step=0, 
    subsample=0.9,       # 含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
    colsample_bytree=1, 
    colsample_bylevel=1, 
    reg_alpha=1,         # L1正则化系数，默认为1
    reg_lambda=3,        # L2正则化系数，默认为1
    scale_pos_weight=8,  # 正样本的权重，在二分类任务中，当正负样本比例失衡时，设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时，
                         # scale_pos_weight=10。
    base_score=0.5, 
    random_state=0,
    seed=None,
    missing=None)
    ----------------------------------------------------------------------
    '''
    print('模型开始训练中...')
    s_m = time.clock()
    if W_train is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, W_train)
    e_m = time.clock()
    print('Running time: %.4g Minutes' % ((e_m - s_m) * 1.0 / 60))
    
    print('输出模型训练结果...')
    y_ins_pred = model.predict(X_train)
    y_ins_predprob = model.predict_proba(X_train)[:,1]
    print("KS Value  (INS): %.4g" % ks_compute(proba_arr=y_ins_predprob, target_arr=y_train))
    print("Accuracy  (INS): %.4g" % metrics.accuracy_score(y_train, y_ins_pred))
    print("AUC Score (INS): %f" % metrics.roc_auc_score(y_train, y_ins_predprob))
    
    print('-' * 20)
    if X_valid is None or y_valid is None:
        pass
    else:
        y_oos_pred = model.predict(X_valid)
        y_oos_predprob = model.predict_proba(X_valid)[:,1]
        print("KS Value  (INS): %.4g" % ks_compute(proba_arr=y_oos_predprob, target_arr=y_valid))
        print("Accuracy  (OOS): %.4g" % metrics.accuracy_score(y_valid, y_oos_pred))
        print("AUC Score (OOS): %f" % metrics.roc_auc_score(y_valid, y_oos_predprob))
        print('训练完成')
    
    return model


def test_model(model, X_test, y_test):
    '''
    ----------------------------------------------------------------------
    功能：利用机器学习模型进行预测
    ----------------------------------------------------------------------
    :param model:  训练完成的机器学习模型
    :param X_test: pd.DataFrame, 测试集特征
    :param y_test: pd.DataFrame, 测试集标签
    ----------------------------------------------------------------------
    :return pred: list, 预测分类, [0, 1,...] 
    :return prob: list, 预测概率, [0.2, 0.8,...]
    ----------------------------------------------------------------------
    示例：
    >>> pred, prob = test_model(trained_model, df[c_cols].fillna(999999), df[target])
    >>>
    开始模型预测...
    KS Value  (OOT): 0.5412
    Accuracy  (OOT): 0.7722
    AUC Score (OOT): 0.854296
    ----------------------------------------------------------------------
    '''
    print('开始模型预测...')
    y_oot_pred = model.predict(X_test)
    y_oot_predprob = model.predict_proba(X_test)[:,1]
    print("KS Value  (OOT): %.4g" % ks_compute(proba_arr=y_oot_predprob, target_arr=y_test))
    print("Accuracy  (OOT): %.4g" % metrics.accuracy_score(y_test, y_oot_pred))
    print("AUC Score (OOT): %f" % metrics.roc_auc_score(y_test, y_oot_predprob))
    print('-' * 20)
    
    return y_oot_pred, y_oot_predprob


def feat_importance(model, feat_list, accumulate_score=0.95):
    '''
    ----------------------------------------------------------------------
    功能：得到特征重要度
    ----------------------------------------------------------------------
    :param model: 训练完成的机器学习模型
    :param feat_list: list，特征列表，如['x1', 'x2']
    :param accumulate_score: float，累积重要度（所有特征之和=1）, 默认为0.95
    ----------------------------------------------------------------------
    return output_df: pd.DataFrame, 包括特征，特征重要度，并按重要度降序输出
    ----------------------------------------------------------------------
    示例：
    >>> feat_importance(model=trained_model, feat_list=c_cols, accumulate_score=0.95)
    >>>
    累积score达到0.95时的特征序号为4
    	var	score	score_rank	topk
    0	Age	0.436644	1	
    1	Fare	0.416096	2	
    2	SibSp	0.090753	3	
    3	Pclass	0.056507	4	<<<
    ----------------------------------------------------------------------
    '''
    if accumulate_score > 1 or accumulate_score < 0:
        raise Exception('参数accumulate_score取值范围应在[0,1]，请检查!')
        
    output_df = pd.DataFrame()
    output_df.loc[:, 'var'] = feat_list
    output_df.loc[:, 'score'] = model.feature_importances_
    output_df = output_df.sort_values(by=['score'], ascending=0).reset_index(drop=1)
    score_lst = list(output_df['score'].values)
    
    acc_score = 0
    idx = 0
    while True:
        acc_score += score_lst[idx]
        if acc_score >= accumulate_score:
            break
        else:
            idx = idx + 1
    idx = idx + 1
    print('累积score达到%s时的特征序号为%s' % (str(accumulate_score), str(idx)))
    
    output_df.loc[:, 'score_rank'] = range(1, len(output_df)+1)
    output_df.loc[:, 'topk'] = output_df['score_rank'].apply(lambda x: '<<<' if x == idx else ' ')
    #output_df = output_df.drop(['score_rank'], axis=1)
    
    return output_df