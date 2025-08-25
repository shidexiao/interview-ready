#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18-12-15
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import auc, roc_curve, precision_recall_curve,roc_auc_score
import math
import copy
from functools import wraps
import sys
import re


params = { 'tree_method':'gpu_hist', 'predictor':'gpu_predictor' }


def cal_auc_ks(y_true, y_predprob):
    fpr, tpr, _ = roc_curve(y_true, y_predprob)
    auc_result = round(auc(fpr, tpr), 4)
    ks_result = abs(fpr - tpr).max()
    return auc_result, ks_result


def cal_auc(y_true, y_predprob):
    fpr, tpr, _ = roc_curve(y_true, y_predprob)
    return round(auc(fpr, tpr), 4)


def roc_curve_plot(y_true, y_predprob, title):
    fpr, tpr, _ = roc_curve(y_true, y_predprob)
    auc_result = round(cal_auc(y_true, y_predprob), 4)
    plt.plot(fpr, tpr, label="ROC-AUC overall, \n AUC Score={}".format(auc_result))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot([0, 1], [0, 1])
    plt.title(title + " ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


def ks_plot(y_pred, y_true, title):
    """
    分析作图之KS
    """
    df = pd.DataFrame(y_pred[:, 1])
    df = df.rename(columns={0: 'y_pred'})
    df['y_bad'] = y_true.values
    df['y_good'] = 1 - df['y_bad']
    df_grouped = df[['y_pred', 'y_bad', 'y_good']].groupby('y_pred').agg(
        {'y_bad': np.sum, 'y_good': np.sum}).reset_index()
    df_grouped['good_cum_pct'] = (df_grouped['y_good'] / df_grouped['y_good'].sum()).cumsum()
    df_grouped['bad_cum_pct'] = (df_grouped['y_bad'] / df_grouped['y_bad'].sum()).cumsum()
    df_grouped['ks'] = (abs(df_grouped['good_cum_pct'] - df_grouped['bad_cum_pct'])) * 100
    max_ks = df_grouped.loc[df_grouped['ks'].idxmax(): df_grouped['ks'].idxmax()]
    plt.plot(df_grouped['y_pred'], df_grouped['good_cum_pct'], linewidth=2)
    plt.plot(df_grouped['y_pred'], df_grouped['bad_cum_pct'], linewidth=2)
    plt.plot([max_ks['y_pred'].iloc[0], max_ks['y_pred'].iloc[0]],
             [max_ks['good_cum_pct'].iloc[0], max_ks['bad_cum_pct'].iloc[0]], color='grey')
    plt.ylim([0, 1])
    plt.text(0.6 * np.max(df_grouped['y_pred']), np.min(df_grouped['bad_cum_pct']) + 0.46,
             title, fontsize=14)
    plt.text(0.6 * np.max(df_grouped['y_pred']), np.min(df_grouped['bad_cum_pct']) + 0.4,
             'Prob. Distribution', fontsize=14)
    plt.text(0.6 * np.max(df_grouped['y_pred']), np.min(df_grouped['bad_cum_pct']) + 0.34,
             'KS = %.2f' % max_ks['ks'].iloc[0], fontsize=14)
    plt.show()


def ks_xgb(preds, train_data):
    true = train_data.get_label()
    fpr, tpr, _ = roc_curve(true, preds, pos_label=1)
    return 'ks', abs(fpr - tpr).max()


def ks_xgb_sklearn(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    return 'ks', abs(fpr - tpr).max()


def max_recall(preds, train_data):
    true = train_data.get_label()
    precison, recall, _ = precision_recall_curve(true, preds, pos_label=1)
    pr = pd.DataFrame({'precision': precison, 'recall': recall})
    return 'Max Recall', pr.loc[pr.precision >= 0.9].loc[:, 'recall'].max()


def split_data(data, test_size=0.3,random_state=314):
    traindf, testdf = train_test_split(data, test_size=test_size, random_state=random_state)
    return traindf, testdf

def max_depth_space(feature_size):
    if feature_size > 1000:
        max_depth = range(5, 14, 2)
    else:
        max_depth = range(3, 10, 2)
    return list(max_depth)


def min_child_weight_space(train_size):
    if train_size > 10000:
        min_child_weight = [10, 50, 100, 200, 500, 2000]
    else:
        min_child_weight = [10, 50, 100, 200, 300]
    return min_child_weight


"""
下面是调参部分
"""


class XgbParamTuning(object):
    """
    xgboost手动调参
    estimator:通常是经过初步训练过的模型,基于Scikit-Learn API
    """

    def __init__(self,estimator, traindf, validdf, testdf, target, cv_folds=5,
                 early_stopping_rounds=20, booster='gbtree', silent=True,n_jobs=-1,
                 missing=None, random_state=314,gpu=False,gpu_hist=False,gpu_predict=False,
                 grow_policy='lossguide'):
        self.train = traindf[traindf.columns.difference([target])]
        self.labels_train = traindf[target]
        self.xgbtrain = xgb.DMatrix(self.train, label=self.labels_train)
        self.valid = validdf[validdf.columns.difference([target])]
        self.labels_valid = validdf[target]
        self.xgbvalid = xgb.DMatrix(self.valid, label=self.labels_valid)
        self.test = testdf[testdf.columns.difference([target])]
        self.labels_test = testdf[target]
        self.xgbtest = xgb.DMatrix(self.test, label=self.labels_test)
        self.eval_set = [(self.valid, self.labels_valid)]
        self.weight = float(len(self.labels_train[self.labels_train==0]))/len(self.labels_train[self.labels_train==1])
        self.feature_size = self.train.shape[1]
        self.train_size = self.train.shape[0]
        self.params = estimator.get_params() 
        otherparams = {}
        if gpu:
            if gpu_hist:
                otherparams['tree_method'] = 'gpu_hist'
            else:
                otherparams['tree_method'] = 'gpu_exact'
            if gpu_predict:
                otherparams['predictor'] = 'gpu_predictor'
        otherparams['grow_policy'] = grow_policy
        estimator = XGBClassifier(**otherparams)
        estimator.set_params(**self.params)
        self.estimator = estimator if estimator._Booster else estimator.fit(self.train, self.labels_train)
        self.cv_folds = cv_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.params['booster'] = booster
        self.params['silent'] = silent
        self.params['missing'] = missing if missing is not None else np.nan
        self.params['random_state'] = random_state
        self.estimator.set_params(**self.params)
        self.best_estimators = None
        self.best_max_depth = None
        self.best_min_child_weight = None
        self.best_gamma = None
        self.best_subsample = None
        self.best_colsample_bytree = None
        self.best_reg_alpha = None
        self.best_reg_lambda = None

    def init_param(self):
        self.params['learning_rate'] = 0.2  # 设一个较高的初始学习率用于调节树的个数
        self.params['max_depth'] = 5
        self.params['min_child_weight'] = 1  # 在这里选了一个比较小的值
        self.params['gamma'] = 0  # 在这里选了一个比较小的值
        self.params['subsample'] = 0.8
        self.params['colsample_bytree'] = 0.8
        self.params['scale_pos_weight'] = 1  # 考虑信贷业务样本十分不平衡。
        self.estimator.set_params(**self.params)

    def estimate(self):
        """
        默认用最新的self.estimator，看当前参数下模型的ks和auc
        """
        self.estimator.fit(self.train, self.labels_train, eval_set=self.eval_set, eval_metric=ks_xgb)
        train_prob = self.estimator.predict_proba(self.train)
        valid_prob = self.estimator.predict_proba(self.valid)
        test_prob = self.estimator.predict_proba(self.test)
        ks_plot(train_prob, self.labels_train, 'Train_KS')
        ks_plot(valid_prob, self.labels_valid, 'Valid_KS')
        ks_plot(test_prob, self.labels_test, 'Test_KS')
        roc_curve_plot(self.labels_train, train_prob[:, 1], 'Train')
        roc_curve_plot(self.labels_valid, valid_prob[:, 1], 'Valid')
        roc_curve_plot(self.labels_test, test_prob[:, 1], 'Test')

    def get_auc_ks_scikit(self, param, param_list, text=True):
        """
        用于针对某一个参数调参，计算每个参数值对应的KS和AUC，然后画图
        """
        estimator = copy.deepcopy(self.estimator)
        param_dict = {}
        train_auc = []
        train_ks = []
        test_auc = []
        test_ks = []
        params = []
        for i in param_list:
            param_dict[param] = i
            estimator.set_params(**param_dict)
            estimator.fit(self.train, self.labels_train)
            train_prob = estimator.predict_proba(self.train)
            test_prob = estimator.predict_proba(self.valid)
            train_auc_value, train_ks_value = cal_auc_ks(self.labels_train, train_prob[:, 1])
            test_auc_value, test_ks_value = cal_auc_ks(self.labels_valid, test_prob[:, 1])
            params.append(i)
            train_auc.append(train_auc_value)
            train_ks.append(train_ks_value)
            test_auc.append(test_auc_value)
            test_ks.append(test_ks_value)
        x = params
        _, axs = plt.subplots(2, 1, figsize=(15, 15))
        axs[0].plot(x, train_auc, label='train_auc')
        axs[0].plot(x, test_auc, label='valid_auc')
        axs[0].legend(loc=2)
        axs[1].plot(x, train_ks, label='train_ks')
        axs[1].plot(x, test_ks, label='valid_ks')
        axs[1].legend(loc=2)
        axs[1].set_xlabel(param)
        if text:  # 通常X数量比较小的时候
            axs[0].set_xticks(x)
            axs[1].set_xticks(x)
            for a, b in zip(x, train_auc):
                axs[0].text(a, b, str(round(b, 4)))
            for a, c in zip(x, test_auc):
                axs[0].text(a, c, str(round(c, 4)))
            for a, d in zip(x, train_ks):
                axs[1].text(a, d, str(round(d, 4)))
            for a, e in zip(x, test_ks):
                axs[1].text(a, e, str(round(e, 4)))
        sns.despine()
        return pd.Series(dict(zip(test_auc, test_ks)))

    def tune_n_estimators(self, n_estimators, plot=True):
        self.estimator.set_params(n_estimators=n_estimators)
        xgb_param = self.estimator.get_xgb_params()  # XGBOOST type param
        cvresult = xgb.cv(xgb_param, self.xgbtrain,
                          num_boost_round=n_estimators,
                          nfold=self.cv_folds,
                          metrics=['auc'],
                          feval=ks_xgb,
                          maximize=True,
                          as_pandas=True,
                          early_stopping_rounds=self.early_stopping_rounds,
                          verbose_eval=30)
        # best_estimators = cvresult.shape[0]-self.early_stopping_rounds
        best_estimators = cvresult.shape[0]
        self.best_estimators = best_estimators
        self.estimator.set_params(n_estimators=best_estimators)
        if plot:
            x = cvresult.index
            train_auc = cvresult['train-auc-mean']
            train_auc_std = cvresult['train-auc-std']
            train_ks = cvresult['train-ks-mean']
            train_ks_std = cvresult['train-ks-std']
            test_auc = cvresult['test-auc-mean']
            test_auc_std = cvresult['test-auc-std']
            test_ks = cvresult['test-ks-mean']
            test_ks_std = cvresult['test-ks-std']
            _, axs = plt.subplots(2, 1, figsize=(15, 15))
            line, = axs[0].plot(x, train_auc, label='train_auc')
            axs[0].fill_between(x, train_auc - train_auc_std, train_auc + train_auc_std, color=line.get_color(),
                                alpha=.3)
            line, = axs[0].plot(x, test_auc, label='test_auc')
            axs[0].fill_between(x, test_auc - test_auc_std, test_auc + test_auc_std, color=line.get_color(), alpha=.3)
            axs[0].legend(loc=2)
            axs[1].plot(x, train_ks, label='train_ks')
            axs[1].fill_between(x, train_ks - train_ks_std, train_ks + train_ks_std, color=line.get_color(), alpha=.3)
            axs[1].plot(x, test_ks, label='test_ks')
            axs[1].fill_between(x, test_ks - test_ks_std, test_ks + test_ks_std, color=line.get_color(), alpha=.3)
            axs[1].legend(loc=2)
            axs[1].set_xlabel('n_estimators')
            sns.despine()
            # plt.show()
        return cvresult

    def set_n_estimators(self, n_estimators_hand):
        """
        有时候就是希望自己手工来设置树的个数就用这个函数
        """
        self.estimator.set_params(n_estimators=n_estimators_hand)

    def tune_depth_child_weight(self, max_depth=None, min_child_weight=None):
        """
        max_depth and min_child_weight input: list
        """
        max_depth = max_depth if max_depth is not None else max_depth_space(self.feature_size)
        min_child_weight = min_child_weight if min_child_weight is not None else min_child_weight_space(self.train_size)
        param_grid = {'max_depth': max_depth,
                      'min_child_weight': min_child_weight}
        gsearch = GridSearchCV(estimator=self.estimator, param_grid=param_grid,
                               scoring='roc_auc', iid=False, cv=self.cv_folds,return_train_score=True)
        gsearch.fit(self.train, self.labels_train)
        best_parameters = gsearch.best_params_
        best_max_depth = best_parameters["max_depth"]
        best_min_child_weight = best_parameters['min_child_weight']
        self.best_max_depth = best_max_depth
        self.best_min_child_weight = best_min_child_weight
        self.estimator.set_params(max_depth=best_max_depth)
        self.estimator.set_params(min_child_weight=best_min_child_weight)
        gresult = pd.DataFrame(gsearch.cv_results_)
        return gresult

    def set_depth_child_weight(self, max_depth_hand, min_child_weight_hand):
        """
        max_depth_hand and min_child_weight_hand input: int
        """
        self.estimator.set_params(max_depth=max_depth_hand)
        self.estimator.set_params(min_child_weight=min_child_weight_hand)

    def tune_gamma(self, gamma=None):
        gamma = gamma if gamma is not None else [i / 10.0 for i in range(0, 10)]
        param_grid = {'gamma': gamma}
        gsearch = GridSearchCV(estimator=self.estimator, param_grid=param_grid,
                               scoring='roc_auc', iid=False, cv=self.cv_folds,return_train_score=True)
        gsearch.fit(self.train, self.labels_train)
        best_parameters = gsearch.best_params_
        best_gamma = best_parameters['gamma']
        self.best_gamma = best_gamma
        self.estimator.set_params(gamma=best_gamma)
        gresult = pd.DataFrame(gsearch.cv_results_)
        return gresult

    def set_gamma(self, gamma_hand):
        self.estimator.set_params(gamma=gamma_hand)

    def tune_gamma_child_weight(self, gamma=None, min_child_weight=None):
        gamma = gamma if gamma is not None else [i / 10.0 for i in range(0, 10)]
        min_child_weight = min_child_weight if min_child_weight is not None else min_child_weight_space(self.train_size)
        param_grid = {'gamma': gamma,
                      'min_child_weight': min_child_weight}
        gsearch = GridSearchCV(estimator=self.estimator, param_grid=param_grid,
                               scoring='roc_auc', iid=False, cv=self.cv_folds,return_train_score=True)
        gsearch.fit(self.train, self.labels_train)
        best_parameters = gsearch.best_params_
        best_gamma = best_parameters['gamma']
        best_min_child_weight = best_parameters['min_child_weight']
        self.best_gamma = best_gamma
        self.best_min_child_weight = best_min_child_weight
        self.estimator.set_params(gamma=best_gamma, min_child_weight=best_min_child_weight)
        gresult = pd.DataFrame(gsearch.cv_results_)
        return gresult

    def set_gamma_child_weight(self, gamma_hand, min_child_weight_hand):
        """
        max_depth_hand and min_child_weight_hand input: int
        """
        self.estimator.set_params(gamma=gamma_hand)
        self.estimator.set_params(min_child_weight=min_child_weight_hand)

    def tune_subsample_colsample(self, subsample=None, colsample_bytree=None):
        subsample = subsample if subsample is not None else [i / 10.0 for i in range(5, 10)]
        colsample_bytree = colsample_bytree if colsample_bytree is not None else [i / 10.0 for i in range(5, 10)]
        param_grid = {'subsample': subsample,
                      'colsample_bytree': colsample_bytree}
        gsearch = GridSearchCV(estimator=self.estimator, param_grid=param_grid,
                               scoring='roc_auc', iid=False, cv=self.cv_folds,return_train_score=True)
        gsearch.fit(self.train, self.labels_train)
        # best_parameters = gsearch.best_estimator_.get_params()
        best_parameters = gsearch.best_params_
        best_subsample = best_parameters["subsample"]
        best_colsample_bytree = best_parameters['colsample_bytree']
        self.best_subsample = best_subsample
        self.best_colsample_bytree = best_colsample_bytree
        self.estimator.set_params(subsample=best_subsample)
        self.estimator.set_params(colsample_bytree=best_colsample_bytree)
        gresult = pd.DataFrame(gsearch.cv_results_)
        return gresult

    def set_subsample_colsample(self, subsample_hand, colsample_bytree_hand):
        self.estimator.set_params(subsample=subsample_hand)
        self.estimator.set_params(colsample_bytree=colsample_bytree_hand)

    def tune_alpha_lambda(self, reg_alpha=None, reg_lambda=None):
        reg_alpha = reg_alpha if reg_alpha is not None else [0.001, 0.01, 0.1, 1, 10, 100]
        reg_lambda = reg_lambda if reg_lambda is not None else [0.001, 0.01, 0.1, 1, 10, 100]
        param_grid = {'reg_alpha': reg_alpha,
                      'reg_lambda': reg_lambda}
        gsearch = GridSearchCV(estimator=self.estimator, param_grid=param_grid,
                               scoring='roc_auc', iid=False, cv=self.cv_folds,return_train_score=True)
        gsearch.fit(self.train, self.labels_train)
        best_parameters = gsearch.best_params_
        best_reg_alpha = best_parameters["reg_alpha"]
        best_reg_lambda = best_parameters['reg_lambda']
        self.best_reg_alpha = best_reg_alpha
        self.best_reg_lambda = best_reg_lambda
        self.estimator.set_params(reg_alpha=best_reg_alpha)
        self.estimator.set_params(reg_lambda=best_reg_lambda)
        gresult = pd.DataFrame(gsearch.cv_results_)
        return gresult

    def set_alpha_lambda(self, reg_alpha_hand, reg_lambda_hand):
        self.estimator.set_params(reg_alpha=reg_alpha_hand)
        self.estimator.set_params(reg_lambda=reg_lambda_hand)

    def tune_learning_rate(self, learning_rate=None):
        learning_rate = learning_rate if learning_rate is not None else [0.001, 0.01, 0.1, 0.2]
        param_grid = {'learning_rate': learning_rate}
        gsearch = GridSearchCV(estimator=self.estimator, param_grid=param_grid,
                               scoring='roc_auc', iid=False, cv=self.cv_folds,return_train_score=True)
        gsearch.fit(self.train, self.labels_train)
        best_parameters = gsearch.best_params_
        best_learning_rate = best_parameters['learning_rate']
        self.best_learning_rate = best_learning_rate
        self.estimator.set_params(learning_rate=best_learning_rate)
        gresult = pd.DataFrame(gsearch.cv_results_)
        return gresult

    def set_learning_rate(self, learning_rate_hand):
        self.estimator.set_params(learning_rate=learning_rate_hand)

    def tune_n_estimators_learning_rate(self, learning_rate=None, n_estimators=None):
        learning_rate = learning_rate if learning_rate is not None else [0.001, 0.01, 0.1, 0.2]
        n_estimators = n_estimators if n_estimators is not None else [50, 100, 150, 200, 250, 300, 400]
        param_grid = {'learning_rate': learning_rate,
                      'n_estimators': n_estimators}
        gsearch = GridSearchCV(estimator=self.estimator, param_grid=param_grid,
                               scoring='roc_auc', iid=False, cv=self.cv_folds,return_train_score=True)
        gsearch.fit(self.train, self.labels_train)
        best_parameters = gsearch.best_params_
        best_learning_rate = best_parameters['learning_rate']
        best_n_estimators = best_parameters['n_estimators']
        self.best_learning_rate = best_learning_rate
        self.best_n_estimators = best_n_estimators
        self.estimator.set_params(learning_rate=best_learning_rate, n_estimators=best_n_estimators)
        gresult = pd.DataFrame(gsearch.cv_results_)
        return gresult

    def set_n_estimators_learning_rate(self, n_estimators,learning_rate):
        self.estimator.set_params(learning_rate=learning_rate, n_estimators=n_estimators)

    def tune_gamma_weight(self, gamma=None, min_child_weight=None):
        gamma = gamma if gamma is not None else [0.1, 0.5, 1, 3, 5, 7, 9, 11, 15]
        min_child_weight = min_child_weight if min_child_weight is not None else [1, 5, 10, 20, 50, 80, 120, 150, 200,
                                                                                  250, 300, 400]
        param_grid = {'gamma': gamma,
                      'min_child_weight': min_child_weight}
        gsearch = GridSearchCV(estimator=self.estimator, param_grid=param_grid,
                               scoring='roc_auc', iid=False, cv=self.cv_folds,return_train_score=True)
        gsearch.fit(self.train, self.labels_train)
        # best_parameters = gsearch.best_estimator_.get_params()
        best_parameters = gsearch.best_params_
        best_gamma = best_parameters["gamma"]
        best_min_child_weight = best_parameters['min_child_weight']
        self.best_gamma = best_gamma
        self.best_min_child_weight = best_min_child_weight
        self.estimator.set_params(max_depth=best_gamma)
        self.estimator.set_params(min_child_weight=best_min_child_weight)
        gresult = pd.DataFrame(gsearch.cv_results_)
        return gresult

    def tune_skip_drop(self, skip_drop=0.5, plot=True):
        """
        Scikit-Learn api在Dart下暂时没法gridsearch,或者说是我自己还没法发现姿势
        """
        xgb_param = self.estimator.get_xgb_params()  # XGBOOST type param
        xgb_param['skip_drop'] = skip_drop
        cvresult = xgb.cv(xgb_param, self.xgbtrain,
                          num_boost_round=200,
                          nfold=self.cv_folds,
                          metrics=['auc'],
                          feval=ks_xgb,
                          maximize=True,
                          as_pandas=True,
                          early_stopping_rounds=self.early_stopping_rounds,
                          verbose_eval=30)
        if plot:
            x = cvresult.index
            train_auc = cvresult['train-auc-mean']
            train_auc_std = cvresult['train-auc-std']
            train_ks = cvresult['train-ks-mean']
            train_ks_std = cvresult['train-ks-std']
            test_auc = cvresult['test-auc-mean']
            test_auc_std = cvresult['test-auc-std']
            test_ks = cvresult['test-ks-mean']
            test_ks_std = cvresult['test-ks-std']
            _, axs = plt.subplots(2, 1, figsize=(15, 15))
            axs[0].plot(x, train_auc, label='train_auc')
            axs[0].fill_between(x, train_auc - train_auc_std, train_auc + train_auc_std, color=line.get_color(),
                                alpha=.3)
            axs[0].plot(x, test_auc, label='test_auc')
            axs[0].fill_between(x, test_auc - test_auc_std, test_auc + test_auc_std, color=line.get_color(), alpha=.3)
            axs[0].legend(loc=2)
            axs[1].plot(x, train_ks, label='train_ks')
            axs[1].fill_between(x, train_ks - train_ks_std, train_ks + train_ks_std, color=line.get_color(), alpha=.3)
            axs[1].plot(x, test_ks, label='test_ks')
            axs[1].fill_between(x, test_ks - test_ks_std, test_ks + test_ks_std, color=line.get_color(), alpha=.3)
            axs[1].legend(loc=2)
            axs[1].set_xlabel('n_estimators')
            sns.despine()
            # plt.show()
        return cvresult

    def feature_imp_score(self, plot=True):
        fscore = self.estimator.get_booster().get_score(importance_type='total_gain')
        fscore_sort = pd.Series(fscore).sort_values(ascending=False)
        fcount = len(fscore_sort)
        width = 15 if fcount <= 60 else fcount / 4
        if plot:
            fscore_sort.plot(kind='bar', title='Feature Importances', figsize=(width, 8))
            # plt.xticks(rotation=60)
        return fscore_sort

    def get_pred_leaf(self):
        pred_leaf = self.estimator.get_booster().predict(self.xgbtest, pred_leaf=True)
        return pred_leaf

    def get_pred_contribs(self):
        pred_contribs = self.estimator.get_booster().predict(self.xgbtest, pred_contribs=True)
        return pred_contribs

    def dump_best_model_file(self, filename):
        self.estimator.get_booster().dump_model(filename, with_stats=True)

    def dump_best_model(self):
        dump = self.estimator.get_booster().get_dump('', with_stats=True)
        return dump


"""
下面的部分为解析XGBoost模型
"""


class XgbModel:
    def __init__(self):
        self.XgbTrees = []

    def add_tree(self, tree):
        self.XgbTrees.append(tree)


class XgbTreeNode:
    def __init__(self):
        self.Feature = ''
        self.Gain = 0.0
        self.Cover = 0.0
        self.Number = -1
        self.LeftChild = None
        self.RightChild = None
        self.Missing = None
        self.LeafValue = 0.0
        self.SplitValue = 0.0
        self.IsLeaf = False

    def __lt__(self, other):
        return self.Number < other.Number


class XgbTree:
    def __init__(self, node):
        self.left = None
        self.right = None
        self.node = node


def parse_tree(xgbfir_tree):
    node = xgbfir_tree.node
    left = xgbfir_tree.left
    right = xgbfir_tree.right
    return node, left, right


def parse_node(tree_node):
    node_dict = {}
    if tree_node.IsLeaf:
        node_dict['isleaf'] = tree_node.IsLeaf
        node_dict['feature'] = tree_node.Feature
        node_dict['splitvalue'] = tree_node.SplitValue
        node_dict['leftchild'] = tree_node.LeftChild
        node_dict['rightchild'] = tree_node.RightChild
        node_dict['missingchild'] = tree_node.Missing
        node_dict['gain'] = tree_node.Gain
        node_dict['LeafValue'] = tree_node.LeafValue
    else:
        node_dict['isleaf'] = tree_node.IsLeaf
        node_dict['feature'] = tree_node.Feature
        node_dict['splitvalue'] = tree_node.SplitValue
        node_dict['leftchild'] = tree_node.LeftChild
        node_dict['rightchild'] = tree_node.RightChild
        node_dict['missingchild'] = tree_node.Missing
        node_dict['gain'] = tree_node.Gain
    return node_dict


def add_def(func):
    @wraps(func)
    def wrapper(*args):
        if args[1] <= 1:
            print("def tree{}(r):".format(args[2]))
        return func(*args)

    return wrapper


@add_def
def tree_code(xgbfir_tree, depth, tree_num):
    node, _, _ = parse_tree(xgbfir_tree)
    node_dict = parse_node(node)
    if node_dict:
        isleaf = node_dict.get('isleaf')
        feature = node_dict['feature']
        splitvalue = node_dict['splitvalue']
        leftchild = node_dict['leftchild']
        # rightchild = node_dict['rightchild']
        missingchild = node_dict['missingchild']
        indent = "    " * depth
        if not isleaf:
            if missingchild == leftchild:
                print(indent + "if r['" + feature + "'] <= " + str(splitvalue) + " or pd.isnull(r['" + feature + "'])" + ":")
                tree_code(xgbfir_tree.left, depth + 1, tree_num)
                print(indent + "elif r['" + feature + "'] > " + str(splitvalue) + ":")
                tree_code(xgbfir_tree.right, depth + 1, tree_num)
            else:
                print(indent + "if r['" + feature + "'] <= " + str(splitvalue) + ":")
                tree_code(xgbfir_tree.left, depth + 1, tree_num)
                print(indent + "elif r['" + feature + "'] > " + str(splitvalue) + " or pd.isnull(r['" + feature + "'])" + ":")
                tree_code(xgbfir_tree.right, depth + 1, tree_num)
        else:

            leaf_value = node_dict.get('LeafValue')
            print(indent + "return " + str(leaf_value))


class XgbParser(object):
    def __init__(self, xgbfile=None, xgbdump=None, verbosity=0,maxtreenum=1000):
        self._verbosity = verbosity
        self.nodeRegex = re.compile("(\d+):\[(.*)<(.+)\]\syes=(.*),no=(.*),missing=(.*),gain=(.*),cover=(.*)")
        self.leafRegex = re.compile("(\d+):leaf=(.*),cover=(.*)")
        self.model = XgbModel()
        self.model_file = xgbfile
        self.model_dump = xgbdump
        self.maxtree = maxtreenum

    def construct_xgbtree(self, tree):
        if tree.node.LeftChild != None:
            tree.left = XgbTree(self.xgbNodeList[tree.node.LeftChild])
            self.construct_xgbtree(tree.left)
        if tree.node.RightChild != None:
            tree.right = XgbTree(self.xgbNodeList[tree.node.RightChild])
            self.construct_xgbtree(tree.right)

    def parse_xgbtree_node(self, line):
        node = XgbTreeNode()
        if "leaf" in line:
            m = self.leafRegex.match(line)
            node.Number = int(m.group(1))
            node.LeafValue = float(m.group(2))
            node.Cover = float(m.group(3))
            node.IsLeaf = True
        else:
            m = self.nodeRegex.match(line)
            node.Number = int(m.group(1))
            node.Feature = m.group(2)
            node.SplitValue = float(m.group(3))
            node.LeftChild = int(m.group(4))
            node.RightChild = int(m.group(5))
            node.Missing = int(m.group(6))
            node.Gain = float(m.group(7))
            node.Cover = float(m.group(8))
            node.IsLeaf = False
        return node

    def get_xgbmodel_from_file(self):
        self.xgbNodeList = {}
        numTree = 0
        with open(self.model_file) as f:
            for line in f:
                line = line.strip()
                if (not line) or line.startswith('booster'):
                    if any(self.xgbNodeList):
                        numTree += 1
                        if self._verbosity >= 2:
                            sys.stdout.write("Constructing tree #{}\n".format(numTree))
                        tree = XgbTree(self.xgbNodeList[0])
                        self.construct_xgbtree(tree)

                        self.model.add_tree(tree)
                        self.xgbNodeList = {}
                        if numTree == self.maxtree:
                            if self._verbosity >= 1:
                                print('maxTrees reached')
                            break
                else:
                    node = self.parse_xgbtree_node(line)
                    if not node:
                        return None
                    self.xgbNodeList[node.Number] = node

            if any(self.xgbNodeList) and ((self.maxtree < 0) or (numTree < self.maxtree)):
                numTree += 1
                if self._verbosity >= 2:
                    sys.stdout.write("Constructing tree #{}\n".format(numTree))
                tree = XgbTree(self.xgbNodeList[0])
                self.construct_xgbtree(tree)
                self.model.add_tree(tree)
                self.xgbNodeList = {}

    def get_xgbmodel_from_memory(self):
        self.xgbNodeList = {}
        numTree = 0
        for booster_line in self.model_dump:
            self.xgbNodeList = {}
            for line in booster_line.split('\n'):
                line = line.strip()
                if not line:
                    continue
                node = self.parse_xgbtree_node(line)
                if not node:
                    return None
                self.xgbNodeList[node.Number] = node
            numTree += 1
            tree = XgbTree(self.xgbNodeList[0])
            self.construct_xgbtree(tree)
            self.model.add_tree(tree)
            if numTree == self.maxtree:
                print('maxTrees reached')
                break

    def deploy(self, filename):
        if self.model_file:
            self.get_xgbmodel_from_file()
        elif self.model_dump:
            self.get_xgbmodel_from_memory()
        else:
            print('请导入模型文件')
        trees = self.model.XgbTrees
        tree_num = len(trees)
        orig_stdout = sys.stdout
        f = open(filename, 'w')
        sys.stdout = f
        for i in range(tree_num):
            tree_code(trees[i], 1, i)
        sys.stdout = orig_stdout
        f.close()


def sigmod(x):
    return np.exp(x) / (1 + np.exp(x))


def xgb_proba(df):
    tree_columns = [i for i in df.columns if i.startswith('tree')]
    df['sum_tree'] = df.loc[:, tree_columns].sum(axis=1)
    df['proba'] = df['sum_tree'].map(sigmod)
    return df
