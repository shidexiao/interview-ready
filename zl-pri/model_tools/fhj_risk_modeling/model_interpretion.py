# # -*- coding:utf-8 -*-
# __author__ = 'fenghaijie'

# import math
# import numpy as np
# import pandas as pd
# import shap

# """
# 模块描述：模型解释性(Model Interpretion, MI)
# 功能包括：
# 1. shap_value
# """


# def shap_value(trained_model, X_train, var_list):
#     """
#     ----------------------------------------------------------------------
#     功能：得到机器学习树模型各入模特征在训练集上的shap值
#     ----------------------------------------------------------------------
#     :param trained_model: 训练完成的机器学习树模型, 如xgboost或rf
#     :param X_train: pd.DataFrame, 训练集特征
#     :param var_list: list, 变量列表
#     ----------------------------------------------------------------------
#     :return output_df: pd.DataFrame, 各特征变量在训练集上的shap值
#     ----------------------------------------------------------------------
#     示例：
#     >>> trained_rf = train_model(model=rf, 
#                          X_train=develop_data_ins[feats].fillna(999999), 
#                          y_train=develop_data_ins[target_var], 
#                          X_valid=develop_data_oos[feats].fillna(999999), 
#                          y_valid=develop_data_oos[target_var])
#     >>> var_shap_df = shap_value(trained_model=trained_rf,
#                                  X_train=develop_data_ins[feats].fillna(999999),
#                                  var_list=c_cols)
#     >>> shap_df
#     	var	shap_value
#     0	Age	-17.900106
#     1	Fare	-49.150763
#     ----------------------------------------------------------------------
#     """
#     import shap
#     explainer = shap.TreeExplainer(trained_model)
#     shap_values = explainer.shap_values(X_train)
    
#     output_df = pd.DataFrame(list(X_train.columns), columns=['var'])
#     shap_value_list = [sum(shap_values[:,i]) for i in range(len(var_list))]
#     output_df['shap_value'] = shap_value_list
    
#     return output_df
#     