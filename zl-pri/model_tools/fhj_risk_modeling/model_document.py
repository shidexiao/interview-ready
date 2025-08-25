# # -*- coding:utf-8 -*-
# __author__ = 'fenghaijie / hjfeng0630@qq.com'

# import os
# import json
# import xlsxwriter
# import pandas as pd
# import numpy as np


# """
# 模块描述：模型开发文档(Model Document)
# 功能包括：将产生的模型文件进行汇总
# """


# def write_to_xlsx(save_file_path)
#     workbook = xlsxwriter.Workbook('xxx模型开发文档.xlsx')
    
#     """
#     #sheet1: 模型概述Summary
#     """
#     worksheet1 = workbook.add_worksheet('#Summary')
#     bold = workbook.add_format({'bold': 1})
    
    
#     """
#     #sheet2: 探索性数据分析
#     """
#     worksheet2 = workbook.add_worksheet('#数据探索')
    
    
#     """
#     #sheet3: 特征处理——WOE变换
#     """
#     worksheet3 = workbook.add_worksheet('#特征处理')
    
    
#     """
#     #sheet4: 特征筛选
#     """
#     worksheet4 = workbook.add_worksheet('#特征筛选')
    
    
#     """
#     #sheet5: 模型评估
#     """
#     worksheet5 = workbook.add_worksheet('#模型评估')
    
    


# # # 结果文件路径
# # result = pd.ExcelWriter(model_performance_path)

# # for csv_file_name in origin_file_list:
# #     file_path = out_version_path + '/%s' % csv_file_name
    
# #     # 临时文件
# #     temp_file_path = out_version_path + '/%s'% 'temp.xlsx'
# #     df = pd.read_csv(file_path)
# #     df.to_excel(temp_file_path)

# #     # 读取文件内容
# #     content = pd.read_excel(temp_file_path)
# #     if os.path.exists(temp_file_path):
# #         os.remove(temp_file_path)
        
# #     # sheet_name重命名
# #     sheet_name = csv_file_name[19: -4] # 需调整，我的取名格式为：'20180301_20181201_同盾子模型_特征CV.csv'
# #     print(sheet_name)
    
# #     # 转换为同一个表多个sheet
# #     content.to_excel(result, sheet_name, index=False)
     
# def model_desc_info(config):
#     """
#     功能：根据用户配置信息生成模型文件信息
#     ----------
#     :param config: json, 取值如下:
#             {
#             "project_name": "人品贷生意金",
#             "product_id": "人品贷",
#             "model_owner": "fenghaijie",
#             "version": "20190501",
#             "sample_divide": {"INS & OOS": "201801,02,03", "OOT1": '201901'},
#             "target_var": "s3d15",
#             }
#     ----------
#     :return model_desc: json, 模型描述信息
#     """
#     model_desc = {}
#     model_desc['model_id'] = 
#     model_desc['owner'] =  config['model_owner']
    
#     # target_var
#     model_desc['目标变量'] = 
    
#     # model datetime
#     model_desc['建模日期'] = time.strftime('%Y%m%d', time.localtime(time.time()))
    
#     # target_rate_stat
#     model_desc['目标变量分布_文件路径'] = 
#     # missing_rate_stat
#     model_desc['变量缺失率_文件路径'] = 
    
#     # edd_for_continue_var
#     model_desc['连续变量数据分布_文件路径'] = 
#     # edd_for_discrete_var
#     model_desc['离散变量数据分布_文件路径'] = 
    
#     # psi_for_continue_var
#     model_desc['连续变量稳定性_文件路径'] = 
#     # psi_for_discrete_var
#     model_desc['离散变量稳定性_文件路径'] = 
    
#     # ks_grouply_calculate
#     model_desc['连续变量KS_文件路径'] = 
#     # ks_table
#     model_desc['连续变量KS-Table_文件路径'] = 
#     # 
    
#     # 
#     model_desc['模型PMML文件路径']
#     # 
#     model_desc['模型PKL文件路径']
    
    
#     return pd.Series(model_desc)
    



# # 项目名称
# source = '人品贷标准版生意金A卡（信用）V2.0-同盾子模型'
# root_path = '/home/jovyan/经营贷/'

# # 目标变量
# target_var = 's3d15'

# # 版本管理
# version = '20190302'
# out_version_path = root_path + 'OUT/%s' % version
# model_version_path = root_path + 'MODEL/%s' % version
# if not os.path.exists(out_version_path): os.mkdir(out_version_path)  # 创建该版本的所有输出文件存放目录
# if not os.path.exists(model_version_path): os.mkdir(model_version_path) # 创建该版本的模型文件存放目录

# # 数据源表
# rawdata_path  = root_path + 'DATA/%s_%s_%s.csv'   % (sd, ed, source)
# data_dict_path = root_path + 'DATA/%s_数据字典.csv'   % (source)

# # 样本分布
# sample_distribution_path = root_path + 'OUT/%s/%s_%s_%s_样本分布.csv'   % (version, sd, ed, source)

# # 特征筛选
# var_edd_path          = root_path + 'OUT/%s/%s_%s_%s_特征EDD.csv'   % (version, sd, ed, source)
# var_missing_rate_path = root_path + 'OUT/%s/%s_%s_%s_特征缺失率.csv' % (version, sd, ed, source)
# feat_importance_path  = root_path + 'OUT/%s/%s_%s_%s_特征重要性.csv' % (version, sd, ed, source)
# var_ks_path           = root_path + 'OUT/%s/%s_%s_%s_特征KS.csv'    % (version, sd, ed, source)
# var_psi_path          = root_path + 'OUT/%s/%s_%s_%s_特征PSI.csv'   % (version, sd, ed, source)
# var_iv_path           = root_path + 'OUT/%s/%s_%s_%s_特征IV.csv'   % (version, sd, ed, source)

# # 模型结果
# #1.模型区分度(KS)
# model_ks_profile_path = root_path + 'OUT/%s/%s_%s_%s_模型KS_Profile.csv' % (version, sd, ed, source)
# model_ks_table_path   = root_path + 'OUT/%s/%s_%s_%s_模型KS_Table.csv' % (version, sd, ed, source)
# #2.模型稳定性(PSI)
# model_psi_profile_path = root_path + 'OUT/%s/%s_%s_%s_模型PSI_Profile.csv' % (version, sd, ed, source)
# model_psi_table_path   = root_path + 'OUT/%s/%s_%s_%s_模型PSI_Table.csv' % (version, sd, ed, source)
# #3.模型排序性(Lift, Bad Rate)
# model_bad_rate_path    = root_path + 'OUT/%s_%s_%s_%s_模型Bad_Rate.csv' % (version, sd, ed, source)
# model_lift_chart_path  = root_path + 'OUT/%s/%s_%s_%s_模型Lift_Chart.csv' % (version, sd, ed, source)

# # 模型参数
# #1.评分卡模型参数(LR)
# scorecard_params_path = root_path + 'MODEL/%s/%s_%s_%s_评分卡模型参数.csv'     % (version, sd, ed, source)
# binmap_path           = root_path + 'MODEL/%s/%s_%s_%s_评分卡binmap.csv'      % (version, sd, ed, source)
# #2.机器学习模型参数(XGBoost, RF)
# pmml_path       = root_path + 'MODEL/%s/%s_%s_%s_机器学习模型参数.pmml'      % (version, sd, ed, source)
# pkl_path        = root_path + 'MODEL/%s/%s_%s_%s_机器学习模型参数.pkl'       % (version, sd, ed, source)

# # 打分结果
# accepted_score_path = root_path + 'OUT/%s/%s_%s_%s_放贷订单得分.csv'    % (version, sd, ed, source)

# # 模型部署
# deploy_dataset_path = root_path + 'DATA/%s_%s_%s_%s_模型部署测试集.csv' % (version, sd, ed, source)