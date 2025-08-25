# -*- coding:utf-8 -*-
__author__ = 'fenghaijie'

import os
import pickle
import pandas as pd
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn_pandas import DataFrameMapper
from sklearn.externals import joblib

"""
模块描述：保存及载入模型文件（Save & Load Model）
功能包括：
1.pkl_save:    sklearn机器学习模型, 保存为pkl格式
2.pmml_save:   sklearn机器学习模型, 保存为pmml格式
3.binmap_save: 评分卡模型, 保存为binmap.csv格式
4.pkl_to_pmml: 将sklearn机器学习模型pkl文件转换为pmml文件
5.load_pkl_model: 载入pkl文件。
6.load_pkl_predict: 载入pkl文件直接预测
"""

def pkl_save(model, save_file_path):
    '''
    ----------------------------------------------------------------------
    功能：将训练完成的模型保存为pkl文件
    ----------------------------------------------------------------------
    :param model: sklearn机器学习包实例化后训练完毕的模型
    :param save_file_path: str, 保存模型文件的目标路径
    ----------------------------------------------------------------------
    :return None
    ----------------------------------------------------------------------
    '''
    if not save_file_path.endswith('.pkl'):
        raise Exception('参数save_file_path后缀必须为pkl, 请检查！')
        
    with open(save_file_path, 'wb') as f:
        pickle.dump(model, f, protocol=2)
    f.close()
    
    print('模型文件已保存至{}'.format(save_file_path))

    
def pmml_save(model, feat_list, save_file_path):
    '''
    ----------------------------------------------------------------------
    功能：将训练完成的模型保存为pmml文件
    ----------------------------------------------------------------------
    :param model: sklearn机器学习包实例化后训练完毕的模型
    :param feat_list: list, 最终入模的特征变量列表。
                      若不指定feats_list, 那么写入的pmml中特征名默认取值为['x0', 'x1', ...]
    :param save_file_path: str, 保存模型文件的目标路径
    ----------------------------------------------------------------------
    :return None
    ----------------------------------------------------------------------
    '''
    if not save_file_path.endswith('.pmml'):
        raise Exception('参数save_file_path后缀必须为pmml, 请检查！')
        
    mapper = DataFrameMapper([([i], None) for i in feat_list])  # 特征工程，注意这里的[i]
    pipeline = PMMLPipeline([('mapper', mapper), ("classifier", model)])
    sklearn2pmml(pipeline, pmml=save_file_path)
    
    print('模型文件已保存至{}'.format(save_file_path))
    
    
def binmap_save(input_df, save_file_path):
    '''
    ----------------------------------------------------------------------
    功能：模型使用，载入pkl文件。注意此时预测时列名为['x0', 'x1', ...]
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 评分卡建模生成的binmap
    :param save_file_path: str, 保存模型文件的目标路径
    ----------------------------------------------------------------------
    :return None
    ----------------------------------------------------------------------
    '''
    if save_file_path.endswith('.csv'):
        input_df.to_csv(save_file_path, encoding='utf-8', index=0)
    elif save_file_path.endswith('.xlsx'):
        input_df.to_excel(save_file_path)
    else:
        raise Exception('参数save_file_path后缀必须为csv 或 xlsx, 请检查！')
    
    print('模型文件已保存至{}'.format(save_file_path))
    
    
def pkl_to_pmml(pkl_path, pmml_path, feat_list):
    '''
    ----------------------------------------------------------------------
    功能：将pkl文件转换为pmml文件
    ----------------------------------------------------------------------
    :param pkl_path:  str, 需读取的pkl文件路径
    :param pmml_path: str, 需写入的pmml文件路径
    :param feat_list: list, 入模特征列表，注意顺序必须和pkl写入时一致.
    ----------------------------------------------------------------------
    :return : 生成pmml文件
    ----------------------------------------------------------------------
    '''
    if not os.path.exists(pkl_path):
        raise Exception('参数pkl_path指向的文件路径不存在, 请检查！')
        
    if not pmml_path.endswith('.pmml'):
        raise Exception('参数pmml_path后缀必须为pmml, 请检查！')
        
    mapper = DataFrameMapper([([i], None) for i in feat_list])
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    pipeline = PMMLPipeline([('mapper', mapper), ("classifier", model)])
    sklearn2pmml(pipeline, pmml=pmml_path)
    
    print('转换成功！模型已保存至{}'.format(pmml_path))
    
    
def load_pkl_model(save_file_path):
    '''
    ----------
    功能：模型使用，载入pkl文件。注意此时预测时列名为['x0', 'x1', ...]
    ----------
    :param save_file_path: str, 保存模型文件的目标路径
    ----------
    :return model: sklearn机器学习包实例类型。
                   预测时用法: model.predict_proba(df[feat_list])[:, 1]
    '''
    if not os.path.exists(save_file_path):
        raise Exception('参数save_file_path指向的文件路径不存在, 请检查！')
        
    model = joblib.load(save_file_path)
        
    return model


def load_pkl_predict(input_df, feat_list, save_file_path):
    '''
    ----------------------------------------------------------------------
    功能：载入所保存的模型pkl文件, 批量预测. 注意此时预测时列名为feat_list
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 待预测的样本集
    :param feat_list: list, 最终入模的特征变量列表。必须按原顺序.
    :param save_file_path: str, 保存模型文件的目标路径
    ----------------------------------------------------------------------
    :return output_df: pd.DataFrame, 预测完毕的样本集, 在input_df上新增一列['score']
    ----------------------------------------------------------------------
    '''
    cols = set(input_df.columns)
    if not set(feat_list).issubset(cols):
        raise Exception('参数feat_list取值包含不属于input_df的变量，请检查!')
    
    with open(save_file_path, "rb") as f:
         load_pipline = pickle.load(f)
    f.close()
    
    output_df = pd.DataFrame(load_pipline.predict_proba(input_df[feat_list])[:, 1], \
                            columns=['score'], index=input_df[feat_list].index)
    
    output_df = pd.concat([input_df, output_df], axis=1, sort=False)
    
    return output_df
   