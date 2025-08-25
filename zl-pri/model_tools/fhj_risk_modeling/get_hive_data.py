import os
import time
import pandas as pd
from pyhive import hive


"""
模块描述：获取数据（Get Data, GD）
功能包括：
1. 取数（Get Data, GD）: 连接Hive数据库，获取数据并保存在本地
"""


def get_data(query_sql, save_data_path, username='fenghaijie', password='xxx'):
    """
    ----------------------------------------------------------------------
    功能: 连接Hive数据库，获取数据并保存在本地
    ----------------------------------------------------------------------
    :param query_sql: str, hive sql语句
    :param save_data_path: str, 文件保存路径, 必须是csv格式
    :param username: str, 连接hive所用的用户名。本人OA用户名
    :param password: str, 本人OA密码
    ----------------------------------------------------------------------
    :return output_df: dataframe, 原始数据
    ----------------------------------------------------------------------
    """
    s_r = time.clock()
    
    if (type(save_data_path) != str) or (type(query_sql) != str):
        print('输入参数类型不匹配')
    
    print('获取订单数据...')
    conn = hive.Connection(host='172.16.28.87', port=10000, username=username, 
                           password=password, database='vdm_ds_dev', auth='LDAP')

    if os.path.exists(save_data_path):
        output_df = pd.read_csv(save_data_path, encoding='utf-8')
    else:
        output_df = pd.read_sql(query_sql, conn)
        output_df.rename(columns=lambda x: x.split('.')[1], inplace=True)
        output_df.to_csv(save_data_path, index=False, sep=',', encoding='utf-8', header=True) 
        print('已经保存在：' + save_data_path)
        
    e_r = time.clock()
    print('Running time: %.4g Minutes' % ((e_r - s_r) * 1.0 / 60))
    
    return output_df