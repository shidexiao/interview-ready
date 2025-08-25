# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
模块描述：特征处理（Feature Process, FP）
功能包括：
1. vif_table          :方差膨胀因子（VIF）计算, 只适用于LR模型
2. iv_grouply_table   :信息量分组计算
3. var_cluster        :变量聚类实现降维, 再进行变量筛选
4. correlation_plot   :计算各变量的Pearson相关性系数, 并可视化
"""


def vif_table(input_df, var_list, save_file_path=None):
    """
    ----------------------------------------------------------------------
    功能：方差膨胀因子（VIF）计算, 只适用于LR模型
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param var_list: list, 需要计算VIF的变量列表
    ----------------------------------------------------------------------
    :return output_df: pd.DataFrame, 计算结果
    ----------------------------------------------------------------------
    示例：
    >>> vif_table(input_df=df, var_list=c_cols)
    >>>
    	var	VIF Factor
    0	Intercept	1.3
    1	gf_score_woe	1.2
    2	app2vec_score_woe	1.2
    3	applist_score_woe	1.2
    4	tongdun_score_woe	1.1
    ----------------------------------------------------------------------
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm

    X = input_df[var_list].copy()
    X.loc[:, 'Intercept'] = 1

    output_df = pd.DataFrame()
    output_df['var'] = X.columns
    output_df['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    output_df = output_df.sort_values(by='VIF Factor', ascending=False).reset_index(drop=True)
    output_df.round(1)
    
    return output_df


def iv_grouply_table(input_df, target_var, var_list, 
                     group_var=None, group_list=None, 
                     bin_num=5, detail=False, save_file_path=None):
    """
    ----------------------------------------------------------------------
    功能：信息量分组计算
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param target_var: str, 目标变量
    :param var_list: list, 需要计算IV的变量列表
    :param group_var: string, 分组依据，如按[组，月，周]。默认值=None, 对整个数据集统计。示例: 'apply_month'
    :param group_list: list, 分组元素取值, 若为None，则默认为所有取值。示例: [201701, 201702]
    :param bin_num: int, 分箱数。默认值=5.
    :param detail: bool, 是否显示IV计算过程各分箱明细. 默认值=False, 不显示。
    :param save_file_path: string, csv文件保存路径. 默认值=None
    ----------------------------------------------------------------------
    :return output_df: 变量iv值
    ----------------------------------------------------------------------
    示例：
    >>> iv_grouply_table(input_df=raw_data[raw_data['user_gaopao_180d'] == 1], 
                         target_var='flag1', 
                         var_list=spec_feats,
                         group_var='apply_month', 
                         group_list=None, 
                         bin_num=5,
                         detail=0, save_file_path=None)
    >>> 
    	var	seg1	seg2	mean	std	cv	iv_rank	target
    1	Fare	0.648085	0.571201	0.609643	0.054365	0.089029	1	Survived
    0	Age	0.222415	0.265318	0.243866	0.030337	0.123892	2	Survived
    ------------
    >>> iv_grouply_table(input_df=raw_data[raw_data['user_gaopao_180d'] == 1], 
                         target_var='flag1', 
                         var_list=spec_feats,
                         group_var='apply_month', 
                         group_list=None, 
                         bin_num=5,
                         detail=1, save_file_path=None)
    >>>
    	apply_month	feature	labels	bins	min_score	max_score	obs	bad	good	bad_rate	good_rate	odds_good	woe	bin_iv	iv	target
    2018-01	app_platform_num_3m	0	-inf	-inf	3.000000	73	2	71	0.027397	0.972603	35.500000	1.088421	0.194885	1.122779	flag1
    2018-01	app_platform_num_3m	1	3.000000	3.000000	5.000000	63	0	63	0.000000	1.000000	inf	2.355170	0.564166	1.122779	flag1
    2018-01	app_platform_num_3m	2	5.000000	5.000000	7.000000	51	10	41	0.196078	0.803922	4.100000	-1.070125	0.319595	1.122779	flag1
    2018-01	app_platform_num_3m	3	7.000000	7.000000	10.000000	43	5	38	0.116279	0.883721	7.600000	-0.452963	0.037499	1.122779	flag1
    2018-01	app_platform_num_3m	4	10.000000	10.000000	inf	55	5	50	0.090909	0.909091	10.000000	-0.178526	0.006634	1.122779	flag1
    ----------------------------------------------------------------------
    """
    input_df_copy = input_df.copy()
    input_df_copy = input_df_copy.loc[(input_df_copy[target_var].isin([0,1])), :]

    if not isinstance(input_df_copy, pd.core.frame.DataFrame):
        raise Exception('参数input_df的格式错误，应为pandas.core.frame.DataFrame')

    cols = list(input_df_copy.columns)
    if group_var is not None:
        if group_var not in set(input_df_copy.columns):
            raise Exception('参数group_var取值包含不属于input_df的变量，请检查!')
        input_df_copy[group_var] = input_df_copy[group_var].apply(lambda x: str(x))
        input_df_copy['group'] = input_df_copy[group_var]
    else:
        group_var = 'group'
        input_df_copy[group_var] = 'all'
          
    if group_list is None: 
        group_list = sorted(list(set(input_df_copy[group_var])))
    else:
        group_list = sorted(list(set(group_list)))
    group_list = [str(x) for x in group_list]

    if var_list is not None:
        if not set(var_list).issubset(set(cols)):
            raise Exception('参数var_list取值包含不属于input_df的变量，请检查!')
        else:
            cols = var_list
    
    col_dtype = dict(input_df_copy.dtypes)
    def iv_calculate(x, y, bin_num, col_dtype, detail):
        """
        ----------------------------------------------------------------------
        功能：调用weight_of_evidence模块计算变量iv
        ----------------------------------------------------------------------
        :param x: pd.Series, 自变量取值列表
        :param y: pd.Series, 因变量取值列表
        :param col_dtype: dict, 自变量类型
        ----------------------------------------------------------------------
        :return iv: float, 信息量
        ----------------------------------------------------------------------
        """ 
        v_type = 'd' if col_dtype[x.name] == object else 'c'
        import weight_of_evidence as py_woe
        woe = py_woe.WoE(qnt_num=bin_num-1, min_block_size=2, v_type=v_type, t_type='b') 
        woe.fit(x, y)
        return woe.bins if detail else woe.iv
        
    """对各变量统计IV"""
    def iv_grouply_calculate(df_):
        iv_ = df_[var_list] \
                 .apply(iv_calculate, axis=0, args=(df_[target_var], bin_num, col_dtype, detail)) \
                 .reset_index()
        iv_.columns = ['var','iv']
        return iv_ 
    
    if detail:
        output_df = None
        for gp in group_list:
            gp_df = input_df_copy[input_df_copy[group_var] == gp]
            for var in var_list:
                woe_tmp = iv_calculate(x=gp_df[var], y=gp_df[target_var], bin_num=bin_num, col_dtype=col_dtype, detail=detail)
                woe_tmp[group_var] = str(gp)
                output_df = pd.concat([output_df, woe_tmp])
        
        output_df['target'] = target_var
        output_cols = [group_var, 'feature', 'labels', 'bins', 'min_score', 'max_score', 'obs', \
                       'bad', 'good', 'bad_rate', 'good_rate', 'odds_good', 'woe', 'bin_iv', 'iv', 'target']
        output_df = output_df[output_cols]
    else:
        if group_list is None: 
            stat_df = input_df_copy[cols] \
                                 .apply(iv_calculate, axis=0, args=(input_df_copy[target_var],  bin_num, col_dtype, detail)) \
                                 .reset_index()
            stat_df.columns = ['var','iv']
        else:
            stat_df = input_df_copy.groupby(group_var) \
                                 .apply(iv_grouply_calculate) \
                                 .reset_index(level=group_var)

        sort_value = ['group', 'iv'] if group_list is not None else ['iv']
        stat_df = stat_df.sort_values(by=['iv'], ascending=False)
        stat_df['group'] = stat_df[group_var]
        group_list = sorted(list(set(stat_df['group'])))
        
        output_df = pd.DataFrame(columns=['var'] + group_list)
        output_df['var'] = sorted(cols)
        for g in group_list:
            sub_stat_df = stat_df[stat_df['group'] == g]
            sub_stat_df = sub_stat_df.sort_values(by=['var'], ascending=1)
            output_df[str(g)] = list(sub_stat_df['iv'])
        output_df['mean'] = output_df.apply(lambda x: x[group_list].mean(), axis=1)
        output_df['std'] = output_df.apply(lambda x: x[group_list].std(), axis=1)
        output_df['temp'] = 0.001
        output_df['cv'] = output_df['std'] / (output_df['temp'] + output_df['mean'])
        output_df = output_df.sort_values(by=['mean'], ascending=0)
        output_df['iv_rank'] = [x for x in range(1, output_df.shape[0] + 1)]
        output_df = output_df.drop(['temp'], axis=1)
        output_df['target'] = target_var
    
    """文件保存"""
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.csv'):
            raise Exception('参数save_file_path不是csv文件后缀，请检查!')
        output_df.to_csv(save_file_path, encoding='utf-8', index=False)
        
    return output_df


def var_cluster(input_df, n_clusters=10, var_list=None, save_file_path=None):
    '''
    ----------------------------------------------------------------------
    功能：变量聚类实现降维, 再进行变量筛选
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param n_clusters, int, 聚类目标簇数, 默认=10
    :param var_list: list, 变量列表，默认值=None，对所有变量统计。必须要在input_df中，否则抛异常。
    :param save_file_path: string, csv文件保存路径. 默认值=None
    ----------------------------------------------------------------------
    :return output_df: pd.DataFrame, 变量聚类结果
    ----------------------------------------------------------------------
    示例：
    >>> var_cluster(input_df=df, n_clusters=3, var_list=None)
    >>>  
    	var	cluster
    0	Pclass	0
    1	SibSp	0
    2	Parch	0
    3	PassengerId	1
    4	Age	1
    5	Survived	2
    6	Fare	2
    7	score	2
    ----------------------------------------------------------------------
    '''
    from sklearn.cluster import FeatureAgglomeration
    from sklearn import preprocessing

    input_df_copy = input_df.copy()
    input_df_copy.fillna(value=-9999999, inplace=True)
    
    cols = list(input_df_copy.columns)
    cols_type_dict = dict(input_df_copy.dtypes)
    str_cols = [x for x in cols if cols_type_dict[x] == object]

    if var_list is None:
        var_list = list(set(cols) - set(str_cols))
    else:
        if not set(var_list).issubset(set(cols)):
            raise Exception('参数var_list取值包含不属于input_df_copy的变量，请检查!')
        var_list = list(set(var_list) - set(str_cols))
        
    X_Scale = preprocessing.StandardScaler().fit_transform(input_df_copy[var_list])
    ward = FeatureAgglomeration(n_clusters=n_clusters, linkage='ward')
    ward.fit(X_Scale)
    clusters = list(ward.labels_)
    
    output_df = pd.DataFrame()
    output_df['var'] = var_list
    output_df['cluster'] = clusters
    output_df = output_df.sort_values(by=['cluster'], ascending=1).reset_index(drop=1)
    
    """文件保存至指定路径"""
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.csv'):
            raise Exception('参数save_file_path不是csv文件后缀，请检查!')
        output_df.to_csv(save_file_path, encoding='utf-8', index=True)
    
    return output_df


def correlation_plot(input_df, var_list, save_file_path=None):
    '''
    功能：计算各变量的Pearson相关性系数, 并可视化
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param var_list: 特征list
    :param save_file_path: string, jpg文件保存路径. 默认值=None
    ----------------------------------------------------------------------
    :return output_df, 相关性系数矩阵
    ----------------------------------------------------------------------
    示例：
    >>> correlation_plot(input_df=df, var_list=d_cols, save_file_path=None)
    >>>      
    	Parch	SibSp	Pclass
    Parch	1.000000	0.414838	0.018443
    SibSp	0.414838	1.000000	0.083081
    Pclass	0.018443	0.083081	1.000000
    ----------------------------------------------------------------------
    '''
    output_df = input_df[var_list].corr() 
    xticks = list(output_df.columns)
    yticks = list(output_df.index)[::-1]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    
    import seaborn as sns
    sns.heatmap(output_df, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})

    ax1.set_xticklabels(xticks, rotation=90, fontsize=10)
    ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
    plt.show()
    
    """step4: 保存输出"""
    if save_file_path:
        if not isinstance(save_file_path, str):
            raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
        elif save_file_path.endswith('.jpg'):
            raise Exception('参数save_file_path不是jpg文件后缀，请检查!')
        plt.savefig(save_file_path)
        print('成功保存至:{}'.format(save_file_path))
    
    return output_df

