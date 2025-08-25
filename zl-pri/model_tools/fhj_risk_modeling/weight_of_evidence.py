# -*- coding:utf-8 -*-
__author__ = 'fenghaijie / hjfeng0630@qq.com'

import math
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import cross_val_score

"""
WOE知识：
1）WOE变化趋势
这个趋势变化主要针对连续数值型变量，假设WOE的计算方式是bad/good，某个变量按照业务理解是值越大，坏用户概率越大。那么变量分箱后，WOE的变化趋势应该与实际的业务经验一致，也就是变量的值越大，WOE越大，且呈单调性变化。对于WOE没有呈单调性变化的变量（例如U型或倒U型），如果业务上能解释的通，那也可以采用该变量。WOE呈波浪形变化的变量建议不采用。
PS：在做WOE趋势分析时不考虑缺失的箱体，并且最好将变量分为4-8箱。

2）箱体之间WOE的差异分析
分箱的原则是组内差异小，组间差异大，所以箱体之间的WOE要有显著差异，个人认为WOE的差值至少要在0.1以上，这样每个箱体的好坏比才有区别。另外我觉得WOE最好不要出现跃阶式变化，例如第一个箱体的woe是0.1，下一个箱体WOE直接增大到0.9，这样会导致最后转化的分数也会呈跃阶式变化，这个对总体评分的分布及稳定性会有很大影响。箱体的WOE最好是单调线性变化的。

3）箱体的WOE绝对值大小
箱体的WOE值最好是在-1至1之间，如果WOE的绝对值大于1，说明这个箱体的坏用户占比或者好用户占比在65%以上，这种变量适合做单条策略，如果放到模型中，这个变量的权重可能会很大，会增加模型过拟合的危险，并会影响评分卡的稳定性。
"""


class WoE:
    """
    :Function Basic functionality for WoE bucketing of continuous and discrete variables
    :param self.bins: DataFrame WoE transformed variable and all related statistics
    :param self.iv: Information Value of the transformed variable
    """
    def __init__(self, qnt_num=5, min_block_size=10, spec_values=None, v_type='c', bins=None, t_type='b'):
        """
        ----------------------------------------------------------------------
        :param qnt_num: int, 分箱数. Number of buckets (quartiles) for continuous variable split
        :param min_block_size: int, 分箱内最少样本数. 
                               min number of obs in bucket (continuous variables), incl. optimization restrictions
        :param spec_values: List or Dictionary {'label': value} of special values (frequent items etc.)
        :param v_type: str, 'c' = continuous variable, 'd' = discrete variable
        :param bins: list, 预定义的分箱边界。例如[1, 2, 3], 则代表分为(1, 2], (2, 3]两个分箱
                     Predefined bucket borders for continuous variable split
        :param t_type : str, Binary 'b' or continous 'c' target variable
        ----------------------------------------------------------------------
        :return: 初始化的实例对象(initialized class)
        ----------------------------------------------------------------------
        """
        self.__qnt_num = qnt_num  # Num of buckets/quartiles
        self._predefined_bins = None if bins is None else np.array(bins)  # user bins for continuous variables
        self.v_type = v_type   
        self.v_name = None     # 变量名
        self._min_block_size = min_block_size  # Min num of observation in bucket
        self._gb_ratio = None  # Ratio of good and bad in the sample
        self.bins = None       # WoE Buckets (bins) and related statistics
        self.df = None         # 训练集,Training sample DataFrame with initial data and assigned woe
        self.qnt_num = None    # Number of quartiles used for continuous part of variable binning
        self.t_type = t_type   # 目标变量类型
        
        if type(spec_values) == dict:  # Parsing special values to dict for cont variables
            self.spec_values = {}
            for k, v in spec_values.items():
                if v.startswith('d_'):
                    self.spec_values[k] = v
                else:
                    self.spec_values[k] = 'd_' + v
        else:
            if spec_values is None:
                self.spec_values = {}
            else:
                self.spec_values = {i: 'd_' + str(i) for i in spec_values}

    def fit(self, x, y, max_buckets=100):
        """
        ----------------------------------------------------------------------
        Fit WoE transformation
        ----------------------------------------------------------------------
        :param x: pd.Series, continuous or discrete predictor
        :param y: pd.Series, binary target variable
        ----------------------------------------------------------------------
        :return: WoE class
        ----------------------------------------------------------------------
        """
        # Data quality checks
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if not x.size == y.size:
            raise Exception("X size don't match Y size")
            
        # Calc total good bad ratio in the sample
        t_bad = np.sum(y)
        if t_bad == 0 or t_bad == y.size:
            raise ValueError("There should be BAD and GOOD observations in the sample, sample size is {}".format(y.size))
        if np.max(y) > 1 or np.min(y) < 0:
            raise ValueError("Y range should be between 0 and 1")
            
        # 自变量名
        self.v_name = x.name
        
        # setting discrete values as special values
        if self.v_type == 'd':
            sp_values = {i: 'd_' + str(i) for i in x.unique()} # 离散变量
            if len(sp_values) > max_buckets:
                raise type("DiscreteVarOverFlowError", (Exception,),
                           {"args": ('Discrete variable with too many unique values (more than {})'.format(max_buckets),)})
            else:
                if self.spec_values:
                    sp_values.update(self.spec_values)
                self.spec_values = sp_values
                
        # Make data frame for calculations
        df = pd.DataFrame({"X": x, "Y": y, 'order': np.arange(x.size)})
        
        # Separating NaN and Special values
        df_sp_values, df_cont = self._split_sample(df)
        
        # labeling data
        df_cont, c_bins = self._cont_labels(df_cont)
        df_sp_values, d_bins = self._disc_labels(df_sp_values)
        
        # getting continuous and discrete values together
        self.df = df_sp_values.append(df_cont)
        self.bins = d_bins.append(c_bins)
        
        # calculating woe and other statistics
        self._calc_stat()
        
        # sorting appropriately for further cutting in transform method
        self.bins['feature'] = x.name
        self.bins.sort_values('bins', inplace=True)
        
        # 分箱区间
        range_list = list(self.bins['bins'])
        label_list = list(self.bins['labels'])
        if not str(label_list[0]).startswith('d_'): 
            # 连续变量
            if math.isnan(range_list[-1]):
                # 有空分箱
                lower = range_list[0:-1] + [math.nan]
                upper = range_list[1:-1] + [math.inf, math.nan]
            else:
                # 无空分箱
                lower = range_list
                upper = range_list[1:] + [math.inf]
        else:
            lower = range_list
            upper = range_list
            
        self.bins['min_score'] = lower
        self.bins['max_score'] = upper
        
        self.bins = self.bins[['feature','labels','bins','min_score','max_score','obs','bad','good',
                               'bad_rate','good_rate','odds_good','woe', 'bin_iv', 'iv']].reset_index(drop=1)
        
        # returning to original observation order
        self.df.sort_values('order', inplace=True)
        self.df.set_index(x.index, inplace=True)
        
        return self

    def fit_transform(self, x, y):
        """
        ----------------------------------------------------------------------
        功能：Fit WoE transformation
        ----------------------------------------------------------------------
        :param x: pd.Series, continuous or discrete predictor
        :param y: pd.Series, binary target variable
        ----------------------------------------------------------------------
        :return: pd.Series, WoE transformed variable
        ----------------------------------------------------------------------
        示例：
        >>> woe1 = woe.fit(df['age'], df[target])
        ----------------------------------------------------------------------
        """
        self.fit(x, y)
        return self.df['woe']

    def _split_sample(self, df):
        """
        ----------------------------------------------------------------------
        功能：样本划分
        ----------------------------------------------------------------------
        :return df_sp_values: 特殊变量
        :return df_cont: 连续变量
        ----------------------------------------------------------------------
        """
        if self.v_type == 'd':
            return df, None
        sp_values_flag = df['X'].isin(self.spec_values.keys()).values | df['X'].isnull().values
        df_sp_values = df[sp_values_flag].copy()
        
        df_cont = df[np.logical_not(sp_values_flag)].copy()
        
        return df_sp_values, df_cont

    def _disc_labels(self, df):
        """
        ----------------------------------------------------------------------
        功能：离散变量分箱标签
        ----------------------------------------------------------------------
        :return df: 
        :return d_bins: dataframe, 离散变量分箱标签
        ----------------------------------------------------------------------
        """
        df['labels'] = df['X'].apply(
            lambda x: self.spec_values[x] if x in self.spec_values.keys() else 'd_' + str(x))
        d_bins = pd.DataFrame({"bins": df['X'].unique()})
        d_bins['labels'] = d_bins['bins'].apply(
            lambda x: self.spec_values[x] if x in self.spec_values.keys() else 'd_' + str(x))
        
        return df, d_bins

    def _cont_labels(self, df):
        """
        ----------------------------------------------------------------------
        功能：连续变量分箱标签
        ----------------------------------------------------------------------
        :return df: 
        :return c_bins: dataframe, 连续变量分箱标签
        ----------------------------------------------------------------------
        """
        # check whether there is a continuous part
        if df is None:
            return None, None
        
        # Max buckets num calc(最大分箱数 = MIN(qnt_num, 去重后的样本数 / min_block_size))
        self.qnt_num = int(np.minimum( \
                           df['X'].unique().size / self._min_block_size, \
                           self.__qnt_num)) + 1
        
        # cuts - label num for each observation, bins - quartile thresholds
        bins = None
        cuts = None
        if self._predefined_bins is None:
            try: # 等频（深）分箱 —— pd.qcut
                cuts, bins = pd.qcut(df["X"], self.qnt_num, retbins=True, labels=False, duplicates='drop')
            except ValueError as ex:
                if ex.args[0].startswith('Bin edges must be unique'):
                    ex.args = ('Please reduce number of bins or encode frequent items as special values',) + ex.args
                    raise
            bins = np.append((-float("inf"),), bins[1:-1])
        else:
            bins = self._predefined_bins
            if bins[0] != float("-Inf"):
                bins = np.append((-float("inf"),), bins)
                # 等距分箱 —— pd.cut
            cuts = pd.cut(df['X'], bins=np.append(bins, (float("inf"),)),
                          labels=np.arange(len(bins)).astype(str))
        df["labels"] = cuts.astype(str)
        c_bins = pd.DataFrame({"bins": bins, "labels": np.arange(len(bins)).astype(str)})
        
        return df, c_bins

    def _calc_stat(self):
        """
        ----------------------------------------------------------------------
        功能：计算WOE(calculating WoE)
        ----------------------------------------------------------------------
        :param self: 
        ----------------------------------------------------------------------
        :return: None
        ----------------------------------------------------------------------
        """
        col_names = {'count_nonzero': 'bad', 'size': 'obs'}
        stat = self.df.groupby("labels")['Y'].agg([np.mean, np.count_nonzero, np.size]).rename(columns=col_names).copy()
        if self.t_type != 'b':
            stat['bad'] = stat['mean'] * stat['obs']
        stat['good'] = stat['obs'] - stat['bad']
        # total goods & bads
        t_good = np.maximum(stat['good'].sum(), 0.5)
        t_bad = np.maximum(stat['bad'].sum(), 0.5)
        
        stat['good_rate'] = stat['good'] / stat['obs']
        stat['bad_rate'] = stat['bad'] / stat['obs']
        
        # woe = ln(分箱内good客户数 / 分箱内bad客户数) - ln(总的good客户数 / 总的bad客户数)
        stat['woe'] = stat.apply(self._bucket_woe, axis=1) - np.log(t_good / t_bad) 
        stat['odds_good'] = stat['good'] / stat['bad']  # 好坏比
        stat['bin_iv'] = (stat['good'] / t_good - stat['bad'] / t_bad) * stat['woe']
        stat['iv'] = sum(stat['bin_iv'])
        
        iv_stat = (stat['good'] / t_good - stat['bad'] / t_bad) * stat['woe']
        self.iv = iv_stat.sum()
        
        # adding stat data to bins
        self.bins = pd.merge(stat, self.bins, left_index=True, right_on=['labels'])
        label_woe = self.bins[['woe', 'labels']].drop_duplicates()
        self.df = pd.merge(self.df, label_woe, left_on=['labels'], right_on=['labels'])     

    def transform(self, x, manual_woe=None, replace_missing=None):
        """
        ----------------------------------------------------------------------
        Transforms input variable according to previously fitted rule
        ----------------------------------------------------------------------
        :param x: input variable
        :param manual_woe: one can change fitted woe with manual values by providing dict {label: new_woe_value}
        :param replace_missing: replace woe for labels not observable in traning dataset by this value
        ----------------------------------------------------------------------
        :return: DataFrame with transformed with original and transformed variables
        ----------------------------------------------------------------------
        """
        if not isinstance(x, pd.Series):
            raise TypeError("入参x类型不正确, 应为pandas.Series")
        if self.bins is None:
            raise Exception('Fit the model first, please')
            
        df = pd.DataFrame({"X": x, 'order': np.arange(x.size)})
        
        # splitting to discrete and continous pars
        df_sp_values, df_cont = self._split_sample(df)
        
        # Replacing original with manual woe
        tr_bins = self.bins[['woe', 'labels']].copy()
        if manual_woe:
            if not type(manual_woe) == dict:
                TypeError("manual_woe should be dict")
            else:
                for key in manual_woe:
                    tr_bins['woe'].mask(tr_bins['labels'] == key, manual_woe[key], inplace=True)

        if replace_missing is not None:
            tr_bins = tr_bins.append({'labels': 'd__transform_missing_replacement__', 
                                      'woe': replace_missing}, ignore_index=True)

        # function checks existence of special values, raises error if sp do not exist in training set
        def get_sp_label(x_):
            if x_ in self.spec_values.keys():
                return self.spec_values[x_]
            else:
                str_x = 'd_' + str(x_)
                if str_x in list(self.bins['labels']):
                    return str_x
                else:
                    if replace_missing is not None:
                        return 'd__transform_missing_replacement__'
                    else:
                        raise ValueError('Value {} does not exist in the training set'.format(str_x))

        # assigning labels to discrete part
        df_sp_values['labels'] = df_sp_values["X"].apply(get_sp_label)
        
        # assigning labels to continuous part
        c_bins = self.__get_cont_bins()
        if self.v_type != 'd':
            cuts = pd.cut(df_cont["X"], bins=np.append(c_bins["bins"], (float("inf"),)), labels=c_bins["labels"])
            df_cont['labels'] = cuts.astype(str)
            
        # Joining continuous and discrete parts
        df = df_sp_values.append(df_cont)
        
        # assigning woe
        df = pd.merge(df, tr_bins[['woe', 'labels']].drop_duplicates(), left_on=['labels'], right_on=['labels'])
        
        # returning to original observation order
        df.sort_values('order', inplace=True)
        
        return df.set_index(x.index)

    def __get_cont_bins(self):
        """
        ----------------------------------------------------------------------
        功能：返回连续变量的分箱（剔除离散分箱, 包括d_nan）
        ----------------------------------------------------------------------
        :return: return continous part of self.bins
        ----------------------------------------------------------------------
        """
        return self.bins[self.bins['labels'].apply(lambda z: not z.startswith('d_'))]
    
    def _is_woe_monotonic(self):
        """
        ----------------------------------------------------------------------
        功能：判断woe是否单调
        ----------------------------------------------------------------------
        :return: bool, 1=单调, 0=不单调
        ----------------------------------------------------------------------
        """    
        cont_bins = self.__get_cont_bins()
         
        is_desc_monotonic = True # 递减性单调
        for i, w in enumerate(cont_bins[1:]['woe']):
            cur_bin_woe = cont_bins.iloc[i].loc['woe']
            next_bin_woe = w
            if operator.lt(cur_bin_woe, next_bin_woe):
                is_desc_monotonic = False
                break
                
        is_asc_monotonic = True  # 递增性单调
        for i, w in enumerate(cont_bins[1:]['woe']):
            cur_bin_woe = cont_bins.iloc[i].loc['woe']
            next_bin_woe = w
            if operator.gt(cur_bin_woe, next_bin_woe):
                is_asc_monotonic = False
                break       
                
        return np.logical_or(is_desc_monotonic, is_asc_monotonic)
    
    def disc_merge(self):
        """
        ----------------------------------------------------------------------
        功能：离散分箱合并调整
        策略是：将原始离散分箱里样本数小于min_block_size的小箱子合并在一起，作为单独一个分箱。
              另外的箱子表示样本量足够，根据woe值降序后，根据woe值等频均分 or 等距分箱， 把原始小分箱合并为几个大箱子
        ----------------------------------------------------------------------
        :return: var_bin_map: dict, 原始离散变量分箱与合并后的分箱映射
        ----------------------------------------------------------------------
        """    
        disc_var_bin_map = {}
        
        woe_df1 = self.bins[self.bins['obs'] <  self._min_block_size]
        woe_df2 = self.bins[self.bins['obs'] >= self._min_block_size].reset_index(drop=1)
        woe_df2 = woe_df2.sort_values(by=['woe'], ascending=0).reset_index(drop=1)
        cuts, bins = pd.qcut(woe_df2["woe"], self.__qnt_num, retbins=True, labels=False, duplicates='drop')
        woe_df2['new_bin'] = cuts

        for idx, var in enumerate(woe_df2['bins']):
            disc_var_bin_map[var] = 'd_{}'.format(cuts[idx])

        for idx, var in enumerate(woe_df1['bins']):
            if var != 'd_nan':
                disc_var_bin_map[var] = 'd_{}'.format(max(cuts)+1)
            else:
                disc_var_bin_map[np.nan] = 'd_nan'
            
        return disc_var_bin_map
        
    def merge(self, label1, label2=None):
        """
        ----------------------------------------------------------------------
        功能：(两个)分箱合并. Merge of buckets with given labels
        In case of discrete variable, both labels should be provided. As the result labels will be merged to one bucket.
        In case of continous variable, only label1 should be provided. It will be merged with the next label.
        ----------------------------------------------------------------------
        :param label1: first label to merge
        :param label2: second label to merge
        ----------------------------------------------------------------------
        :return:
        ----------------------------------------------------------------------
        """
        spec_values = self.spec_values.copy()
        c_bins = self.__get_cont_bins().copy()
        if label2 is None and not label1.startswith('d_'): 
            # removing bucket for continuous variable
            c_bins = c_bins[c_bins['labels'] != label1]
        else:
            if not (label1.startswith('d_') and label2.startswith('d_')):
                raise Exception('分箱label必须都是离散变量！')
            for i in self.bins[self.bins['labels'] == label1]['bins']:
                spec_values[i] = label1 + '_' + label2
            bin2 = self.bins[self.bins['labels'] == label2]['bins'].iloc[0]
            spec_values[bin2] = label1 + '_' + label2
            
        new_woe = WoE(self.__qnt_num, self._min_block_size, spec_values, self.v_type, c_bins['bins'], self.t_type)
        
        return new_woe.fit(self.df['X'], self.df['Y'])

    def force_monotonic(self, direct_hypothesis=1):
        """
        ----------------------------------------------------------------------
        功能：连续变量单调性调整。
        分箱调整策略：不断合并下一个分箱，直到满足单调性
        Makes transformation monotonic if possible, given relationship hypothesis 
        (otherwise - MonotonicConstraintError exception)
        ----------------------------------------------------------------------
        :direct: int, 默认值为1, 此时自变量取值越大, woe(好坏比)值越大; （正相关）
                      当取值为0时, 自变量取值越大, woe(好坏比)值越小; （负相关）
                      direct (1) = relationship between predictor and target variable
        ----------------------------------------------------------------------
        :return: new WoE object with monotonic transformation
        ----------------------------------------------------------------------
        """
        if direct_hypothesis == 1:
            op_func = operator.gt # Same as a>b. (目的让woe升序)
        else:
            op_func = operator.lt # Same as a<b. (目的让woe降序)
            
        cont_bins = self.__get_cont_bins()
        new_woe = self
        for i, w in enumerate(cont_bins[1:]['woe']):
            cur_bin_woe = cont_bins.iloc[i].loc['woe']
            next_bin_woe = w
            if op_func(cur_bin_woe, next_bin_woe):
                if cont_bins.shape[0] <= 2:
              
                    raise type("MonotonicConstraintError", (Exception,),
                               {"args": ('分箱数只有{}个, 无法再进行调整'.format(cont_bins.shape[0]),)})
                else:
                    new_woe = self.merge(cont_bins.iloc[i+1].loc['labels'])
                    new_woe = new_woe.force_monotonic(direct_hypothesis)
                    return new_woe
                
        return new_woe
    
    def optimize(self, criterion=None, fix_depth=None, max_depth=None, cv=3, scoring=None, min_samples_leaf=None):
        """
        ----------------------------------------------------------------------
        功能：基于决策树的WOE分箱优化. WoE bucketing optimization (continuous variables only)
        ----------------------------------------------------------------------
        :param criterion: binary tree split criteria
        :param fix_depth: use tree of a fixed depth (2^fix_depth buckets)
        :param max_depth: maximum tree depth for a optimum cross-validation search
        :param cv: number of cv buckets
        :param scoring: scorer for cross_val_score
        :param min_samples_leaf: minimum number of observations in each of optimized buckets
        ----------------------------------------------------------------------
        :return: WoE class with optimized continuous variable split
        ----------------------------------------------------------------------
        """
        if self.t_type == 'b':
            tree_type = tree.DecisionTreeClassifier
        else:
            tree_type = tree.DecisionTreeRegressor
            
        m_depth = int(np.log2(self.__qnt_num)) + 1 if max_depth is None else max_depth
        cont = self.df['labels'].apply(lambda z: not z.startswith('d_'))
        x_train = np.array(self.df[cont]['X'])
        y_train = np.array(self.df[cont]['Y'])
        x_train = x_train.reshape(x_train.shape[0], 1)
        if not min_samples_leaf:
            min_samples_leaf = self._min_block_size
        start = 1
        cv_scores = []
        if fix_depth is None:
            for i in range(start, m_depth):
                if criterion is None:
                    d_tree = tree_type(max_depth=i, min_samples_leaf=min_samples_leaf)
                else:
                    d_tree = tree_type(criterion=criterion, max_depth=i, min_samples_leaf=min_samples_leaf)
                scores = cross_val_score(d_tree, x_train, y_train, cv=cv, scoring=scoring)
                cv_scores.append(scores.mean())
            best = np.argmax(cv_scores) + start
        else:
            best = fix_depth
        final_tree = tree_type(max_depth=best, min_samples_leaf=min_samples_leaf)
        final_tree.fit(x_train, y_train)
        opt_bins = final_tree.tree_.threshold[final_tree.tree_.feature >= 0]
        opt_bins = np.sort(opt_bins)
        new_woe = WoE(self.__qnt_num, self._min_block_size, self.spec_values, self.v_type, opt_bins, self.t_type)
        
        return new_woe.fit(self.df['X'], self.df['Y'])

    def plot(self, sort_values=True, labels=False):
        """
        ----------------------------------------------------------------------
        功能：Plot WoE transformation and default rates
        ----------------------------------------------------------------------
        :param sort_values: bool, 是否根据woe值来调整分箱顺序。whether to sort discrete variables by woe, continuous by labels
        :param labels: plot labels or intervals for continuous buckets
        ----------------------------------------------------------------------
        :return: plotting object
        ----------------------------------------------------------------------
        """
        woe_fig = plt.figure()
        plt.title('Number of Observations and WoE per bucket')
        ax = woe_fig.add_subplot(111)
        ax.set_xlabel('Var = {}'.format(self.v_name))
        ax.set_ylabel('Observations')
        plot_data = self.bins[['labels', 'woe', 'obs', 'bins']].copy().drop_duplicates()

        # creating plot labels
        if self.v_type != 'd':
            cont_labels = plot_data['labels'].apply(lambda z: not z.startswith('d_'))
            plot_data['plot_bins'] = plot_data['bins'].apply(lambda x: '{:0.2g}'.format(x))
            temp_data = plot_data[cont_labels].copy()

            right_bound = temp_data['plot_bins'].iloc[1:].append(pd.Series(['Inf']))
            temp_data['plot_bins'] = temp_data['plot_bins'].add(' : ').add(list(right_bound))

            plot_data = temp_data.append(plot_data[~cont_labels])
            cont_labels = plot_data['labels'].apply(lambda z: not z.startswith('d_'))
            plot_data['plot_bins'] = np.where(cont_labels, plot_data['plot_bins'], plot_data['labels'])
        else:
            plot_data['plot_bins'] = plot_data['bins']

        # sorting
        if sort_values:
            if self.v_type != 'd':
                cont_labels = plot_data['labels'].apply(lambda z: not z.startswith('d_'))
                temp_data = plot_data[cont_labels].sort_values('bins')
                plot_data = temp_data.append(plot_data[~cont_labels].sort_values('labels'))
            else:
                plot_data.sort_values('woe', inplace=True)

        # start plotting
        index = np.arange(plot_data.shape[0])
        bar_width = 0.8
        plt.xticks(index, plot_data['labels'] if labels else plot_data['plot_bins'])
        plt.bar(index, plot_data['obs'], bar_width, color='b', label='Observations')
        ax2 = ax.twinx()
        ax2.set_ylabel('Weight of Evidence')
        ax2.plot(index, plot_data['woe'], 'bo-', linewidth=4.0, color='r', label='WoE')
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        plt.legend(handles, labels)
        woe_fig.autofmt_xdate()
        
        return woe_fig

    @staticmethod
    def _bucket_woe(x):
        """
        ----------------------------------------------------------------------
        功能：计算分箱内的Log(odds), odds = goods / bads
             当goods或bads数量为0时，置为0.5
        ----------------------------------------------------------------------
        :return log(odds): float
        ----------------------------------------------------------------------
        """
        bin_bad = x['bad']
        bin_good = x['good']
        bin_bad = 0.5 if bin_bad == 0 else bin_bad
        bin_good = 0.5 if bin_good == 0 else bin_good
        bin_odds = bin_good / bin_bad
        
        return np.log(bin_odds)  
    
    @staticmethod
    def _help():
        """
        使用方法：
        >> import weight_of_evidence as pyWoE
        >> woe = pyWoE.WoE(qnt_num=3, min_block_size=2, spec_values=None, v_type='c', bins=None, t_type='b')

        >> woe1 = woe.fit(df['age'], df[target])
        >> woe1.bins
            labels	bins	min_score	max_score	obs	bad	good	bad_rate	good_rate	odds_good	woe
        0	0	-inf	-inf	20.125000	179	82	97	0.458101	0.541899	1.182927	-0.305296
        1	1	20.125000	20.125000	28.000000	183	66	117	0.360656	0.639344	1.772727	0.099231
        2	2	28.000000	28.000000	38.000000	175	76	99	0.434286	0.565714	1.302632	-0.208901
        3	3	38.000000	38.000000	inf	177	66	111	0.372881	0.627119	1.681818	0.046588
        4	d_nan	NaN	NaN	NaN	177	52	125	0.293785	0.706215	2.403846	0.403782

        >> woe1.iv
        >> 0.08758253960602345

        >> woe2 = woe1.force_monotonic()
        >> woe2.bins
            labels	bins	min_score	max_score	obs	bad	good	bad_rate	good_rate	odds_good	woe
        0	0	-inf	-inf	20.125000	179	82	97	0.458101	0.541899	1.182927	-0.305296
        1	1	20.125000	20.125000	38.000000	358	142	216	0.396648	0.603352	1.521127	-0.053836
        2	2	38.000000	38.000000	inf	177	66	111	0.372881	0.627119	1.681818	0.046588
        3	d_nan	NaN	NaN	NaN	177	52	125	0.293785	0.706215	2.403846	0.403782

        >> df.loc[:, 'age_woe'] = woe2.transform(df['age'])['woe']
        >> df[['age', 'age_woe']].head()
        >>
            age	age_woe
        0	22.0	-0.053836
        1	38.0	-0.053836
        2	26.0	-0.053836
        3	35.0	-0.053836
        4	35.0	-0.053836

        >> woe_monotonic, woe_bins = WOE_Transform(Feature, data=develop_data_ins, TargetVar=target_var)
        >> woe2_bins = list(woe2.bins['bins'].values)

        # step 2: 查看该WOE分箱模式在['INS', 'OOS', 'OOT1']上的稳定性
        >> woe_transfer = pyWoE.WoE(bins=woe_bins[1:], v_type='c', t_type='b')
        >> bins_map = woe_monotonic.bins.copy()
        >> bins_map.loc[:, 'group'] = 'INS'
        >> for group_i in ['OOS', 'OOT1', 'OOT2']:
                print('-' * 50)
                df = develop_data[develop_data['group'] == group_i]
                woe_transfer.fit(pd.Series(df[feature]), pd.Series(df[target_var]))
                bins_map_temp = woe_transfer.bins.copy()
                bins_map_temp.loc[:, 'group'] = group_i
                bins_map = pd.concat([bins_map, bins_map_temp])
                fig = woe_transfer.plot()
                plt.show(fig)
        >> bins_map.loc[:, 'feature'] = FeatureWoe
        ----------------------------------------------------------------------
        知识：
        1. WOE（Weight of Evidence）——证据权重
        可以将logistic回归模型转化为标准评分卡格式.
        WOE是对原始自变量的一种编码形式，要对一个变量进行WOE编码，需要首先把这个变量进行分组处理（也叫离散化、分箱）。
        1) 当定义WOE值越大，P(Good)越大时:
        WOE1 = ln(odds_good)
             = ln(好客户占比 / 坏客户占比) 
             = ln( (分箱内good客户数 / 总的good客户数) / (分箱内bad客户数 / 总的bad客户数) )
             = ln( (分箱内good客户数 / 分箱内bad客户数) / (总的good客户数 / 总的bad客户数) )
             = ln(分箱内good客户数 / 分箱内bad客户数) - ln(总的good客户数 / 总的bad客户数)

        2) 当定义WOE值越大，P(Bad)越大时:
        WOE2 = ln(odds_bad)
             = ln(坏客户占比 / 好客户占比) 
             = ln( (分箱内bad客户数 / 总的bad客户数) / (分箱内good客户数 / 总的good客户数) )
             = ln( (分箱内bad客户数 / 分箱内good客户数) / (总的bad客户数 / 总的good客户数) )
             = ln(分箱内bad客户数 / 分箱内good客户数) - ln(总的bad客户数 / 总的good客户数)

        因此：WOE1 + WOE2 = 0
        在本代码中，WOE定义为第一种.

        2. IV(Information Value)——信息量
        用来衡量自变量的预测能力
        IV = SUM( (好客户占比 / 坏客户占比) * ln(好客户占比 / 坏客户占比) )
        其中：
        好客户占比 = 分箱内好客户数 / 总的好客户数
        坏客户占比 = 分箱内坏客户数 / 总的坏客户数
        -----------------------------------
        IV           解释
        < 0.03       无预测能力
        0.03～0.09   低
        0.10～0.29   中
        0.30～0.49   高
        > 0.50       极高
        -----------------------------------
        """
        pass
    
    
