# -*- coding:utf-8 -*-
__author__ = 'fenghaijie / hjfeng0630@qq.com'

from datetime import datetime
import statsmodels.api as sm
import pandas as pd
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

"""
模块描述：逻辑斯特回归（Logistic Regression, LR）
功能包括：
1. statsmodels.lr:  调用statsmodels模块中的lr模型进行回归
2. sklearn_lr:      调用sklean模块中的lr模型进行回归，可调整权重

函数用法：
1) 实例化对象: lr = Logistic(X=df[feat_cols], y=df[target])
2) 模型拟合: logit_model, logit_result, logit_result_0, var_list = lr.logistic_fit(stepwise='BS', sls=0.05)
3) 模型拟合报表：desc, params, evaluate, quality = lr.logistic_output(logit_model, logit_result, logit_result_0)

结果示例：

desc:
--------------------------------------------------------------------------------------------
模型                               二元logistic模型
使用的观测个数                                   714
含缺失值观测个数                                  177
总观测个数                                     891
自变量         [Pclass, Age, SibSp, Fare, Parch]
因变量                                  Survived
方法                                     最大似然估计
日期时间               2019-05-30 20:28:26.718789
--------------------------------------------------------------------------------------------

params:
--------------------------------------------------------------------------------------------
	参数估计	标准误差	z值	wald卡方	p值	置信下界	置信上界
const	3.401026	0.505176	6.732357	45.324625	1.669370e-11	2.410899	4.391153
Pclass	-1.153008	0.145943	-7.900382	62.416041	2.780492e-15	-1.439051	-0.866964
Age	-0.044566	0.007210	-6.181269	38.208091	6.358821e-10	-0.058697	-0.030435
SibSp	-0.292273	0.106079	-2.755238	7.591334	5.864954e-03	-0.500183	-0.084362
Parch	0.247881	0.109075	2.272570	5.164574	2.305211e-02	0.034097	0.461664
Fare	0.003294	0.002537	1.298759	1.686776	1.940265e-01	-0.001677	0.008266
--------------------------------------------------------------------------------------------

quality
--------------------------------------------------------------------------------------------
似然比      1.493370e+02
自由度      5.000000e+00
似然比p值    1.847754e-30
--------------------------------------------------------------------------------------------
"""
class Logistic:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def model_fit(self, X, y, constant=True):
        """
        功能: 模型拟合
        -------------------------------
        :param X: pandas dataframe, 样本特征
        :param y: pandas series, 样本目标变量
        :param constant: bool，添加截距intercept
        -------------------------------
        :return model: sm model obj
        :return results: fit result with x and intercept
        :return results_0: fit result with only with intercept
        -------------------------------
        知识:
        import statsmodels.api as sm
        其中, sm.Logit()参数说明如下:
        Parameters
        ----------
        endog : array-like
            1-d endogenous response variable. The dependent variable.
        exog : array-like
            A nobs x k array where `nobs` is the number of observations and `k`
            is the number of regressors. An intercept is not included by default
            and should be added by the user. See
            :func:`statsmodels.tools.add_constant`.
        missing : str
            Available options are 'none', 'drop', and 'raise'. If 'none', no nan
            checking is done. If 'drop', any observations with nans are dropped.
            If 'raise', an error is raised. Default is 'none.'
        ----
        # 加权重的一种实现：
        import statsmodels.api as sm
        logmodel = sm.GLM(trainingdata[['Successes', 'Failures']], 
                          trainingdata[['const', 'A', 'B', 'C', 'D']], 
                          family=sm.families.Binomial(sm.families.links.logit)).fit()
        """
        if constant:
            # 添加截距项 = 1
            X = sm.add_constant(X)

        model = sm.Logit(y, X, missing='drop')
        results = model.fit()

        model_0 = sm.Logit(y, X.const, missing='drop')
        results_0 = model_0.fit()

        return model, results, results_0

    def model_describe(self, model, results):
        """
        return information of model
        ----------------------------------
        :param model: sm model
        :param results: fit result with x and intercept
        ----------------------------------
        :return pandas series
        """
        model_desc = {
            "模型":           "二元logistic模型",
            "使用的观测个数":   results.nobs,
            "含缺失值观测个数": self.X.shape[0] - results.nobs,
            "总观测个数":      self.X.shape[0],
            "自变量":         list(self.X.columns),
            "因变量":         self.y.name,
            "方法":           "最大似然估计",
            "日期时间":        datetime.now()
        }
        return pd.Series(model_desc)

    def param_estimate(self, results):
        """
        功能: LR模型参数估计(return params estimate from model fit results)
        ----------------------------------
        :param results: model fit result
        ----------------------------------
        :return pandas dataframe
        """
        param_df = pd.concat([
            results.params,
            results.bse,
            results.tvalues,
            (results.params / results.bse) ** 2,
            results.pvalues,
            results.conf_int()
        ], axis=1)
        param_df.columns = [u'参数估计', u'标准误差', u'z值', u'wald卡方', u'p值', u'置信下界', u'置信上界']

        return param_df

    def fit_evaluation(self, results_1, results_0):
        """
        功能：模型拟合程度指标评估(metrics of model evaluation)
        ------------------------------------
        :param results_1: fit result with x and intercept
        :param results_0: fit result with only with intercept
        ------------------------------------
        :return rlt: pandas dataframe
        :return Series(rsq):
        """
        indx = ['aic', 'bic', '-2*logL']

        S0 = pd.Series([results_0.aic, results_0.bic, -2 * results_0.llf], index=indx)
        S0.name = '仅含截距'

        S1 = pd.Series([results_1.aic, results_1.bic, -2 * results_1.llf], index=indx)
        S1.name = '包含截距和协变量'

        rlt = pd.concat([S0, S1], axis=1)
        rsq = {"mcfadden R^2": results_1.prsquared}

        return rlt, pd.Series(rsq)

    def model_quality(self, results):
        """
        功能: 模型质量评估(return metrics of model Quality)
        ------------------------------------
        :param results: model fitting results
       ------------------------------------
        :return Series(rsq): pandas series
        """
        rlt = {
            "似然比": results.llr,
            "自由度": results.df_model,
            "似然比p值": results.llr_pvalue
        }
        return pd.Series(rlt)

    def confuse_matrix(self, results):
        """
        功能：混淆矩阵(create confuse matrix)
        ----------------------------------
        :param results: model fitting results
        ----------------------------------
        :return confu_mat: confuse matrix
        """
        confuse_mat = pd.DataFrame(results.pred_table())
        confuse_mat.index = ['real_0', 'real_1']
        confuse_mat.columns = ['pred_0', 'pred_1']

        return confuse_mat

    def cov_matrix(self, results, normalized=False):
        """
        -----------------------------------------
        功能: 协方差矩阵(cov_matrix)
        -----------------------------------------
        :param results: model fitting results
        -----------------------------------------
        :return Series(rsq): pandas series
        -----------------------------------------
        """
        if normalized:
            res_df = results.normalized_cov_params
        else:
            res_df = results.cov_params()
        return res_df
    
    """
    def residual(self, results):

        return results.resid_generalized

    def standard_residual(self, results):

        return results.resid_pearson

    def dev_residual(self, results):

        return results.resid_response
    """

    def forward_selection(self, X, y):
        """
        -----------------------------------------
        功能: 前向逐步回归(Linear model designed by forward selection.)
        -----------------------------------------
        :param data : pandas DataFrame with all possible predictors and response
        :param response: string, name of response column in data
        -----------------------------------------
        :return model: an "optimal" fitted statsmodels linear model
               with an intercept
               selected by forward selection
               evaluated by adjusted R-squared
        -----------------------------------------
        """
        import statsmodels.formula.api as smf

        input_df = pd.concat([X, y], axis=1)
        target_var = y.name
        remaining = set(input_df.columns)
        remaining.remove(target_var)

        selected_var_list = []
        current_score, best_new_score = 0.0, 0.0
        while remaining and current_score == best_new_score:
            scores_with_candidates = []
            for candidate in remaining:
                formula = "{} ~ {} + 1".format(target_var, ' + '.join(selected_var_list + [candidate]))
                mod = smf.logit(formula, input_df).fit()
                score = mod.prsquared
                scores_with_candidates.append((score, candidate))
            
            scores_with_candidates.sort(reverse=False)
            best_new_score, best_candidate = scores_with_candidates.pop()

            if current_score < best_new_score:
                remaining.remove(best_candidate)
                selected_var_list.append(best_candidate)
                current_score = best_new_score

        return selected_var_list

    def backward_selection(self, X, y, sls=0.05):
        """
        -----------------------------------------
        功能: 后向逐步回归(Linear model designed by backward selection.)
        -----------------------------------------
        :param X: pandas DataFrame with all possible predictors
        :param y: pandas Series with response
        :param sls: measure for drop variable
        -----------------------------------------
        :return var_list
        -----------------------------------------
        """
        import statsmodels.formula.api as smf

        data = pd.concat([X, y], axis=1)
        var_list = X.columns
        target_var = y.name

        while True:
            formula = "{} ~ {} + 1".format(target_var, ' + '.join(var_list))
            mod = smf.logit(formula, data).fit()

            p_list = mod.pvalues.sort_values()  # 按p值升序排列
            if p_list[-1] > sls:
                var = p_list.index[-1]          # 提取p_list中最后一个index
                var_list = var_list.drop(var)   # var_list中删除
            else:
                break

        return var_list

    def logistic_fit(self, constant=True, stepwise='BS', sls=0.05):
        """
        -----------------------------------------
        功能: 模型拟合
        -----------------------------------------
        :param X: pandas dataframe, 样本特征
        :param y: pandas series, 样本目标变量
        :param constant：bool, True means add constant
        :param stepwise: str, variable select, "BS" is backward, "FS" is forward
        :param sls: float, threshold for variable select metric(该阈值与p值进行比较. 注意，前向选择函数没有用到该阈值)
        -----------------------------------------
        :return logit_instance: instance of logit model
        :return logit_model: sm model object of logit model
        :return logit_result: fit results of logit model
        :return logit_result_0: fit results of logit model(only with constant)
        -----------------------------------------
        """
        # step1：特征筛选
        feat_df = None
        var_list = None

        if self.X.shape[1] == 1:
            raise Exception('入模变量只有1个!')

        if stepwise == 'FS':   # 前向选择
            var_list = self.forward_selection(self.X, self.y)
            feat_df = self.X.loc[:, var_list]
        elif stepwise == 'BS': # 后向选择
            var_list = self.backward_selection(self.X, self.y, sls=sls)
            feat_df = self.X.loc[:, var_list]
        else:
            var_list = self.X.columns
            feat_df = self.X

        # step2：模型拟合
        logit_model, logit_result, logit_result_0 = self.model_fit(X=feat_df, y=self.y, constant=constant)

        return logit_model, logit_result, logit_result_0, var_list

    def logistic_output(self, logit_model, logit_result, logit_result_0):
        """
        功能：generate logistic model output
        -------------------------------------------------------
        :param logit_model: sm model object of logit model
        :param logit_result: fit results of logit model
        :param logit_result_0: fit results of logit model(only with constant)
        -------------------------------------------------------
        :return desc: describe of model
        :return params: estimated results
        :return evaluate: evaluate for model
        :return quality: model quality metric
        """
        desc = self.model_describe(logit_model, logit_result)
        params = self.param_estimate(logit_result)
        evaluate = self.fit_evaluation(logit_result, logit_result_0)
        quality = self.model_quality(logit_result)

        return desc, params, evaluate, quality
