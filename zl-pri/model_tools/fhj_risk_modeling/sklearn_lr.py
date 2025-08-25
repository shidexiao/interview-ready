# # -*- coding:utf-8 -*-
# __author__ = 'fenghaijie'


# class LinearRegression(linear_model.LinearRegression):

#     def __init__(self,*args,**kwargs):
#         # *args is the list of arguments that might go into the LinearRegression object
#         # that we don't know about and don't want to have to deal with. Similarly, **kwargs
#         # is a dictionary of key words and values that might also need to go into the orginal
#         # LinearRegression object. We put *args and **kwargs so that we don't have to look
#         # these up and write them down explicitly here. Nice and easy.

#         if not "fit_intercept" in kwargs:
#             kwargs['fit_intercept'] = False

#         super(LinearRegression,self).__init__(*args,**kwargs)

#     # Adding in t-statistics for the coefficients.
#     def fit(self,x,y):
#         # This takes in numpy arrays (not matrices). Also assumes you are leaving out the column
#         # of constants.

#         # Not totally sure what 'super' does here and why you redefine self...
#         self = super(LinearRegression, self).fit(x,y)
#         n, k = x.shape
#         yHat = np.matrix(self.predict(x)).T

#         # Change X and Y into numpy matricies. x also has a column of ones added to it.
#         x = np.hstack((np.ones((n,1)),np.matrix(x)))
#         y = np.matrix(y).T

#         # Degrees of freedom.
#         df = float(n-k-1)

#         # Sample variance.     
#         sse = np.sum(np.square(yHat - y),axis=0)
#         self.sampleVariance = sse/df

#         # Sample variance for x.
#         self.sampleVarianceX = x.T*x

#         # Covariance Matrix = [(s^2)(X'X)^-1]^0.5. (sqrtm = matrix square root.  ugly)
#         self.covarianceMatrix = sc.linalg.sqrtm(self.sampleVariance[0,0]*self.sampleVarianceX.I)

#         # Standard erros for the difference coefficients: the diagonal elements of the covariance matrix.
#         self.se = self.covarianceMatrix.diagonal()[1:]

#         # T statistic for each beta.
#         self.betasTStat = np.zeros(len(self.se))
#         for i in xrange(len(self.se)):
#             self.betasTStat[i] = self.coef_[0,i]/self.se[i]

#         # P-value for each beta. This is a two sided t-test, since the betas can be 
#         # positive or negative.
#         self.betasPValue = 1 - t.cdf(abs(self.betasTStat),df)
        
# """
# import pandas as pd
# import numpy as np
# from sklearn import datasets, linear_model
# from sklearn.linear_model import LinearRegression
# import statsmodels.api as sm
# from scipy import stats

# diabetes = datasets.load_diabetes()
# X = diabetes.data
# y = diabetes.target

# X2 = sm.add_constant(X)
# est = sm.OLS(y, X2)
# est2 = est.fit()
# print(est2.summary())
# ------------------
#                          OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.518
# Model:                            OLS   Adj. R-squared:                  0.507
# Method:                 Least Squares   F-statistic:                     46.27
# Date:                Wed, 08 Mar 2017   Prob (F-statistic):           3.83e-62
# Time:                        10:08:24   Log-Likelihood:                -2386.0
# No. Observations:                 442   AIC:                             4794.
# Df Residuals:                     431   BIC:                             4839.
# Df Model:                          10                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const        152.1335      2.576     59.061      0.000     147.071     157.196
# x1           -10.0122     59.749     -0.168      0.867    -127.448     107.424
# x2          -239.8191     61.222     -3.917      0.000    -360.151    -119.488
# x3           519.8398     66.534      7.813      0.000     389.069     650.610
# x4           324.3904     65.422      4.958      0.000     195.805     452.976
# x5          -792.1842    416.684     -1.901      0.058   -1611.169      26.801
# x6           476.7458    339.035      1.406      0.160    -189.621    1143.113
# x7           101.0446    212.533      0.475      0.635    -316.685     518.774
# x8           177.0642    161.476      1.097      0.273    -140.313     494.442
# x9           751.2793    171.902      4.370      0.000     413.409    1089.150
# x10           67.6254     65.984      1.025      0.306     -62.065     197.316
# ==============================================================================
# Omnibus:                        1.506   Durbin-Watson:                   2.029
# Prob(Omnibus):                  0.471   Jarque-Bera (JB):                1.404
# Skew:                           0.017   Prob(JB):                        0.496
# Kurtosis:                       2.726   Cond. No.                         227.
# ==============================================================================


# -----------------
# lm = LinearRegression()
# lm.fit(X,y)
# params = np.append(lm.intercept_, lm.coef_)
# predictions = lm.predict(X)

# newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
# MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

# # Note if you don't want to use a DataFrame replace the two lines above with
# # newX = np.append(np.ones((len(X),1)), X, axis=1)
# # MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX[0]))

# var_b = MSE * (np.linalg.inv(np.dot(newX.T,newX)).diagonal())
# sd_b = np.sqrt(var_b)
# ts_b = params/ sd_b

# p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

# sd_b = np.round(sd_b,3)
# ts_b = np.round(ts_b,3)
# p_values = np.round(p_values,3)
# params = np.round(params,4)

# myDF3 = pd.DataFrame()
# myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilites"] = [params, sd_b, ts_b, p_values]
# print(myDF3)

# ----------------
#     Coefficients  Standard Errors  t values  Probabilites
# 0       152.1335            2.576    59.061         0.000
# 1       -10.0122           59.749    -0.168         0.867
# 2      -239.8191           61.222    -3.917         0.000
# 3       519.8398           66.534     7.813         0.000
# 4       324.3904           65.422     4.958         0.000
# 5      -792.1842          416.684    -1.901         0.058
# 6       476.7458          339.035     1.406         0.160
# 7       101.0446          212.533     0.475         0.635
# 8       177.0642          161.476     1.097         0.273
# 9       751.2793          171.902     4.370         0.000
# 10       67.6254           65.984     1.025         0.306
# -----------
# """