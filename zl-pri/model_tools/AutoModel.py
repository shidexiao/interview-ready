# -*- coding: utf-8 -*-
"""
Auto XGBoost and Auto Scorecard
"""
import os
import pickle
import warnings
from abc import abstractmethod

import cloudpickle
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier

import scorecardpy as sc

from .data import DataHelper
from .metrics import ks, roc_auc_score
from .Model.model_utils import (GreedyThresholdSelector,
                                GreedyThresholdSelector1)
from .Model.params_tune import XGBoostBayesOptim
from .Preprocessing.stabler import get_trend_stats
from .ScoreCard import model_helper, modeler
from .utils import timer

warnings.filterwarnings("ignore")


def is_numpy(x):
    return isinstance(x, np.ndarray)


def is_pandas(x):
    return isinstance(x, pd.DataFrame)


class CategoryEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_features, first_category=1, copy=True):
        self.categorical_features = categorical_features
        self.first_category = first_category
        self.copy = copy
        self.encoders = {}
        self.ive = None

    def fit(self, X, y=0):
        if not is_pandas(X):
            raise TypeError("Input X must a dataframe!")

        x = X[self.categorical_features].fillna("")
        self.encoders = {}
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        for i in self.categorical_features:
            ind_v = x[i].fillna("").value_counts().rank(
                ascending=0, method='first').astype(int).to_dict()
            self.encoders.update({i: ind_v})

        return self

    def fit_transform(self, X, y=0):
        self.fit(X, y=0)
        return self.transform(X, y=0)

    def transform(self, X, y=0):
        if not is_pandas(X):
            raise TypeError("Input X must a dataframe!")

        inner_categorical = [
            x for x in X.columns if x in self.categorical_features]
        x = X[inner_categorical].fillna("")
        if self.copy:
            x = X[inner_categorical].fillna("").copy()

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        for i in inner_categorical:
            enc = self.encoders[i]
            x.loc[:, i] = x.loc[:, i].fillna("").map(enc)
        X[inner_categorical] = x.values

        return X


class CountEncoder(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            categorical_features,
            min_count=0,
            nan_value=-1,
            copy=True):
        self.categorical_features = categorical_features
        self.min_count = min_count
        self.nan_value = nan_value
        self.copy = copy
        self.counts = {}

    def fit(self, X, y=0):
        self.counts = {}
        if not is_pandas(X):
            raise TypeError("Input x must a dataframe!")

        x = X[self.categorical_features].fillna("")
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        for i in self.categorical_features:
            cnt = x.loc[:, i].value_counts().to_dict()
            if self.min_count > 0:
                cnt = dict((k, self.nan_value if v < self.min_count else v)
                           for k, v in cnt.items())
            self.counts.update({i: cnt})
        return self

    def fit_transform(self, X, y=0):
        self.fit(X, y=0)
        return self.transform(X, y=0)

    def transform(self, X, y=0):
        if not is_pandas(X):
            raise TypeError("Input x must a dataframe!")

        inner_categorical = [
            x for x in X.columns if x in self.categorical_features]
        x = X[inner_categorical].fillna("")

        if self.copy:
            x = X[inner_categorical].fillna("").copy()

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        for i in inner_categorical:
            cnt = self.counts[i]
            x.loc[:, i] = x.loc[:, i].map(cnt)
        x = x.add_prefix('cnt_enc_')
        output = pd.concat([X, x], axis=1)
        return output


class AutoXGBoost(object):

    def __init__(
            self,
            train,
            test,
            target,
            key,
            unuse_variables,
            categorical_features,
            date_cols,
            missing_rate_threshold,
            select_min,
            max_features_num,
            trend_correlation_list,
            output_file,
            project_name,
            bin_method='tree',
            bin_num=9):

        self.train = train
        self.test = test
        self.target = target
        self.key = key
        self.unuse_variables = unuse_variables
        self.categorical_features = categorical_features
        self.date_cols = date_cols
        self.missing_rate_threshold = missing_rate_threshold
        self.select_min = select_min
        self.max_features_num = max_features_num
        self.bin_method = bin_method
        self.bin_num = bin_num
        self.trend_correlation_list = trend_correlation_list
        self.output_file = output_file
        self.project_name = project_name
        assert self.key not in self.unuse_variables, "key must not in unuse_variables. "

    @abstractmethod
    def _data_preprocess(self, **kwargs):
        return self

    def _data_combine(self):
        self.train = self.train[[
            x for x in self.train.columns if x not in self.unuse_variables]]
        self.test = self.test[[
            x for x in self.test.columns if x not in self.unuse_variables]]
        self.datahper = DataHelper(target=self.target,
                                   train_path=None,
                                   test_path=None,
                                   trainfile=self.train,
                                   testfile=self.test,
                                   date_cols=self.date_cols)
        data = self.datahper.combine()
        dtypes = data.dtypes
        self.categorical_variables = dtypes[dtypes == 'object'].index.tolist(
        ) if self.categorical_features is None or self.categorical_features == [] else self.categorical_features
        print("Categorical Variables: ", self.categorical_features)
        self.continues_features = self.datahper.continues_features
        self.bool_variables = dtypes[dtypes == 'bool'].index.tolist()
        for col in self.bool_variables:
            data[col] = data[col].apply(lambda x: 1 if x else 0)
        self.data = data
        return self

    def _data_report(self):
        if not os.path.exists(self.output_file +
                              "{0}_Data_Report.csv".format(self.project_name)):
            self.data_report = model_helper.data_report(self.train)
            self.data_report.to_csv(
                self.output_file +
                "{0}_Data_Report.csv".format(self.project_name),
                index=False)
        else:
            self.data_report = pd.read_csv(self.output_file +
                                           "{0}_Data_Report.csv".format(self.project_name))
        return self

    def _feature_engineer(self):
        variable_filter = self.data_report[self.data_report['unique'] > 1]
        variable_filter = variable_filter[~(
            (variable_filter['dtype'] == 'object') & (variable_filter['unique'] >= 20))]
        variable_filter = variable_filter[variable_filter['missing_rate']
                                          <= self.missing_rate_threshold]
        variable_filter = variable_filter[variable_filter['dtype']
                                          != 'datetime64[ns]']
        filt_variables = list(variable_filter['column'])

        self.categorical_variables = [
            x for x in self.categorical_variables if x in filt_variables]

        if self.categorical_variables:
            pipe = Pipeline([
                ('CatgoryEncoder', CategoryEncoder(self.categorical_variables)),
                ('CountEncoder', CountEncoder(self.categorical_variables)),
            ])
            self.data = pipe.fit_transform(self.data[filt_variables])

            cloudpickle.dump(
                pipe,
                open(
                    self.output_file +
                    "{0}_pipeline.pkl".format(self.project_name),
                    "wb"))
        self.train, self.test = self.datahper.split(self.data)
        use_cols = [
            x for x in filt_variables if x not in [
                self.key, self.target]] + [x for x in self.train.columns if x.startswith('cnt_enc_')]
        self.train[use_cols] = self.train[use_cols].fillna(-999)
        self.test[use_cols] = self.test[use_cols].fillna(-999)
        self.use_cols = use_cols if self.target in use_cols else use_cols + \
            [self.target]

    def _feature_select(self, correlate=False, repeat=False):
        with timer("data process"):
            self._data_preprocess()
            self._data_combine()
        with timer("data report"):
            self._data_report()
        with timer("feature engineer"):
            self._feature_engineer()
        with timer("feature select"):
            if not os.path.exists(self.output_file +
                                  "{0}_Trend_Correlation.csv".format(self.project_name)):
                if correlate:
                    stats = get_trend_stats(
                        data=self.train[self.use_cols],
                        target_col=self.target,
                        bins=self.bin_num,
                        data_test=self.test[self.use_cols],
                        method=self.bin_method)
                    stats.sort_values('Trend_correlation',
                                      ascending=False, inplace=True)
                else:
                    stats = pd.DataFrame(
                        columns=['Feature', 'Trend_changes', 'Trend_changes_test', 'Trend_correlation'])
                    stats['Feature'] = [
                        x for x in self.use_cols if x != self.target]
                    stats['Trend_changes'] = 1
                    stats['Trend_changes_test'] = 1
                    stats['Trend_correlation'] = 1

                stats.to_csv(
                    self.output_file +
                    "{0}_Trend_Correlation.csv".format(self.project_name),
                    index=False)
            else:
                stats = pd.read_csv(self.output_file +
                                    "{0}_Trend_Correlation.csv".format(self.project_name))
            self.stats = stats

            if not os.path.exists(self.output_file +
                                  "{0}_GS_Result.csv".format(self.project_name)) or repeat:

                gbm_model = LGBMClassifier(boosting_type='gbdt', num_leaves=2 ** 5, max_depth=5,
                                           learning_rate=0.1, n_estimators=350,  # class_weight=20,
                                           min_child_samples=20,
                                           subsample=0.95, colsample_bytree=0.95,
                                           reg_alpha=0.1, reg_lambda=0.1,
                                           sample_weight=None, seed=1001  # init_score=0.5
                                           )

                best_model, shap_correlation, result = GreedyThresholdSelector1(
                    self.train[self.use_cols],
                    self.target,
                    self.test[self.use_cols],
                    gbm_model,
                    stats,
                    self.trend_correlation_list,
                    5,
                    self.select_min,
                    self.max_features_num,
                    [1001],
                    eval_metric='auc',
                    verbose=True)

                shap_correlation.to_csv(
                    self.output_file +
                    "{0}_Shap_Importance.csv".format(self.project_name),
                    index=False)

                result.to_csv(
                    self.output_file +
                    "{0}_GS_Result.csv".format(self.project_name),
                    index=False)

            else:
                result = pd.read_csv(self.output_file +
                                     "{0}_GS_Result.csv".format(self.project_name))

            self.gs_result = result
            self.best_model = best_model

            init_variables = result.sort_values('test_auc', ascending=False).head(1)[
                'sub_columns'].values[0]
            self.init_variables = eval(init_variables) if type(
                init_variables) == type("a") else init_variables

    def xgboost_model(
            self,
            proba_index=1,
            sel_cols=None,
            xgboost_params=None,
            num_boost_round=None,
            group=10,
            save=True):
        assert hasattr(
            self, 'init_variables') or sel_cols is not None, "Before run xgboost model, must select variables."
        if (not hasattr(self, 'init_variables')
            ) and sel_cols is not None and not hasattr(self, 'sel_cols'):
            with timer("data process"):
                self._data_preprocess()
                self._data_combine()
            with timer("data report"):
                self._data_report()
            with timer("feature engineer"):
                self._feature_engineer()

        with timer("build model"):
            if xgboost_params is None:
                model = XGBClassifier(
                    base_score=0.5,
                    colsample_bylevel=1,
                    colsample_bytree=0.843,
                    gamma=0.1,
                    learning_rate=0.05,
                    max_delta_step=0,
                    max_depth=4,
                    min_child_weight=5,
                    missing=None,
                    n_estimators=301,
                    nthread=-1,
                    n_jobs=-1,
                    objective='binary:logistic',
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    scale_pos_weight=1,
                    seed=42,
                    silent=True,
                    subsample=0.9)
            else:
                model = XGBClassifier(**xgboost_params)

            if sel_cols is None and hasattr(self, 'init_variables'):
                sel_cols = self.init_variables

            self.sel_cols = sel_cols

            params = model.get_xgb_params()
            xgb_train = xgb.DMatrix(
                self.train[sel_cols], label=self.train[self.target])
            cv_result = xgb.cv(
                params,
                xgb_train,
                num_boost_round=301,
                nfold=5,
                metrics='auc',
                seed=42,
                early_stopping_rounds=25,
                verbose_eval=False)
            if num_boost_round is not None:
                assert num_boost_round < cv_result.shape[0], "num boost round must less than best round."
            num_round_best = cv_result.shape[0] if num_boost_round is None else num_boost_round
            print('Best round num: ', num_round_best)
            # train
            params['n_estimators'] = num_round_best

            model = XGBClassifier(**params)
            model.fit(self.train[sel_cols], self.train[self.target])
            self.model = model
            dev_xgb_predict = model.predict_proba(
                self.train[sel_cols])[:, proba_index]
            oot_xgb_predict = model.predict_proba(
                self.test[sel_cols])[:, proba_index]

            self.train['proba'] = dev_xgb_predict
            self.test['proba'] = oot_xgb_predict

            auc_mean = round(
                cv_result['test-auc-mean'].tolist()[num_round_best], 4)
            auc_std = round(
                cv_result['test-auc-std'].tolist()[num_round_best], 4)
            print(f'- 5Folds AUC: {auc_mean}, STD: {auc_std}')
            print('- Test AUC: %.4f' %
                  roc_auc_score(self.test[self.target], self.test['proba']))
            print('- Test KS : %.4f' %
                  ks(self.test[self.target], self.test['proba']))

            importance_df = pd.DataFrame()
            importance_df['feature'] = sel_cols
            importance_df['importance'] = model.feature_importances_

            shap_values = model.get_booster().predict(xgb.DMatrix(
                self.train[sel_cols].fillna(-999)), pred_contribs=True)
            shap_df = pd.DataFrame(
                np.abs(shap_values[:, :-1]), columns=sel_cols)

            shap_imp = shap_df.mean().sort_values(ascending=False).reset_index()
            shap_imp.columns = ['Feature', 'Shap_Importance']
            shap_imp.to_csv(self.output_file +
                            "{0}_Input_Shap_Importance.csv".format(self.project_name), index=False)
            if save:
                pickle.dump(
                    model, open(
                        self.output_file +
                        '{0}_model.pkl'.format(self.project_name), 'wb'))
                pickle.dump(
                    sel_cols, open(
                        self.output_file +
                        "{0}_input_variables.pkl".format(
                            self.project_name), 'wb'))

                importance_df.to_csv(
                    self.output_file +
                    "{0}_Feature_Importance.csv".format(self.project_name),
                    index=False)

                group_analysis = model_helper.model_group_monitor(
                    self.test, self.target, 'proba', bool(1-proba_index), group)
                group_analysis.to_csv(
                    self.output_file +
                    "{0}_Group_Analysis.csv".format(self.project_name),
                    index=False)

                self.test[[self.key, self.target, 'proba']].to_csv(
                    self.output_file +
                    "{0}_Test_Sample.csv".format(self.project_name),
                    index=False)

    def optim_params(self, init_points=5, n_iter=15):
        best_params = XGBoostBayesOptim(
            self.train[self.sel_cols + [self.target]], self.target,
            init_points=init_points, n_iter=n_iter)
        self.best_params = best_params
        return self

    def bin_woe(self, input_variables=None):
        if input_variables is None:
            input_variables = self.sel_cols
        bins_chimerge = sc.woebin(self.train[input_variables+[self.target]],
                                  y=self.target,
                                  bin_num_limit=6,
                                  method="chimerge",
                                  count_distr_limit=0.04,
                                  no_cores=16)

        bins_df = pd.DataFrame()
        for k, v in bins_chimerge.items():
            bins_df = pd.concat([bins_df, v])

        bins_df.sort_values(['total_iv', 'breaks'], ascending=[
                            False, True], inplace=True)
        bins_df.to_csv(
            self.output_file + "{0}_Bin_Report.csv".format(self.project_name), index=False)

    def auto_report(self):
        data_report = pd.read_csv(
            self.output_file + "{0}_Data_Report.csv".format(self.project_name))
        if not os.path.exists(self.output_file +
                              "{0}_Bin_Report.csv".format(self.project_name)):
            self.bin_woe()
        bin_report = pd.read_csv(
            self.output_file + "{0}_Bin_Report.csv".format(self.project_name))
        feature_importance = pd.read_csv(
            self.output_file + "{0}_Input_Shap_Importance.csv".format(self.project_name))
        group_analysis = pd.read_csv(
            self.output_file + "{0}_Group_Analysis.csv".format(self.project_name))
        target_distribution = pd.read_csv(
            self.output_file + "target_distribution.csv")

        writer = pd.ExcelWriter(
            self.output_file + "{0}_Model_Report.xlsx".format(self.project_name))
        workbook = writer.book

        # 设置格式
        text_fmt = workbook.add_format({'font_name': '微软雅黑', 'font_size': 9})
        percent_fmt = workbook.add_format(
            {'font_name': '微软雅黑', 'font_size': 9, 'num_format': '0.00%'})
        int_fmt = workbook.add_format(
            {'font_name': '微软雅黑', 'font_size': 9, 'num_format': '#,##0'})
        float_fmt = workbook.add_format(
            {'font_name': '微软雅黑', 'font_size': 9, 'num_format': '#,##0.00'})
        float_fmt_1 = workbook.add_format(
            {'font_name': '微软雅黑', 'font_size': 9, 'num_format': '#,##0.0000'})
        fill_fmt = workbook.add_format({'bg_color': '#FCE4D6'})

        header_1 = {
            'bold': True,  # 粗体
            'font_name': '微软雅黑',
            'font_size': 9,
            'font_color': '#FFFFFF',
            'border': False,  # 边框线
            'align': 'center',  # 水平居中
            'valign': 'vcenter',  # 垂直居中
            'bg_color': '#1F4E78'  # 背景颜色
        }
        header_fmt = workbook.add_format(header_1)

        header_2 = {
            'bold': True,  # 粗体
            'font_name': '微软雅黑',
            'border': False,
            'font_size': 9,
            'align': 'center',  # 水平居中
            'valign': 'vcenter'  # 垂直居中
        }
        header_fmt_2 = workbook.add_format(header_2)
        header_fmt_2.set_bottom()
        date_fmt = workbook.add_format(
            {'bold': True, 'font_size': 9, 'font_name': u'微软雅黑', 'num_format': 'yyyy-mm',
                'valign': 'vcenter', 'align': 'center'})

        # 1.Data_Report
        data_report.to_excel(writer, sheet_name=u'1.Data_Report', encoding='utf8',
                             header=False, index=False, startcol=1, startrow=1, freeze_panes=(1, 0))
        worksheet1 = writer.sheets[u'1.Data_Report']
        for col_num, value in enumerate(data_report.columns.values):
            worksheet1.write(0, col_num+1, value, header_fmt)

        # 1.1生效单元格格式
        # 增加个表格说明
        # worksheet1.merge_range('B1:C1', u'数据报告', note_fmt)
        worksheet1.set_column('A:A', 1)
        worksheet1.set_column('B:C', 9, text_fmt)
        for col in 'DEG':
            worksheet1.set_column(f'{col}:{col}', 9, int_fmt)
        worksheet1.set_column('F:F', 9, percent_fmt)
        worksheet1.set_column('H:P', 9, float_fmt)
        d_len = data_report.shape[0]+1
        worksheet1.conditional_format(f'F2:F{d_len}', {'type': 'data_bar',
                                                       'bar_solid': True,
                                                       })

        # 2.BIN_WoE
        bin_report.drop("is_special_values", axis=1, inplace=True)
        bin_report['breaks'] = bin_report['breaks'].replace(
            "missing", -np.inf).replace("inf", np.inf).astype(float)
        bin_report.sort_values(by=["total_iv", "variable", "breaks"], ascending=[
                               False, True, True], inplace=True)
        bin_report.reset_index(drop=True, inplace=True)
        bin_report.to_excel(writer, sheet_name='2.BIN_WoE', encoding='utf8',
                            header=False, index=False, startcol=1, startrow=1, freeze_panes=(1, 0))
        worksheet2 = writer.sheets['2.BIN_WoE']
        for col_num, value in enumerate(bin_report.columns.values):
            worksheet2.write(0, col_num+1, value, header_fmt_2)

        worksheet2.set_column('A:A', 1)
        for col in 'BCL':
            worksheet2.set_column(f'{col}:{col}', 9, text_fmt)
        for col in 'DFG':
            worksheet2.set_column(f'{col}:{col}', 9, int_fmt)
        for col in 'EH':
            worksheet2.set_column(f'{col}:{col}', 9, percent_fmt)
        for col in 'IJK':
            worksheet2.set_column(f'{col}:{col}', 9, float_fmt_1)

        for i, col in enumerate(bin_report['variable'].drop_duplicates().tolist()):
            temp = bin_report.query(f"variable == '{col}'")
            row_min, row_max = min(temp.index)+2, max(temp.index)+2
            if i % 2 == 0:
                worksheet2.conditional_format(f"B{row_min}:L{row_max}", {
                                              'type': 'formula', 'criteria': '1>0', "format": fill_fmt})
                worksheet2.conditional_format(f'H{row_min}:H{row_max}', {'type': 'data_bar',
                                                                         'bar_solid': True, 'bar_color': '#FFB628'
                                                                         })
            else:
                worksheet2.conditional_format(f'H{row_min}:H{row_max}', {'type': 'data_bar',
                                                                         'bar_solid': True, 'bar_color': '#FFB628'
                                                                         })

        # 3.Feature Importance
        feature_importance.to_excel(writer, sheet_name='3.Feature_Importance',
                                    encoding='utf8', header=False, index=False, startcol=1, startrow=1)
        worksheet3 = writer.sheets['3.Feature_Importance']
        for col_num, value in enumerate(feature_importance.columns.values):
            worksheet3.write(0, col_num+1, value, header_fmt_2)

        f_with = max([len(x)
                      for x in feature_importance['Feature'].tolist()])*0.85
        worksheet3.set_column('A:A', 1)
        worksheet3.set_column('B:B', f_with, text_fmt)
        worksheet3.set_column('C:C', 15, float_fmt_1)
        f_len = feature_importance.shape[0]+1
        worksheet3.conditional_format(f'C2:C{f_len}', {'type': 'data_bar',
                                                       'bar_solid': True
                                                       })

        # 4.group_analysis
        group_analysis.drop(['cum_bad', 'cum_good'], axis=1, inplace=True)
        group_analysis['Lift'] = group_analysis['group_bad_rate'] / \
            group_analysis['total_bad_rate']
        group_analysis.to_excel(writer, sheet_name='4.Group_Analysis', encoding='utf8',
                                header=False, index=False, startcol=1, startrow=1, freeze_panes=(1, 0))

        worksheet4 = writer.sheets['4.Group_Analysis']
        for col_num, value in enumerate(group_analysis.columns.values):
            worksheet4.write(0, col_num+1, value, header_fmt)

        worksheet4.set_column('A:A', 1)
        for col in 'BCEFG':
            worksheet4.set_column(f'{col}:{col}', 9, int_fmt)
        for col in 'DHIJ':
            worksheet4.set_column(f'{col}:{col}', 9, percent_fmt)
        for col in 'KLMN':
            worksheet4.set_column(f'{col}:{col}', 9, float_fmt_1)
        worksheet4.set_column('O:O', 9, float_fmt)
        g_len = group_analysis.shape[0]+1
        worksheet4.conditional_format(f'H2:H{g_len}', {'type': 'data_bar',
                                                       'bar_solid': True, 'bar_color': '#FFB628'
                                                       })
        # 5.target distribution
        target_distribution.to_excel(writer, sheet_name='5.Target_Distribution', encoding='utf8',
                                     header=False, index=False, startcol=1, startrow=1, freeze_panes=(1, 0))
        worksheet5 = writer.sheets['5.Target_Distribution']
        for col_num, value in enumerate(target_distribution.columns.values):
            worksheet5.write(0, col_num+1, value, header_fmt)

        worksheet5.set_column('A:A', 1)
        worksheet5.set_column('B:B', 9, date_fmt)
        for col in 'CD':
            worksheet5.set_column(f'{col}:{col}', 9, int_fmt)
        worksheet5.set_column('E:E', 9, percent_fmt)
        writer.save()
