from typing import List, Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd


class ModelPacker(object):
    def __init__(self, *args, **kwargs):
        self.modelname = kwargs['model_name']
        self.feature_it = kwargs['feature_it']

    @property
    def model_name(self):
        return self.modelname

    @property
    def input_columns(self) -> List:
        return self.feature_it

    def __data_process(self):
        # 数据处理
        pass

    def __model_predict(self):
        # 模型预测评分
        pass

    def __threshold_check(self):
        # 根据参数判断是否通过
        pass

    def __make_credit(self):
        # 给出额度
        pass

    def make_judgement(self, input_data_dict: Dict[str, Any], input_params: Dict[str, Any]) -> Tuple[Dict, Dict]:

        n_cols = [x for x in self.feature_it if x not in input_data_dict]
        for col in n_cols:
            input_data_dict[col] = -999

        for key in self.feature_it:
            if pd.isna(input_data_dict[key]):
                input_data_dict[key] = -999

        score_gt = float(input_data_dict['RH_HR_appl_rules_20231129_v4_model_gt_score'])
        score_risk_1 = float(input_data_dict['RH_HR_appl_rules_20231129_v4_model_risk_1_score'])
        score_risk_3 = float(input_data_dict['RH_HR_appl_rules_20231129_v4_model_risk_3_score'])
        score_amt = float(input_data_dict['RH_HR_appl_rules_20231129_v4_model_amt_score'])
        score_amt_new = float(input_data_dict['RH_HR_appl_rules_20231129_v4_model_amt_score_new'])


        flag_model = (score_gt > input_params['model_gt']) \
                     & (score_risk_1 <= input_params['model_risk_1']) \
                     & (score_risk_3 <= input_params['model_risk_3'])

        if not flag_model:
            is_pass = 0
            credit_amount = 0
        else:
            is_pass = 1
            credit_amount = min(20000, max(2000, (7500 + (input_params['model_amt'] - score_amt_new) * 5000000) // 100 * 100))

        judge_info = {
            'is_pass': is_pass,
            'credit_amount': credit_amount,
        }
        other_info = {'model_gt_score': score_gt,
                      'model_risk_1_score': score_risk_1,
                      'model_risk_3_score': score_risk_3,
                      'model_amt_score': score_amt,
                      'model_amt_score_new': score_amt_new, 'cyzy_flag': 0}

        return judge_info, other_info

    def make_judgement_batch(self, input_data_dict: List[Dict[str, Any]], input_params: Dict[str, Any]) -> List[
        Tuple[Dict, Dict]]:
        a_list = []
        for i in range(len(input_data_dict)):
            input_data_dict_ = input_data_dict[i]
            n_cols = [x for x in self.feature_it if x not in input_data_dict_]
            for col in n_cols:
                input_data_dict_[col] = -999

            for key in self.feature_it:
                if pd.isna(input_data_dict_[key]):
                    input_data_dict_[key] = -999

            score_gt = float(input_data_dict_['RH_HR_appl_rules_20231129_v4_model_gt_score'])
            score_risk_1 = float(input_data_dict_['RH_HR_appl_rules_20231129_v4_model_risk_1_score'])
            score_risk_3 = float(input_data_dict_['RH_HR_appl_rules_20231129_v4_model_risk_3_score'])
            score_amt = float(input_data_dict_['RH_HR_appl_rules_20231129_v4_model_amt_score'])
            score_amt_new = float(input_data_dict_['RH_HR_appl_rules_20231129_v4_model_amt_score_new'])


            flag_model = (score_gt > input_params['model_gt']) \
                         & (score_risk_1 <= input_params['model_risk_1']) \
                         & (score_risk_3 <= input_params['model_risk_3'])

            if not flag_model:
                is_pass = 0
                credit_amount = 0
            else:
                is_pass = 1
                credit_amount = min(20000, max(2000, (
                            7500 + (input_params['model_amt'] - score_amt_new) * 5000000) // 100 * 100))

            judge_info = {
                'is_pass': is_pass,
                'credit_amount': credit_amount,
            }
            other_info = {'model_gt_score': score_gt,
                          'model_risk_1_score': score_risk_1,
                          'model_risk_3_score': score_risk_3,
                          'model_amt_score': score_amt,
                          'model_amt_score_new': score_amt_new, 'cyzy_flag': 0}

            a_list.append((judge_info, other_info))

        return a_list


###################################################

import cloudpickle

pkl_path = '/mnt/data/pkls/'

columns_it = ['RH_HR_appl_rules_20231129_v4_model_gt_score', 'RH_HR_appl_rules_20231129_v4_model_risk_1_score',
              'RH_HR_appl_rules_20231129_v4_model_risk_3_score', 'RH_HR_appl_rules_20231129_v4_model_amt_score',
              'RH_HR_appl_rules_20231129_v4_model_amt_score_new']

model_name = 'strategy_apply_model_rules_C_20240528'

input_params = {"model_gt": 0.104, "model_risk_1": 0.08, "model_risk_3": 0.0183, "model_amt": 0.000887}

mp = ModelPacker(model_name=model_name, feature_it=columns_it)

with open(f'/mnt/data/model_pack/{mp.model_name}.cdpkl', 'wb') as f:
    cloudpickle.dump(mp, f)

###############
b = {'RH_HR_appl_rules_20231129_v4_model_gt_score': 0.25982296,
  'RH_HR_appl_rules_20231129_v4_model_risk_1_score': 0.04352325,
  'RH_HR_appl_rules_20231129_v4_model_risk_3_score': 0.022997893,
  'RH_HR_appl_rules_20231129_v4_model_amt_score': 0.5696902,
  'RH_HR_appl_rules_20231129_v4_model_amt_score_new': 0.00043071556501115795}

input_params = {"model_gt": 0.104, "model_risk_1": 0.08, "model_risk_3": 0.0183, "model_amt": 0.000887}

mp.make_judgement(b, input_params)