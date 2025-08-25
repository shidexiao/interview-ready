# docker exec -it ml_container bash

from typing import List, Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
import random


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

    def __make_credit(self, credit_ori, score_third, score_rh):
        score_rh_th = [0,0.01450825,0.01644939,1]
        score_third_th = [0,521,542,558,1000]
        if score_third >= 0 and score_third <= 1000 and score_rh >= 0 and score_rh <= 1:
            th1 = score_rh_th + [score_rh]
            a = sorted(th1).index(score_rh)-1
            th2 = score_third_th + [score_third]
            b = sorted(th2).index(score_third)-1
            coff = 0.7
            if b == 0 and a == 0:
                coff = 0.6
            if b == 0 and a >= 1:
                coff = 0.5
            if b == 1:
                coff = 0.7
            if b == 2 and a <= 1:
                coff = 0.9
            if b == 2 and a == 2:
                coff = 0.8
            if b == 3 and a == 0:
                coff = 0.95
            if b == 3 and a >= 1:
                coff = 0.9
            return coff
        return 0.75

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

        score_third = float(input_data_dict['score_ot_general_v1_1_20240718_score'])

        flag_model_1 = (score_gt > input_params['model_gt']) \
                     & (score_risk_1 <= input_params['model_risk_1']) \
                     & (score_risk_3 <= input_params['model_risk_3'])

        flag_model_2 = (score_gt > input_params['model_gt_new']) \
                     & (score_risk_1 <= input_params['model_risk_1_new']) \
                     & (score_risk_3 <= input_params['model_risk_3_new'])

        amt_flag = 0
        coff = -1
        need_hold = 0
        increased_credit = 0
        available_credit = 0
        total_credit = 0

        if (not flag_model_1) and (not flag_model_2):
            is_pass = 0
            credit_amount = 0
        elif (not flag_model_1) and flag_model_2:
            rnd_i = random.randint(0,9)
            is_pass = 1
            credit_amount = 2000 + rnd_i * 100
            amt_flag = 1

            rnd_j = random.randint(0, 99)
            if rnd_j <= 9:
                need_hold = 1
                total_credit = credit_amount + 500
                available_credit = total_credit
                increased_credit = 500

        elif flag_model_1 and (not flag_model_2):
            # 这一行理论上不应该生效
            is_pass = 1
            credit_amount = 2000
            amt_flag = 2
        else:
            is_pass = 1
            credit_ori = min(20000, max(2000, ((input_params['model_amt'] - score_amt_new) * 15000000) // 100 * 100))
            amt_flag = 3
            coff = self.__make_credit(credit_ori, score_third, score_risk_3)
            credit_amount = (credit_ori * coff) // 100 * 100

            rnd_k = random.randint(0, 99)
            if rnd_k <= 9:
                need_hold = 1
                total_credit = credit_amount * 1.2
                increased_credit = total_credit - credit_amount
                increased_credit = 500 if increased_credit < 500 else increased_credit
                increased_credit = 2000 if increased_credit > 2000 else increased_credit
                total_credit = credit_amount + increased_credit
                available_credit = total_credit

        judge_info = {
            'is_pass': is_pass,
            'credit_amount': credit_amount,
        }
        other_info = {'model_gt_score': score_gt,
                      'model_risk_1_score': score_risk_1,
                      'model_risk_3_score': score_risk_3,
                      'model_amt_score': score_amt,
                      'model_amt_score_new': score_amt_new,
                      'amt_flag': amt_flag,
                      'coff': coff,
                      'need_hold': need_hold,
                      'available_credit': available_credit,
                      'total_credit': total_credit,
                      'increased_credit': increased_credit}

        return judge_info, other_info


###################################################

import cloudpickle

pkl_path = '/mnt/data/pkls/'

columns_it = ['RH_HR_appl_rules_20231129_v4_model_gt_score', 'RH_HR_appl_rules_20231129_v4_model_risk_1_score',
              'RH_HR_appl_rules_20231129_v4_model_risk_3_score', 'RH_HR_appl_rules_20231129_v4_model_amt_score',
              'RH_HR_appl_rules_20231129_v4_model_amt_score_new', 'score_ot_general_v1_1_20240718_score']

model_name = 'strategy_apply_model_qxl_new_amt_hl_20241021'

input_params = {'model_gt': 0.204, 'model_risk_1': 0.0843, 'model_risk_3': 0.0192, 'model_amt': 0.000887}
input_params = {'model_gt_new': 0.0742, 'model_risk_1_new': 0.0960, 'model_risk_3_new': 0.0198, 'model_amt': 0.000887,'model_gt': 0.104, 'model_risk_1': 0.0743, 'model_risk_3': 0.0182}


mp = ModelPacker(model_name=model_name, feature_it=columns_it)

with open(f'/mnt/data/pkls/luwn/model_pack/{mp.model_name}.cdpkl', 'wb') as f:
    cloudpickle.dump(mp, f)

###############
b = {'RH_HR_appl_rules_20231129_v4_model_gt_score': 0.25982296,
  'RH_HR_appl_rules_20231129_v4_model_risk_1_score': 0.04352325,
  'RH_HR_appl_rules_20231129_v4_model_risk_3_score': 0.0197893,
  'RH_HR_appl_rules_20231129_v4_model_amt_score': 0.5696902,
  'RH_HR_appl_rules_20231129_v4_model_amt_score_new': 0.00043071556501115795,
  'score_ot_general_v1_1_20240718_score': 500}

mp.make_judgement(b, input_params)
'''
({'is_pass': 1, 'credit_amount': 2800},
 {'model_gt_score': 0.25982296,
  'model_risk_1_score': 0.04352325,
  'model_risk_3_score': 0.0197893,
  'model_amt_score': 0.5696902,
  'model_amt_score_new': 0.00043071556501115795,
  'amt_flag': 1,
  'coff': -1,
  'need_hold': 0,
  'available_credit': 0,
  'total_credit': 0,
  'increased_credit': 0})
'''
# /mnt/disk1/Data/miaojun/credit/pkls/luwn/model_pack/strategy_apply_model_qxl_new_amt_20240923* 