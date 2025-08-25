from typing import List, Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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

        date_appl_submit = input_data_dict['user_success_credit_time']
        credit_amount = input_data_dict['user_credit_amount_total']
        loan_amount = input_data_dict['apply_principal']
        diff_days_limits = input_params['diff_days_limits']

        credit_info = input_data_dict['ZLC_credit_info']
        channel_code = input_data_dict['channel_code']

        if isinstance(credit_info, list):
            n = len(credit_info)
        else:
            n = 0
        loan_flag = 0
        judge_status = 1
        if n == 0:
            judge_status = 2
            loan_flag = 1
        elif n == 1:
            loan_flag = 1
        else:
            amt_code = {}
            amt_max = []
            try:
                for i in range(n):
                    amt_code[credit_info[i]['riskCode']] = credit_info[i]['usedAmt']
                    if credit_info[i]['riskCode'] != channel_code:
                        amt_max.append(credit_info[i]['usedAmt'])
                amt_max_ = np.max(amt_max)
                if amt_max_ == 0:
                    loan_flag = 1
            except:
                judge_status = 3

        current_date = datetime.now()

        if date_appl_submit in ['NONE', 'None', 'NULL', 'null', 'none', '', ' ']:
            date_appl_submit = date_appl_submit(1997, 1, 1)
        else:
            date_appl_submit = datetime(int(date_appl_submit[: 4]), int(date_appl_submit[5: 7]), int(date_appl_submit[8: 10]))

        diff_days = (current_date - date_appl_submit).days

        if diff_days <= diff_days_limits and 0 < loan_amount <= credit_amount and loan_flag == 1:
            is_pass = 1
        else:
            is_pass = 0
        
        if input_data_dict['RH_HR_score_rh_noqfico_model_20240510_rk_1_score'] not in ['NONE', 'None', 'NULL', 'null', 'none', '', ' ', None] and input_data_dict['RH_HR_appl_rules_20231129_v4_model_gt_score'] not in ['NONE', 'None', 'NULL', 'null', 'none', '', ' ', None]:
            try:
                if float(input_data_dict['RH_HR_score_rh_noqfico_model_20240510_rk_1_score']) > input_params['rh_noqfico_model_20240510_rk_1_score'] or float(input_data_dict['RH_HR_appl_rules_20231129_v4_model_gt_score']) < input_params['model_gt_20231129_v4_score']:
                    is_pass = 0
            except:
                pass

        credit_amount = -999
        judge_info = {
            'is_pass': is_pass,
            'credit_amount': credit_amount,
        }
        other_info = {'loan_apply_diff_days': diff_days, 'judge_status': judge_status, 'rh_noqfico_model_20240510_rk_1_score': input_data_dict['RH_HR_score_rh_noqfico_model_20240510_rk_1_score'], 'model_gt_20231129_v4_score': input_data_dict['RH_HR_appl_rules_20231129_v4_model_gt_score']}

        return judge_info, other_info
    def make_judgement_batch(self, input_data_dict: List[Dict[str, Any]], input_params: Dict[str, Any]) -> List[
        Tuple[Dict, Dict]]:
        a_list = []
        diff_days_limits = input_params['diff_days_limits']
        for i in range(len(input_data_dict)):
            date_appl_submit = input_data_dict[i]['user_success_credit_time']
            credit_amount = input_data_dict[i]['user_credit_amount_total']
            loan_amount = input_data_dict[i]['apply_principal']

            credit_info = input_data_dict[i]['ZLC_credit_info']
            channel_code = input_data_dict[i]['channel_code']

            if isinstance(credit_info, list):
                n = len(credit_info)
            else:
                n = 0
            loan_flag = 0
            judge_status = 1
            if n == 0:
                judge_status = 2
                loan_flag = 1
            elif n == 1:
                loan_flag = 1
            else:
                amt_code = {}
                amt_max = []
                try:
                    for i in range(n):
                        amt_code[credit_info[i]['riskCode']] = credit_info[i]['usedAmt']
                        if credit_info[i]['riskCode'] != channel_code:
                            amt_max.append(credit_info[i]['usedAmt'])
                    amt_max_ = np.max(amt_max)
                    if amt_max_ == 0:
                        loan_flag = 1
                except:
                    judge_status = 3

            current_date = datetime.now()

            if date_appl_submit in ['NONE', 'None', 'NULL', 'null', 'none', '', ' ']:
                date_appl_submit = date_appl_submit(1997, 1, 1)
            else:
                date_appl_submit = datetime(int(date_appl_submit[: 4]), int(date_appl_submit[5: 7]), int(date_appl_submit[8: 10]))

            diff_days = (current_date - date_appl_submit).days

            if diff_days <= diff_days_limits and 0 < loan_amount <= credit_amount and loan_flag == 1:
                is_pass = 1
            else:
                is_pass = 0

            if input_data_dict[i]['RH_HR_score_rh_noqfico_model_20240510_rk_1_score'] not in ['NONE', 'None', 'NULL', 'null', 'none', '', ' ', None] and input_data_dict[i]['RH_HR_appl_rules_20231129_v4_model_gt_score'] not in ['NONE', 'None', 'NULL', 'null', 'none', '', ' ', None]:
                try:
                    if float(input_data_dict[i]['RH_HR_score_rh_noqfico_model_20240510_rk_1_score']) > input_params['rh_noqfico_model_20240510_rk_1_score'] or float(input_data_dict[i]['RH_HR_appl_rules_20231129_v4_model_gt_score']) < input_params['model_gt_20231129_v4_score']:
                        is_pass = 0
                except:
                    pass

            credit_amount = -999
            judge_info = {
                'is_pass': is_pass,
                'credit_amount': credit_amount,
            }
            other_info = {'loan_apply_diff_days': diff_days, 'judge_status': judge_status, 'rh_noqfico_model_20240510_rk_1_score': input_data_dict[i]['RH_HR_score_rh_noqfico_model_20240510_rk_1_score'], 'model_gt_20231129_v4_score': input_data_dict[i]['RH_HR_appl_rules_20231129_v4_model_gt_score']}
            a_list.append((judge_info, other_info))

        return a_list


###################################################

import cloudpickle
model_name = 'strategy_firstloan_rules_20240731_v3'

columns_it = ['user_success_credit_time', 'user_credit_amount_total', 'apply_principal', 'ZLC_credit_info', 'channel_code', 'RH_HR_score_rh_noqfico_model_20240510_rk_1_score', 'RH_HR_appl_rules_20231129_v4_model_gt_score']
input_params = {"diff_days_limits": 30, "rh_noqfico_model_20240510_rk_1_score": 0.077, "model_gt_20231129_v4_score": 0.104}

mp = ModelPacker(model_name=model_name, feature_it=columns_it)

with open(f'./{mp.model_name}.cdpkl', 'wb') as f:
    cloudpickle.dump(mp, f)

b = {'user_success_credit_time': '2024-05-13 13:41:00', 'user_credit_amount_total': 6000, 'apply_principal': 1000, 'RH_HR_score_rh_noqfico_model_20240510_rk_1_score': 0.1, 'RH_HR_appl_rules_20231129_v4_model_gt_score':0.1, 'ZLC_credit_info' : [{'creditAmt': 6000.0, 'usedAmt': 4000.0, 'channelCode': 'ICE_ZLSK_36', 'riskCode': 'FIN360'}, {'creditAmt': 8900.0, 'usedAmt': 0, 'channelCode': 'RS', 'riskCode': 'RS'}], 'channel_code': 'FIN360'}
b = {'user_success_credit_time': '2024-05-13 13:41:00', 'user_credit_amount_total': 6000, 'apply_principal': 1000, 'ZLC_credit_info' : [{'creditAmt': 6000.0, 'usedAmt': 4000.0, 'channelCode': 'ICE_ZLSK_36', 'riskCode': 'FIN360'}, {'creditAmt': 8900.0, 'usedAmt': 0, 'channelCode': 'RS', 'riskCode': 'RS'}], 'channel_code': 'RS', 'RH_HR_score_rh_noqfico_model_20240510_rk_1_score': 0.001, 'RH_HR_appl_rules_20231129_v4_model_gt_score':0.2}
b = {'user_success_credit_time': '2024-05-13 13:41:00', 'user_credit_amount_total': 6000, 'apply_principal': 1000, 'ZLC_credit_info' : [], 'channel_code': 'RS', 'RH_HR_score_rh_noqfico_model_20240510_rk_1_score': 0.001, 'RH_HR_appl_rules_20231129_v4_model_gt_score':0.2}
b = {'user_success_credit_time': '2024-05-13 13:41:00', 'user_credit_amount_total': 6000, 'apply_principal': 1000, 'ZLC_credit_info' : [{'creditAmt': 6000.0, 'channelCode': 'ICE_ZLSK_36', 'riskCode': 'FIN360'}, {'creditAmt': 8900.0, 'usedAmt': 0, 'channelCode': 'RS', 'riskCode': 'RS'}], 'channel_code': 'RS', 'RH_HR_score_rh_noqfico_model_20240510_rk_1_score': "0.001", 'RH_HR_appl_rules_20231129_v4_model_gt_score':"0.2"}
b = {'user_success_credit_time': '2024-05-13 13:41:00', 'user_credit_amount_total': 6000, 'apply_principal': 1000, 'ZLC_credit_info' : [{'creditAmt': 6000.0, 'channelCode': 'ICE_ZLSK_36', 'riskCode': 'FIN360'}, {'creditAmt': 8900.0, 'usedAmt': 0, 'channelCode': 'RS', 'riskCode': 'RS'}], 'channel_code': 'RS', 'RH_HR_score_rh_noqfico_model_20240510_rk_1_score': 0.001, 'RH_HR_appl_rules_20231129_v4_model_gt_score':None}

mp.make_judgement(b, input_params)
mp.make_judgement_batch([b, b], input_params)