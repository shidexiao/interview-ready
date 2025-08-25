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

        user_credit_amount_available = input_data_dict['user_credit_amount_available']
        loan_amount = input_data_dict['apply_principal']
        credit_info = input_data_dict['ZLC_credit_info']
        channel_code = input_data_dict['channel_code']

        credit_amount = -999

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

        if user_credit_amount_available in ['NONE', 'None', 'NULL', 'null', 'none', '', ' ']:
            user_credit_amount_available = -999

        if loan_amount <= user_credit_amount_available and loan_flag == 1:
            is_pass = 1
        else:
            is_pass = 0

        judge_info = {
            'is_pass': is_pass,
            'credit_amount': credit_amount,
        }
        other_info = {'judge_status': judge_status}

        return judge_info, other_info
    def make_judgement_batch(self, input_data_dict: List[Dict[str, Any]], input_params: Dict[str, Any]) -> List[
        Tuple[Dict, Dict]]:
        a_list = []
        for i in range(len(input_data_dict)):
            user_credit_amount_available = input_data_dict[i]['user_credit_amount_available']
            loan_amount = input_data_dict[i]['apply_principal']
            credit_info = input_data_dict[i]['ZLC_credit_info']
            channel_code = input_data_dict[i]['channel_code']

            credit_amount = -999

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
                    for j in range(n):
                        amt_code[credit_info[j]['riskCode']] = credit_info[j]['usedAmt']
                        if credit_info[j]['riskCode'] != channel_code:
                            amt_max.append(credit_info[j]['usedAmt'])
                    amt_max_ = np.max(amt_max)
                    if amt_max_ == 0:
                        loan_flag = 1
                except:
                    judge_status = 3

            if user_credit_amount_available in ['NONE', 'None', 'NULL', 'null', 'none', '', ' ']:
                user_credit_amount_available = -999

            if loan_amount <= user_credit_amount_available and loan_flag == 1:
                is_pass = 1
            else:
                is_pass = 0

            judge_info = {
                'is_pass': is_pass,
                'credit_amount': credit_amount,
            }
            other_info = {'judge_status': judge_status}
            a_list.append((judge_info, other_info))

        return a_list


###################################################

import cloudpickle


pkl_path = '/mnt/data/pkls/'

# model_name = 'reloan_rules_20231012_v1'

model_name = 'strategy_reloan_rules_20240527_v2'

columns_it = ['user_credit_amount_available', 'apply_principal', 'ZLC_credit_info', 'channel_code']
input_params = {}

mp = ModelPacker(model_name=model_name, feature_it=columns_it)


with open(f'/mnt/data/model_pack/{mp.model_name}.cdpkl', 'wb') as f:
    cloudpickle.dump(mp, f)


b = {'user_credit_amount_available': 6000, 'apply_principal': 1000, 'ZLC_credit_info' : [{'creditAmt': 6000.0, 'usedAmt': 4000.0, 'channelCode': 'ICE_ZLSK_36', 'riskCode': 'FIN360'}, {'creditAmt': 8900.0, 'usedAmt': 0, 'channelCode': 'RS', 'riskCode': 'RS'}], 'channel_code': 'RS'}
b = {'user_credit_amount_available': 6000, 'apply_principal': 1000, 'ZLC_credit_info' : [], 'channel_code': 'RS'}
b = {'user_credit_amount_available': 6000, 'apply_principal': 1000, 'ZLC_credit_info' : [{'creditAmt': 6000.0, 'channelCode': 'ICE_ZLSK_36', 'riskCode': 'FIN360'}, {'creditAmt': 8900.0, 'usedAmt': 0, 'channelCode': 'RS', 'riskCode': 'RS'}], 'channel_code': 'RS'}

mp.make_judgement(b, input_params)

mp.make_judgement_batch([b, b], input_params)





