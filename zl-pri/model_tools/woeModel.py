
# 自动化评估报告
class AutoModel:
    def __init__(self):
        # 全局参数
        """
        1) 变量筛选
        - 缺失率阈值
          missing_threshold = 0.85
        - IV计算方式
          iv_method: mono
        2) 算法
        - accumulate_importance_rate = 0.9
        - CatBoost
        3) Y
        target
        """
    def sample_distribution(self, model_data, group_key='source'):
        assert group_key in model_data.columns
        self.distribution_1 = f_mi_1(self.target)
        self.distribution_2 = None

    def input_statistic(self, model_data):
        # data report
        data_report = None
        self.data_report = data_report

    

    def woe(self, model_data, condicate_variables):




    
