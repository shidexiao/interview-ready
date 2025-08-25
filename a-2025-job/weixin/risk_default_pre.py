import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# Step 1: 加载数据（模拟数据或读取业务数据）
# 假设字段有 age, income, loan_amount, past_due, is_default
df = pd.read_csv("loan_data.csv")

# Step 2: 特征工程
df.fillna(0, inplace=True)  # 缺失值填充
X = df.drop("is_default", axis=1)
y = df["is_default"]

# Step 3: 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Step 4: 建模训练
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 4,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}
model = xgb.train(params, dtrain, num_boost_round=50)

# Step 5: 模型评估
y_prob = model.predict(dtest)
y_pred = [1 if p > 0.5 else 0 for p in y_prob]

print("AUC:", roc_auc_score(y_test, y_prob))
print("详细评估：\n", classification_report(y_test, y_pred))
'''
部署上线与监控（实际业务中会加）
阶段	内容
模型部署	转换为 API 接口、模型打包
模型监控	AUC、KS、PSI 持续监控，检查模型老化
模型解释	使用 SHAP 或特征重要性解释输出结果（利于审计）
'''