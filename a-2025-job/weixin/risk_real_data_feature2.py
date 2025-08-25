import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils import shuffle
import xgboost as xgb
import shap

# 1. 生成模拟数据（同之前）
np.random.seed(42)
n = 2000
df = pd.DataFrame({
    'age': np.random.randint(18, 70, size=n),
    'gender': np.random.choice(['Male', 'Female'], size=n),
    'income': np.random.normal(80000, 20000, size=n).clip(20000, 200000),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], size=n),
    'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=n),
    'credit_limit': np.random.normal(50000, 15000, size=n).clip(5000, 150000),
    'payment_history': np.random.poisson(1, size=n),
    'balance': np.random.normal(30000, 10000, size=n).clip(0, 100000),
})
df['default'] = (np.random.rand(n) < 0.05).astype(int)

# 2. 清洗 + 特征工程
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df = pd.get_dummies(df, columns=['marital_status', 'education_level'], drop_first=True)
df['income_to_limit'] = df['income'] / df['credit_limit']
df['avg_late_per_month'] = df['payment_history'] / 6
df['balance_ratio'] = df['balance'] / df['credit_limit']

scale_cols = ['age', 'income', 'credit_limit', 'balance',
              'income_to_limit', 'avg_late_per_month', 'balance_ratio']
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

X = df.drop(columns=['default'])
y = df['default']
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# 3. 逻辑回归 - 使用 class_weight 解决不平衡
lr = LogisticRegression(max_iter=500, class_weight='balanced')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]
print("逻辑回归报告:\n", classification_report(y_test, y_pred_lr))
print("逻辑回归 AUC:", roc_auc_score(y_test, y_prob_lr))

# 4. XGBoost 调参示例 + 交叉验证
param_grid = {
    'max_depth': [3, 5],
    'n_estimators': [50, 100],
    'scale_pos_weight': [1, sum(y_train==0)/sum(y_train==1)]  # 处理类别不平衡
}
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(xgb_clf, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]

print("XGBoost最佳参数:", grid_search.best_params_)
print("XGBoost报告:\n", classification_report(y_test, y_pred_xgb))
print("XGBoost AUC:", roc_auc_score(y_test, y_prob_xgb))

# 5. SHAP解释变量重要性（XGBoost）
explainer = shap.Explainer(best_xgb)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
