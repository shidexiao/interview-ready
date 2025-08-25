from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 只选两类用于二分类
iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Sklearn GBDT
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier()
gbdt.fit(X_train, y_train)
pred_gbdt = gbdt.predict(X_test)
print("✅ GBDT Accuracy:", accuracy_score(y_test, pred_gbdt))

# 2. XGBoost
import xgboost as xgb
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_test)
print("✅ XGBoost Accuracy:", accuracy_score(y_test, pred_xgb))

# 3. LightGBM
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)
pred_lgb = lgb_model.predict(X_test)
print("✅ LightGBM Accuracy:", accuracy_score(y_test, pred_lgb))
