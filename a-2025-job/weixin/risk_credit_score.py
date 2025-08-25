from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 模拟客户信用数据
X, y = make_classification(n_samples=1000, n_features=10,
                           n_informative=6, n_redundant=2,
                           weights=[0.7], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 使用逻辑回归训练信用评分模型
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("信用评分模型结果：")
print(classification_report(y_test, y_pred))
