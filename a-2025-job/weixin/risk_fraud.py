from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 模拟交易数据（不平衡数据）
X, y = make_classification(n_samples=2000, n_features=20,
                           n_informative=10, weights=[0.95],
                           flip_y=0.01, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("欺诈检测混淆矩阵：")
print(confusion_matrix(y_test, y_pred))
