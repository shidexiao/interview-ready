from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 加载数字数据（类似验证码）
digits = load_digits()
# 显示前10张图
plt.figure(figsize=(10, 2))

for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(digits.images[i], cmap='gray')  # 原始图像是二维的 (8x8)
    plt.title(str(digits.target[i]))
    plt.axis('off')

plt.tight_layout()
plt.show()

X, y = digits.data, digits.target  # 每张图是8x8像素，展平为64维

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN 模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("✅ KNN识别准确率：", accuracy_score(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("✅ 决策树准确率：", accuracy_score(y_test, y_pred))

'''
 不适合传统算法的验证码类型：
多字符未分割（如4个字符连在一起）

扭曲/旋转/有背景干扰的

表现为图片图层 + 字体重叠

这些情况通常需要 CNN、深度学习（如 CRNN） 或图像分割技术辅助处理。


'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from sklearn.datasets import load_digits

# 加载数字数据集
digits = load_digits()
X_raw = digits.images  # 原始图像 shape: (n, 8, 8)
y = digits.target

# 提取HOG特征
X_hog = [hog(img, pixels_per_cell=(4, 4), cells_per_block=(1, 1)) for img in X_raw]

# 拆分数据
X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.3, random_state=42)

# 训练逻辑回归
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print("✅ 逻辑回归 + HOG 特征准确率：", accuracy_score(y_test, pred))
'''
验证码类型	推荐方法
清晰、单字符	KNN / 决策树 / 随机森林 / SVM
有轻微噪声	随机森林 / GBDT / SVM
图像结构复杂	SVM + HOG / RF + PCA
多字符连体、扭曲、干扰	🔺传统方法很难，需要 CNN 或 CRNN
'''


