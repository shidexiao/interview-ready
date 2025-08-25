import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

'''

Step 1. 数据收集 → Step 2. 特征工程 → Step 3. 建模训练 → Step 4. 模型评估 → Step 5. 上线部署


传统机器学习方法（如逻辑回归、决策树、随机森林、SVM、KNN、GBDT 等）至今仍在金融风控领域广泛应用，尤其在以下场景中非常有效：

🏦 金融风控领域常见任务：
风控任务	描述
信用评分（Credit Scoring）	判断借款人是否值得授信
欺诈检测（Fraud Detection）	判断一笔交易是否可能为欺诈
贷款违约预测（Default Prediction）	预测借款人是否可能还不上钱
客户流失预测（Churn Prediction）	预测客户是否会流失或关闭账户
金融反洗钱（AML）	检测可疑资金流动行为


这些是模拟数据，
如果你有实际特征字段（如年龄、收入、设备ID、信用等级等），
可以直接用 pandas + scikit-learn 替换 make_classification。

金融场景中特征举例（常用于训练机器学习模型）：
维度	特征示例
用户属性	年龄、婚姻、学历、行业、地址稳定性
历史行为	信贷历史、还款次数、逾期天数、账单金额
设备行为	登录IP、设备ID是否一致、是否频繁更换设备
支付数据	交易频率、金额波动、消费时间分布
交叉特征	收入/还款比、信用卡使用率、上次贷款/还款间隔

逻辑回归（Logistic Regression）
📌 适用任务：
信用评分
贷款审批模型
📊 特点：
模型解释性强，金融机构乐于接受
可以输出“违约概率”
目标：判断某用户是否违约
输入特征：
- 年龄、收入、是否有房、信用卡额度、历史逾期次数等
模型输出：
- P(违约) = sigmoid(W·X + b) ∈ (0, 1)
- 若 P > 0.5，则拒贷


决策树 / 随机森林（Random Forest）
📌 适用任务：
欺诈交易检测
实时风控模型
📊 特点：
可解释性中等
能处理非线性特征
抗噪能力强，适合脏数据
目标：判断一笔交易是否欺诈

输入特征：
- 交易时间是否异常
- IP与设备是否首次出现
- 交易金额是否超阈值
- 用户是否在黑名单设备上登录

模型：随机森林多数投票判断是否欺诈


几年每年举行3️⃣ GBDT / XGBoost / LightGBM
📌 适用任务：
客户违约预测
金融欺诈检测
分期贷款额度模型

📊 特点：
强大的建模能力，能捕捉复杂的非线性关系
对缺失值、类别特征处理较好
训练速度快（LightGBM 尤其快）

目标：预测客户未来3个月是否逾期

输入特征（数百维）：
- 历史借贷行为、信用卡账单、通话记录、支付行为、消费记录

模型：
- LightGBM/ XGBoost 拟合训练集 → 输出概率评分
- 可以结合 scorecard 打分模型

用途：
- 自动调整贷款额度或审批策略


📌 常见评估指标（分类任务，尤其是风控类）：
指标	解释	为什么重要
Accuracy（准确率）	正确分类 / 总样本	在样本平衡时有用，但在不平衡数据中不可靠
Precision（精准率）	真正 / (真正 + 假正)	检测欺诈时要提高 precision，避免误杀
Recall（召回率）	真正 / (真正 + 假负)	信贷风控中非常重要，避免漏判坏客户
F1 Score	精准率与召回率的调和平均	衡量整体模型的平衡性
AUC-ROC	面积越大越好，1最好	不受阈值影响，评估模型对正负样本的区分能力
KS值（Kolmogorov–Smirnov）	正负样本得分差的最大值	信贷模型中用于衡量评分区分度
PSI（Population Stability Index）	训练集与线上分布差异	检查模型是否随时间漂移
'''

# 模拟信贷客户违约数据
X, y = make_classification(n_samples=1500, n_features=12,
                           n_informative=8, weights=[0.8],
                           random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 使用XGBoost训练模型
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 4,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}

model = xgb.train(params, dtrain, num_boost_round=50)
y_prob = model.predict(dtest)

print("违约预测AUC：", roc_auc_score(y_test, y_prob))
