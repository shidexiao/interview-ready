
import pandas as pd

'''
Step 1：数据准备（加载 + 清洗）
'''
# 模拟数据加载
df = pd.read_csv('credit_card_customers.csv')  # 或 pd.read_excel(...)

# 初步检查
print(df.head())
print(df.info())
print(df.isnull().sum())  # 缺失值检查

# 缺失值填补
df['income'].fillna(df['income'].median(), inplace=True)
df['education_level'].fillna('Unknown', inplace=True)

# 异常值处理（例：年龄不应小于18岁）
df = df[df['age'] >= 18]


'''
Step 2：特征工程
✅ 2.1 类别变量编码（one-hot 或 label）
'''
# 性别二值化
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# 教育、婚姻类别变量 One-Hot 编码
df = pd.get_dummies(df, columns=['marital_status', 'education_level'], drop_first=True)

'''
2.2 构造派生特征
'''
# 年收入除以额度（收入与授信比）
df['income_to_limit'] = df['income'] / df['credit_limit']

# 平均每月逾期次数（假设 payment_history 是 6个月总数）
df['avg_late_per_month'] = df['payment_history'] / 6

# 余额比授信额度（风险指标）
df['balance_ratio'] = df['balance'] / df['credit_limit']

'''
2.3 标准化（可选，逻辑回归等模型用得到）
'''
from sklearn.preprocessing import StandardScaler

scale_cols = ['age', 'income', 'credit_limit', 'balance', 'income_to_limit', 'avg_late_per_month', 'balance_ratio']
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

'''
2.4 拆分特征与目标值
'''
X = df.drop(columns=['default'])
y = df['default']


