import pandas as pd
import numpy as np

# 模拟高频交易数据（假设为某股票1分钟的tick数据）
data = {
    'timestamp': pd.date_range('2023-01-01 09:30:00', periods=1000, freq='1s'),
    'price': np.cumsum(np.random.normal(0, 0.1, 1000)) + 100,  # 模拟价格序列
    'bid': np.random.uniform(99.9, 100.1, 1000),  # 模拟买一价
    'ask': np.random.uniform(100.1, 100.3, 1000),  # 模拟卖一价
    'volume': np.random.randint(1, 100, 1000)       # 模拟成交量
}
df = pd.DataFrame(data).set_index('timestamp')

# 1. 计算买卖价差（Bid-Ask Spread）作为流动性因子
df['spread'] = df['ask'] - df['bid']
df['spread_pct'] = df['spread'] / df['price'] * 100  # 相对价差百分比

# 2. 计算滚动波动率（5分钟窗口）
df['returns'] = np.log(df['price'] / df['price'].shift(1))  # 对数收益率
df['5min_volatility'] = df['returns'].rolling('5min').std() * np.sqrt(252 * 24 * 60)  # 年化波动率

# 结果展示
print(df[['price', 'spread_pct', '5min_volatility']].tail())

# 可视化
import matplotlib.pyplot as plt
df[['price', 'spread_pct']].plot(subplots=True, figsize=(12, 6))
plt.show()