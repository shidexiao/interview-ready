import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 模拟3类资产的历史收益率（股票、债券、商品）
np.random.seed(42)
returns = pd.DataFrame({
    'stock': np.random.normal(0.001, 0.02, 1000),
    'bond': np.random.normal(0.0005, 0.01, 1000),
    'commodity': np.random.normal(0.0002, 0.015, 1000)
})

# 计算协方差矩阵
cov_matrix = returns.cov() * 252  # 年化协方差

# 风险平价目标函数：最小化权重与风险贡献的差异
def risk_parity_objective(weights, cov_matrix):
    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    marginal_risk = cov_matrix @ weights / portfolio_vol  # 边际风险贡献
    risk_contributions = weights * marginal_risk
    target_contributions = portfolio_vol / cov_matrix.shape[0]  # 目标：均等风险贡献
    return np.sum((risk_contributions - target_contributions) ** 2)

# 约束条件：权重和为1，无做空
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1) for _ in range(3)]

# 初始猜测权重
init_weights = np.array([1/3, 1/3, 1/3])

# 优化求解
result = minimize(
    risk_parity_objective,
    init_weights,
    args=(cov_matrix),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# 输出最优权重
optimal_weights = result.x
print("Risk Parity Weights:",
      dict(zip(['stock', 'bond', 'commodity'], np.round(optimal_weights, 4))))

# 验证风险贡献
portfolio_vol = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
marginal_risk = cov_matrix @ optimal_weights / portfolio_vol
risk_contributions = optimal_weights * marginal_risk
print("Risk Contributions:", risk_contributions / portfolio_vol)  # 应接近33.3%