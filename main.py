import numpy as np

# 假设有3个资产，每个资产的预期收益率和协方差矩阵如下：
returns = np.array([0.05, 0.1, 0.15])
cov_matrix = np.array([[0.1, 0.05, 0.03],
                       [0.05, 0.12, 0.08],
                       [0.03, 0.08, 0.15]])

# 定义目标函数，即投资组合的风险
def portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# 定义约束条件，即投资组合的权重之和为1
def constraint(weights):
    return np.sum(weights) - 1

# 使用优化算法求解最优投资组合
from scipy.optimize import minimize

# 设置初始权重
initial_weights = np.array([0.3, 0.3, 0.4])

# 设置约束条件
constraints = ({'type': 'eq', 'fun': constraint})

# 设置边界条件，即权重的取值范围为[0, 1]
bounds = [(0, 1)] * len(returns)

# 求解最优投资组合
result = minimize(portfolio_risk, initial_weights, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)

# 输出最优投资组合的权重
optimal_weights = result.x
print(optimal_weights)