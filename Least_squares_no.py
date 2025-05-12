import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# === Step 1: 读取 CSV 文件 ===
df_e1 = pd.read_csv(r"C:\UltraLight-VMUNet-Output\IDRiD_Output\merged-E1.csv")
df_e2 = pd.read_csv(r"C:\UltraLight-VMUNet-Output\IDRiD_Output\merged-E2.csv")
df_e3 = pd.read_csv(r"C:\UltraLight-VMUNet-Output\IDRiD_Output\merged-E3.csv")
df_label = pd.read_csv(r"C:\UltraLight-VMUNet-Output\IDRiD_Output\labeled.csv")
# === Step 2: 合并数据 ===
df_all = df_e1.merge(df_e2, on='ImageName', suffixes=('_E1', '_E2'))
df_all = df_all.merge(df_e3, on='ImageName')
df_all = df_all.merge(df_label, on='ImageName')
df_all.rename(columns={'Score': 'E3', 'Grade': 'Label'}, inplace=True)

# === Step 3: 映射标签为目标值 G（0-4 → 0.0-1.0） ===
def map_grade_to_value(grade):
    # 根据等级返回对应区间的随机值
    if grade == 0:
        return np.random.uniform(0.0, 0.1)
    elif grade == 1:
        return np.random.uniform(0.2, 0.3)
    elif grade == 2:
        return np.random.uniform(0.4, 0.6)
    elif grade == 3:
        return np.random.uniform(0.7, 0.9)
    elif grade == 4:
        return np.random.uniform(1.0, 1.2)  # 可以根据需要调整区间

df_all['G'] = df_all['Label'].apply(map_grade_to_value)

# === Step 4: 构造矩阵 X 和向量 G ===
X = df_all[['Score_E1', 'Score_E2', 'E3']].values
G = df_all['G'].values

# === Step 5: 定义目标函数（最小化 MSE） ===
# 添加 L2 正则化项，lambda_reg 为正则化强度
def objective(lambdas, lambda_reg=0.01):
    # 预测的目标值
    G_pred = X @ lambdas
    # 计算均方误差
    mse = mean_squared_error(G, G_pred)
    # 添加 L2 正则化项
    l2_reg = lambda_reg * np.sum(np.square(lambdas))  # L2正则化
    return mse + l2_reg  # 目标函数为 MSE + L2 正则化项

# === Step 6: 求解最小化问题 ===
initial_guess = np.array([1 / 3, 1 / 3, 1 / 3])  # 初始猜测
bounds = [(0, 1)] * 3  # 限制λ的范围

result = minimize(objective, initial_guess, bounds=bounds)

# === Step 7: 输出结果 ===
if result.success:
    lambdas = result.x
    print("最优系数 λ1, λ2, λ3:")
    print(f"λ1 = {lambdas[0]:.4f}")
    print(f"λ2 = {lambdas[1]:.4f}")
    print(f"λ3 = {lambdas[2]:.4f}")

    G_pred = X @ lambdas
    mse = mean_squared_error(G, G_pred)
    print(f"\n均方误差 MSE: {mse:.6f}")
else:
    print("优化失败：", result.message)
