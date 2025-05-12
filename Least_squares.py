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

# === Step 3: 标签映射为 G 值 ===
def map_grade_to_value(grade):
    if grade == 0:
        return np.random.uniform(0.0, 0.1)
    elif grade == 1:
        return np.random.uniform(0.2, 0.3)
    elif grade == 2:
        return np.random.uniform(0.4, 0.6)
    elif grade == 3:
        return np.random.uniform(0.7, 0.9)
    elif grade == 4:
        return np.random.uniform(1.0, 1.2)

df_all['G'] = df_all['Label'].apply(map_grade_to_value)

# === Step 4: 构造矩阵 X 和目标向量 G ===
X = df_all[['Score_E1', 'Score_E2', 'E3']].values
G = df_all['G'].values

# === Step 5: 目标函数（含惩罚项） ===
def objective(lambdas, alpha=0.1):
    G_pred = X @ lambdas
    mse = mean_squared_error(G, G_pred)
    mean_lambda = np.mean(lambdas)
    var_penalty = np.mean((lambdas - mean_lambda) ** 2)
    return mse + alpha * var_penalty

# === Step 6: 约束和边界 ===
constraints = {'type': 'eq', 'fun': lambda lambdas: np.sum(lambdas) - 1}
bounds = [(0, None)] * 3  # 只允许非负

# === Step 7: 优化求解 ===
initial_guess = np.array([1/3, 1/3, 1/3])
result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# === Step 8: 输出结果 ===
if result.success:
    lambdas = result.x
    print("约束 λ1 + λ2 + λ3 = 1 且均匀化惩罚下的最优系数：")
    print(f"λ1 = {lambdas[0]:.4f}")
    print(f"λ2 = {lambdas[1]:.4f}")
    print(f"λ3 = {lambdas[2]:.4f}")
    print(f"λ总和 = {np.sum(lambdas):.4f}")

    G_pred = X @ lambdas
    mse = mean_squared_error(G, G_pred)
    print(f"\n均方误差 MSE: {mse:.6f}")
else:
    print("优化失败：", result.message)
