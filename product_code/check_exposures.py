import pandas as pd

print("="*100)
print("检查暴露度数值")
print("="*100)

# 读取原始暴露度
raw_exp = pd.read_csv('exposures_raw_v2.csv', encoding='utf-8-sig', index_col=0)
print("\n原始暴露度矩阵（未经波动率调整）:")
print(raw_exp)

print("\n原始暴露度统计:")
print(raw_exp.describe())

# 读取调整后暴露度
adj_exp = pd.read_csv('exposures_adjusted_v2.csv', encoding='utf-8-sig', index_col=0)
print("\n调整后暴露度矩阵:")
print(adj_exp)

print("\n调整后暴露度统计:")
print(adj_exp.describe())

# 读取R²
r2 = pd.read_csv('r_squared_v2.csv', encoding='utf-8-sig', index_col=0)
print("\nR²值（拟合优度）:")
print(r2.round(4))
