"""
查看PCA分析结果
"""

import pandas as pd
import numpy as np

print("="*100)
print("PCA方法分析结果查看")
print("="*100)

# 读取结果
adjusted_exp = pd.read_csv('exposures_adjusted.csv', encoding='utf-8-sig', index_col=0)
raw_exp = pd.read_csv('exposures_raw.csv', encoding='utf-8-sig', index_col=0)
final_results = pd.read_csv('final_results_pca.csv', encoding='utf-8-sig')
pca_loadings = pd.read_csv('pca_loadings.csv', encoding='utf-8-sig', index_col=0)

print("\n【1】PCA主成分载荷（各资产在主成分上的权重）")
print("-"*100)
print(pca_loadings.round(4))

print("\n【2】原始暴露度矩阵（未经波动率调整）")
print("-"*100)
print(raw_exp.round(6))

print("\n【3】调整后暴露度矩阵（除以波动率）")
print("-"*100)
print(adjusted_exp.round(6))

print("\n【4】最终结果汇总")
print("-"*100)
print(final_results.to_string(index=False))

print("\n" + "="*100)
print("【分析】暴露度数值说明")
print("="*100)

print("""
当前结果显示暴露度数值很小（接近0），这可能是因为：

1. Lasso回归的正则化过强，导致投影后的宏观变量变化幅度很小
2. 标准化参数设置（标准差=0.1）使得宏观变量波动较小
3. 12个月累计收益率的标准化消除了大部分变异

建议调整：
- 调整Lasso的alpha参数（目前为0.01）
- 调整宏观变量标准化的标准差参数（目前为0.1）
- 或者跳过Lasso投影，直接使用标准化的宏观变量

让我检查一下各矩阵的统计特征...
""")

print("\n调整后暴露度的统计特征:")
print(adjusted_exp.describe().round(6))

print("\n各资产的暴露度范围:")
for idx in adjusted_exp.index:
    max_val = adjusted_exp.loc[idx].max()
    min_val = adjusted_exp.loc[idx].min()
    range_val = max_val - min_val
    print(f"{idx:10s}: 最大={max_val:10.6f}, 最小={min_val:10.6f}, 范围={range_val:10.6f}")

print("\n各宏观因子的暴露度范围:")
for col in adjusted_exp.columns:
    max_val = adjusted_exp[col].max()
    min_val = adjusted_exp[col].min()
    range_val = max_val - min_val
    print(f"{col:10s}: 最大={max_val:10.6f}, 最小={min_val:10.6f}, 范围={range_val:10.6f}")

# 尝试不使用阈值，直接排序
print("\n" + "="*100)
print("【重新排序】不使用阈值，按绝对暴露度排序")
print("="*100)

for col in adjusted_exp.columns:
    print(f"\n【{col}】")
    sorted_values = adjusted_exp[col].sort_values(ascending=False)
    for i, (asset, value) in enumerate(sorted_values.items(), 1):
        marker = "⬆️" if i <= 2 else ("⬇️" if i >= len(sorted_values) - 1 else "  ")
        print(f"  {marker} {i}. {asset:10s}: {value:10.6f}")

print("\n" + "="*100)
