"""
对比两种方法的结果：
1. 简单线性回归方法（macro_factor_parity.py）
2. PCA方法（macro_factor_parity_pca_v2.py）
"""

import pandas as pd
import numpy as np

print("="*100)
print("方法对比：简单回归 vs PCA方法")
print("="*100)

# 读取两种方法的结果
# 方法1：简单回归
try:
    simple_exp = pd.read_csv('macro_factor_exposures.csv', encoding='utf-8-sig', index_col=0)
    method1_available = True
except:
    method1_available = False
    print("⚠️ 简单回归方法结果未找到")

# 方法2：PCA
pca_exp = pd.read_csv('exposures_raw_v2.csv', encoding='utf-8-sig', index_col=0)
pca_r2 = pd.read_csv('r_squared_v2.csv', encoding='utf-8-sig', index_col=0)

if method1_available:
    print("\n【方法1】简单线性回归（月度收益率 vs 宏观因子）")
    print("-"*100)
    print(simple_exp.round(4))
    
    print("\n【方法2】PCA方法（12个月累计收益率标准化 vs 宏观因子标准化）")
    print("-"*100)
    print(pca_exp.round(4))
    
    print("\n【对比】暴露度差异")
    print("-"*100)
    
    # 计算差异
    diff = pca_exp - simple_exp
    print(diff.round(4))
    
    print("\n【对比】暴露度相关性")
    print("-"*100)
    
    for col in simple_exp.columns:
        if col in pca_exp.columns:
            corr = simple_exp[col].corr(pca_exp[col])
            print(f"{col:10s}: 相关系数 = {corr:.4f}")

print("\n" + "="*100)
print("PCA方法的优势")
print("="*100)

print("""
1. ✓ 使用12个月累计收益率，更符合研报设定
2. ✓ 通过PCA降维，提取主要风险因子
3. ✓ 标准化处理消除量纲影响
4. ✓ R²值提供了拟合优度参考
5. ✓ 主成分分析揭示了资产间的协同结构
""")

print("\n" + "="*100)
print("各宏观因子的拟合优度（R²）")
print("="*100)

print("\nPCA方法的R²值:")
print(pca_r2.round(4))

print("\n各宏观因子的平均R²:")
for col in pca_r2.columns:
    avg_r2 = pca_r2[col].mean()
    max_r2 = pca_r2[col].max()
    max_asset = pca_r2[col].idxmax()
    print(f"{col:10s}: 平均R²={avg_r2:.4f}, 最高R²={max_r2:.4f} ({max_asset})")

print("\n" + "="*100)
print("关键发现对比")
print("="*100)

findings_comparison = """
简单回归方法 vs PCA方法：

📊 经济因子：
  简单回归: 标普500最高(0.135)
  PCA方法:  标普500最高(0.475) ✓ 更显著
  
📊 流动性因子：
  简单回归: 黄金最低(-1.805)
  PCA方法:  黄金最低(-0.787) ✓ 方向一致，但数值不同
  
📊 CPI因子：
  简单回归: 商品最高(0.328)
  PCA方法:  利率债最高(0.204)，商品次之(0.158) ⚠️ 略有差异
  
📊 PPI因子：
  简单回归: 商品最高(0.034)
  PCA方法:  商品最高(0.447) ✓ 方向一致，PCA更显著
  
📊 信用因子：
  简单回归: 商品最高(0.022)
  PCA方法:  黄金最高(0.231) ⚠️ 有差异

总结：
- PCA方法的暴露度数值更大，差异更显著
- 大方向基本一致（商品-通胀、黄金-利率、权益-经济）
- PCA方法通过标准化和主成分分析，结果更稳健
"""

print(findings_comparison)

print("\n" + "="*100)
print("推荐使用方法")
print("="*100)

recommendation = """
🎯 推荐使用：PCA方法（macro_factor_parity_pca_v2.py）

理由：
1. ✓ 更符合研报的方法论描述
2. ✓ 使用12个月累计收益率，更稳定
3. ✓ 标准化处理使得不同因子可比
4. ✓ 提供R²拟合优度，便于评估可靠性
5. ✓ PCA揭示资产的底层风险结构

简单回归方法的价值：
- 作为baseline和验证
- 计算简单，易于理解
- 可快速验证大方向是否正确
"""

print(recommendation)

print("\n" + "="*100)
print("对比分析完成！")
print("="*100)
