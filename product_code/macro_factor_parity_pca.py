"""
申银万国研报复刻：宏观因子平价分析（PCA+Lasso方法）
严格按照研报方法：PCA主成分分析 + Lasso回归投影 + 波动率调整
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("申银万国研报复刻：宏观因子平价分析（PCA+Lasso方法）")
print("="*100)

# ================================
# 数据读取和预处理
# ================================
print("\n【步骤0】数据读取和预处理")
print("-"*100)

ret_data = pd.read_csv('ret_data.csv', index_col=0, parse_dates=True)
macro_data = pd.read_csv('macro_data.csv', index_col=0, parse_dates=True)

# 资产名称映射
asset_mapping = {
    '000300.SH': '沪深300',
    '000905.SH': '中证500',
    '000852.SH': '中证1000',
    'SPX.GI': '标普500',
    'CBA08202.CS': '信用债',
    '931552.CSI': '利率债',
    'AU9999.SGE': '黄金',
    'M.DCE': '商品'
}

# 宏观因子映射
macro_mapping = {
    'M0000545': '经济',
    'S0059749': '流动性',
    'M0000612': 'CPI',
    'M0001227': 'PPI',
    'M5206730': '信用'
}

# 处理工业增加值：滚动3个月平均
macro_data['M0000545'] = macro_data['M0000545'].rolling(window=3, min_periods=1).mean()

# 处理社融：累计12个月新增社融同比增速
macro_data['M5206730_12m'] = macro_data['M5206730'].rolling(window=12).sum()
macro_data['M5206730'] = macro_data['M5206730_12m'].pct_change(12) * 100
macro_data.drop('M5206730_12m', axis=1, inplace=True)

# 选择需要的宏观因子
macro_factors = ['M0000545', 'S0059749', 'M0000612', 'M0001227', 'M5206730']
macro_data_selected = macro_data[macro_factors].copy()

# 对齐数据
common_dates = ret_data.index.intersection(macro_data_selected.index)
ret_data = ret_data.loc[common_dates]
macro_data_selected = macro_data_selected.loc[common_dates]

# 删除缺失值
valid_idx = macro_data_selected.dropna().index.intersection(ret_data.dropna().index)
ret_data_clean = ret_data.loc[valid_idx]
macro_data_clean = macro_data_selected.loc[valid_idx]

print(f"✓ 有效数据期数: {len(valid_idx)}")
print(f"✓ 数据时间范围: {valid_idx[0]} 到 {valid_idx[-1]}")
print(f"✓ 资产数量: {len(ret_data_clean.columns)}")
print(f"✓ 宏观因子数量: {len(macro_data_clean.columns)}")

# ================================
# 步骤1：计算并标准化资产12个月累计收益率
# ================================
print("\n【步骤1】计算并标准化资产12个月累计收益率")
print("-"*100)

# 计算12个月累计收益率
ret_12m = pd.DataFrame(index=ret_data_clean.index, columns=ret_data_clean.columns)

for col in ret_data_clean.columns:
    # 将月度收益率转换为12个月累计收益率
    ret_12m[col] = (1 + ret_data_clean[col] / 100).rolling(window=12).apply(
        lambda x: (np.prod(1 + x) - 1) * 100, raw=True
    )

# 删除前12个月的缺失值
ret_12m = ret_12m.dropna()
macro_data_clean = macro_data_clean.loc[ret_12m.index]

print(f"✓ 12个月累计收益率计算完成")
print(f"✓ 有效数据期数（去除前12个月）: {len(ret_12m)}")

# 标准化收益率（z-score）
scaler_ret = StandardScaler()
ret_12m_scaled = pd.DataFrame(
    scaler_ret.fit_transform(ret_12m),
    index=ret_12m.index,
    columns=ret_12m.columns
)

print(f"✓ 资产收益率标准化完成")
print(f"\n标准化后的收益率统计:")
print(ret_12m_scaled.describe().round(4))

# ================================
# 步骤2：提取主成分（PCA）
# ================================
print("\n【步骤2】提取主成分（PCA，提取前6个）")
print("-"*100)

n_components = 6
pca = PCA(n_components=n_components)
PC_matrix = pca.fit_transform(ret_12m_scaled)

# 转换为DataFrame
PC_df = pd.DataFrame(
    PC_matrix,
    index=ret_12m_scaled.index,
    columns=[f'PC{i+1}' for i in range(n_components)]
)

print(f"✓ PCA分析完成，提取了{n_components}个主成分")
print(f"✓ 主成分矩阵维度: {PC_matrix.shape} (T×{n_components})")
print(f"\n各主成分解释的方差比例:")
for i, var_ratio in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var_ratio*100:.2f}%")
print(f"  累计解释: {pca.explained_variance_ratio_.sum()*100:.2f}%")

print(f"\n主成分载荷（各资产在主成分上的权重）:")
loadings = pd.DataFrame(
    pca.components_.T,
    index=ret_12m_scaled.columns,
    columns=[f'PC{i+1}' for i in range(n_components)]
)
loadings.index = [asset_mapping.get(x, x) for x in loadings.index]
print(loadings.round(4))

# ================================
# 步骤3：标准化宏观变量（均值0，标准差1，然后可选缩放）
# ================================
print("\n【步骤3】标准化宏观变量")
print("-"*100)

# 先做标准z-score标准化
scaler_macro = StandardScaler()
Macro_matrix = pd.DataFrame(
    scaler_macro.fit_transform(macro_data_clean),
    index=macro_data_clean.index,
    columns=macro_factors
)

print(f"✓ 宏观变量标准化完成")
print(f"\n标准化后的宏观变量统计:")
stats = Macro_matrix.describe().round(4)
stats.columns = [macro_mapping.get(x, x) for x in stats.columns]
print(stats)

# ================================
# 步骤4：Lasso回归投影
# ================================
print("\n【步骤4】Lasso回归投影（将宏观变量投影到主成分上）")
print("-"*100)

# 使用Lasso回归将每个宏观变量投影到主成分上
Macro_proj = pd.DataFrame(index=Macro_matrix.index, columns=macro_factors)
lasso_coefs = pd.DataFrame(index=[f'PC{i+1}' for i in range(n_components)], columns=macro_factors)

alpha_lasso = 0.001  # Lasso正则化参数，可调整（降低以减少正则化强度）

for macro_var in macro_factors:
    # Lasso回归: Macro_var = PC_matrix * coef
    y = Macro_matrix[macro_var].values.reshape(-1, 1)
    X = PC_matrix
    
    lasso = Lasso(alpha=alpha_lasso, max_iter=10000)
    lasso.fit(X, y)
    
    # 投影后的宏观变量
    Macro_proj[macro_var] = lasso.predict(X)
    
    # 记录系数
    lasso_coefs[macro_var] = lasso.coef_
    
    # 打印信息
    macro_name = macro_mapping.get(macro_var, macro_var)
    non_zero = np.sum(np.abs(lasso.coef_) > 1e-6)
    print(f"✓ {macro_name:8s}: 使用了{non_zero}个主成分, R²={lasso.score(X, y):.4f}")

print(f"\n各宏观变量在主成分上的Lasso系数:")
lasso_coefs_display = lasso_coefs.copy()
lasso_coefs_display.columns = [macro_mapping.get(x, x) for x in lasso_coefs_display.columns]
print(lasso_coefs_display.round(4))

# ================================
# 步骤5：一元线性回归求暴露度
# ================================
print("\n【步骤5】一元线性回归求暴露度")
print("-"*100)

# 暴露度矩阵 (N × 5)
exposures = pd.DataFrame(index=ret_12m_scaled.columns, columns=macro_factors)
r_squared = pd.DataFrame(index=ret_12m_scaled.columns, columns=macro_factors)

for asset in ret_12m_scaled.columns:
    for macro_var in macro_factors:
        # 回归: R_i = α + β_j * F_j + ε
        y = ret_12m_scaled[asset].values.reshape(-1, 1)
        X = Macro_proj[macro_var].values.reshape(-1, 1)
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        exposures.loc[asset, macro_var] = reg.coef_[0][0]
        r_squared.loc[asset, macro_var] = reg.score(X, y)

exposures = exposures.astype(float)
r_squared = r_squared.astype(float)

print(f"✓ 暴露度计算完成")
print(f"\n原始暴露度矩阵（未调整）:")
exposures_display = exposures.copy()
exposures_display.index = [asset_mapping.get(x, x) for x in exposures_display.index]
exposures_display.columns = [macro_mapping.get(x, x) for x in exposures_display.columns]
print(exposures_display.round(4))

# ================================
# 步骤6：波动率调整
# ================================
print("\n【步骤6】波动率调整（暴露度 / 波动率）")
print("-"*100)

# 计算12个月收益率的波动率
volatility = ret_12m.std()

print(f"各资产的12个月收益率波动率:")
vol_display = volatility.copy()
vol_display.index = [asset_mapping.get(x, x) for x in vol_display.index]
print(vol_display.round(4).to_string())

# 调整后的暴露度 = 原始暴露度 / 波动率
adjusted_exposures = exposures.copy()
for asset in adjusted_exposures.index:
    adjusted_exposures.loc[asset, :] = exposures.loc[asset, :] / volatility[asset]

print(f"\n✓ 波动率调整完成")
print(f"\n调整后的暴露度矩阵:")
adjusted_exposures_display = adjusted_exposures.copy()
adjusted_exposures_display.index = [asset_mapping.get(x, x) for x in adjusted_exposures_display.index]
adjusted_exposures_display.columns = [macro_mapping.get(x, x) for x in adjusted_exposures_display.columns]
print(adjusted_exposures_display.round(4))

# ================================
# 步骤7：选暴露最高/最低的资产
# ================================
print("\n【步骤7】选择暴露最高/最低的资产")
print("-"*100)

threshold = 0.0  # 不使用阈值过滤（设为0）
results = []

for macro_var in macro_factors:
    macro_name = macro_mapping.get(macro_var, macro_var)
    
    # 获取该因子的调整后暴露度
    exp_series = adjusted_exposures[macro_var].copy()
    
    # 排序（不过滤）
    sorted_exp = exp_series.sort_values(ascending=False)
    
    print(f"\n【{macro_name}】暴露度排序:")
    for i, (asset, value) in enumerate(sorted_exp.items(), 1):
        asset_name = asset_mapping.get(asset, asset)
        marker = "⬆️" if i <= 2 else ("⬇️" if i >= len(sorted_exp) - 1 else "  ")
        print(f"  {marker} {i}. {asset_name:10s}: {value:10.4f}")
    
    # 最高暴露的前2个
    top_2 = sorted_exp.head(2)
    top_assets = [asset_mapping.get(x, x) for x in top_2.index]
    top_values = top_2.values
    
    # 最低暴露的后2个
    bottom_2 = sorted_exp.tail(2)
    bottom_assets = [asset_mapping.get(x, x) for x in bottom_2.index]
    bottom_values = bottom_2.values
    
    results.append({
        '宏观变量': macro_name,
        '暴露最高资产': ', '.join(top_assets),
        '最高暴露度': f"{top_values[0]:.4f}, {top_values[1]:.4f}",
        '暴露最低资产': ', '.join(bottom_assets),
        '最低暴露度': f"{bottom_values[0]:.4f}, {bottom_values[1]:.4f}"
    })

# ================================
# 输出最终结果
# ================================
print("\n" + "="*100)
print("最终结果汇总（复刻研报表4）")
print("="*100)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# ================================
# 保存结果
# ================================
print("\n" + "="*100)
print("保存结果文件")
print("="*100)

# 保存各种矩阵
PC_df.to_csv('pca_components.csv', encoding='utf-8-sig')
print("✓ pca_components.csv - 主成分矩阵")

loadings.to_csv('pca_loadings.csv', encoding='utf-8-sig')
print("✓ pca_loadings.csv - 主成分载荷")

lasso_coefs_display.to_csv('lasso_coefficients.csv', encoding='utf-8-sig')
print("✓ lasso_coefficients.csv - Lasso回归系数")

exposures_display.to_csv('exposures_raw.csv', encoding='utf-8-sig')
print("✓ exposures_raw.csv - 原始暴露度矩阵")

adjusted_exposures_display.to_csv('exposures_adjusted.csv', encoding='utf-8-sig')
print("✓ exposures_adjusted.csv - 调整后暴露度矩阵")

results_df.to_csv('final_results_pca.csv', index=False, encoding='utf-8-sig')
print("✓ final_results_pca.csv - 最终结果汇总")

print("\n" + "="*100)
print("分析完成！")
print("="*100)
