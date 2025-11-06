"""
申银万国研报复刻：宏观因子平价分析
复刻研报第二部分：不同宏观变量对应暴露最高、最低的组合（静态数据）
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 读取数据
ret_data = pd.read_csv('ret_data.csv', index_col=0, parse_dates=True)
macro_data = pd.read_csv('macro_data.csv', index_col=0, parse_dates=True)

# 资产名称映射（排除REITs）
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
    'M0000545': '经济增长',  # 工业增加值当月同比
    'S0059749': '流动性',    # 10年期国债到期收益率
    'M0000612': 'CPI',       # CPI同比
    'M0001227': 'PPI',       # PPI同比
    'M5206730': '信用'       # 社融当月值
}

print("="*80)
print("宏观因子平价分析 - 复刻申银万国研报")
print("="*80)

# 数据预处理
print("\n1. 数据预处理...")

# 处理工业增加值：计算滚动3个月平均
if 'M0000545' in macro_data.columns:
    macro_data['M0000545_rolling'] = macro_data['M0000545'].rolling(window=3, min_periods=1).mean()
    macro_data['M0000545'] = macro_data['M0000545_rolling']
    macro_data.drop('M0000545_rolling', axis=1, inplace=True)

# 处理社融：计算累计12个月新增社融同比增速
if 'M5206730' in macro_data.columns:
    # 计算过去12个月累计值
    macro_data['M5206730_12m'] = macro_data['M5206730'].rolling(window=12).sum()
    # 计算同比增速
    macro_data['M5206730_yoy'] = macro_data['M5206730_12m'].pct_change(12) * 100
    macro_data['M5206730'] = macro_data['M5206730_yoy']
    macro_data.drop(['M5206730_12m', 'M5206730_yoy'], axis=1, inplace=True)

# 对齐数据时间范围
common_dates = ret_data.index.intersection(macro_data.index)
ret_data_aligned = ret_data.loc[common_dates]
macro_data_aligned = macro_data.loc[common_dates]

# 选择需要的宏观因子
macro_factors = ['M0000545', 'S0059749', 'M0000612', 'M0001227', 'M5206730']
macro_data_selected = macro_data_aligned[macro_factors].copy()

# 删除缺失值
valid_idx = macro_data_selected.dropna().index.intersection(ret_data_aligned.dropna().index)
ret_data_clean = ret_data_aligned.loc[valid_idx]
macro_data_clean = macro_data_selected.loc[valid_idx]

print(f"有效数据期数: {len(valid_idx)}")
print(f"数据时间范围: {valid_idx[0]} 到 {valid_idx[-1]}")
print(f"资产数量: {len(ret_data_clean.columns)}")
print(f"宏观因子数量: {len(macro_data_clean.columns)}")

# 2. 计算宏观因子暴露度
print("\n2. 计算各资产对宏观因子的暴露度（贝塔系数）...")
print("-"*80)

# 存储暴露度结果
exposures = pd.DataFrame(index=ret_data_clean.columns, columns=macro_factors)

# 对每个资产和每个宏观因子进行回归
for asset in ret_data_clean.columns:
    for factor in macro_factors:
        # 准备回归数据
        y = ret_data_clean[asset].values.reshape(-1, 1)
        X = macro_data_clean[factor].values.reshape(-1, 1)
        
        # 删除NaN
        mask = ~(np.isnan(y.flatten()) | np.isnan(X.flatten()))
        if mask.sum() > 10:  # 至少需要10个观测值
            y_clean = y[mask]
            X_clean = X[mask]
            
            # 线性回归
            model = LinearRegression()
            model.fit(X_clean, y_clean)
            
            exposures.loc[asset, factor] = model.coef_[0][0]
        else:
            exposures.loc[asset, factor] = np.nan

# 转换为数值类型
exposures = exposures.astype(float)

# 重命名列和行以便阅读
exposures_display = exposures.copy()
exposures_display.index = [asset_mapping.get(x, x) for x in exposures_display.index]
exposures_display.columns = [macro_mapping.get(x, x) for x in exposures_display.columns]

print("\n各资产对宏观因子的暴露度（贝塔系数）：")
print(exposures_display.round(4))

# 3. 找出对每个宏观因子暴露最高和最低的资产
print("\n" + "="*80)
print("3. 不同宏观变量对应暴露最高、最低的资产")
print("="*80)

results_summary = []

for factor in macro_factors:
    factor_name = macro_mapping.get(factor, factor)
    
    print(f"\n【{factor_name}】")
    print("-"*80)
    
    # 获取该因子的暴露度
    factor_exposure = exposures[factor].dropna()
    
    if len(factor_exposure) > 0:
        # 排序
        sorted_exposure = factor_exposure.sort_values(ascending=False)
        
        # 最高暴露
        max_asset = sorted_exposure.index[0]
        max_value = sorted_exposure.iloc[0]
        
        # 最低暴露
        min_asset = sorted_exposure.index[-1]
        min_value = sorted_exposure.iloc[-1]
        
        print(f"暴露度最高资产: {asset_mapping.get(max_asset, max_asset):8s} (暴露度: {max_value:8.4f})")
        print(f"暴露度最低资产: {asset_mapping.get(min_asset, min_asset):8s} (暴露度: {min_value:8.4f})")
        print(f"\n所有资产暴露度排序:")
        for i, (asset, exp) in enumerate(sorted_exposure.items(), 1):
            print(f"  {i}. {asset_mapping.get(asset, asset):8s}: {exp:8.4f}")
        
        results_summary.append({
            '宏观因子': factor_name,
            '最高暴露资产': asset_mapping.get(max_asset, max_asset),
            '最高暴露度': max_value,
            '最低暴露资产': asset_mapping.get(min_asset, min_asset),
            '最低暴露度': min_value
        })

# 4. 汇总结果
print("\n" + "="*80)
print("4. 结果汇总表")
print("="*80)

results_df = pd.DataFrame(results_summary)
print(results_df.to_string(index=False))

# 5. 保存结果
print("\n5. 保存结果...")

# 保存暴露度矩阵
exposures_display.to_csv('macro_factor_exposures.csv', encoding='utf-8-sig')
print("已保存: macro_factor_exposures.csv (暴露度矩阵)")

# 保存汇总结果
results_df.to_csv('macro_factor_summary.csv', index=False, encoding='utf-8-sig')
print("已保存: macro_factor_summary.csv (汇总结果)")

print("\n" + "="*80)
print("分析完成！")
print("="*80)

# 6. 额外分析：相关性矩阵
print("\n6. 额外分析：资产收益率与宏观因子的相关性")
print("-"*80)

correlation_matrix = pd.DataFrame(index=ret_data_clean.columns, columns=macro_factors)

for asset in ret_data_clean.columns:
    for factor in macro_factors:
        corr = ret_data_clean[asset].corr(macro_data_clean[factor])
        correlation_matrix.loc[asset, factor] = corr

correlation_matrix = correlation_matrix.astype(float)
correlation_matrix_display = correlation_matrix.copy()
correlation_matrix_display.index = [asset_mapping.get(x, x) for x in correlation_matrix_display.index]
correlation_matrix_display.columns = [macro_mapping.get(x, x) for x in correlation_matrix_display.columns]

print("\n相关性矩阵:")
print(correlation_matrix_display.round(4))

correlation_matrix_display.to_csv('macro_factor_correlations.csv', encoding='utf-8-sig')
print("\n已保存: macro_factor_correlations.csv (相关性矩阵)")

print("\n" + "="*80)
print("全部分析完成！请查看生成的CSV文件以获取详细结果。")
print("="*80)
