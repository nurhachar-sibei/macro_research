"""
生成最终结果总览表
"""
import pandas as pd

exp = pd.read_csv('exposures_raw_v2.csv', encoding='utf-8-sig', index_col=0)
r2 = pd.read_csv('r_squared_v2.csv', encoding='utf-8-sig', index_col=0)

print('='*100)
print('宏观因子平价分析 - 最终结果总览')
print('='*100)

print('\n各宏观因子的TOP暴露资产:')
print('-'*100)

for col in exp.columns:
    s = exp[col].sort_values(ascending=False)
    print(f'\n【{col}】')
    print(f'  ⬆️⬆️ 最高: {s.index[0]:10s} (β={s.values[0]:7.4f}, R²={r2.loc[s.index[0], col]:.4f})')
    print(f'  ⬆️   次高: {s.index[1]:10s} (β={s.values[1]:7.4f}, R²={r2.loc[s.index[1], col]:.4f})')
    print(f'  ⬇️   次低: {s.index[-2]:10s} (β={s.values[-2]:7.4f}, R²={r2.loc[s.index[-2], col]:.4f})')
    print(f'  ⬇️⬇️ 最低: {s.index[-1]:10s} (β={s.values[-1]:7.4f}, R²={r2.loc[s.index[-1], col]:.4f})')

print('\n' + '='*100)
print('项目完成！所有分析结果已生成。')
print('='*100)
print('\n📁 关键文件：')
print('  1. exposures_raw_v2.csv - 暴露度矩阵')
print('  2. exposure_ranking.csv - 排序汇总')
print('  3. pca_analysis_heatmap.png - 可视化')
print('  4. README_PCA.md - 详细文档')
print('  5. QUICK_START.md - 快速入门')
print('  6. PROJECT_SUMMARY.md - 项目总结')
print('='*100)
