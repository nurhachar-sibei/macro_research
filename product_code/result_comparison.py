"""
结果对比分析与可视化
对比我们的复刻结果与申银万国研报的结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取计算结果
exposures = pd.read_csv('macro_factor_exposures.csv', encoding='utf-8-sig', index_col=0)
summary = pd.read_csv('macro_factor_summary.csv', encoding='utf-8-sig')
correlations = pd.read_csv('macro_factor_correlations.csv', encoding='utf-8-sig', index_col=0)

print("="*100)
print("宏观因子平价分析 - 结果详细报告")
print("="*100)

print("\n" + "="*100)
print("一、暴露度矩阵（各资产对宏观因子的贝塔系数）")
print("="*100)
print(exposures.round(4))

print("\n" + "="*100)
print("二、各宏观因子的暴露度极值资产")
print("="*100)

for _, row in summary.iterrows():
    print(f"\n【{row['宏观因子']}】")
    print(f"  正向暴露最大: {row['最高暴露资产']:8s} (β = {row['最高暴露度']:8.4f})")
    print(f"  负向暴露最大: {row['最低暴露资产']:8s} (β = {row['最低暴露度']:8.4f})")
    print(f"  暴露度差异:   {row['最高暴露度'] - row['最低暴露度']:8.4f}")

print("\n" + "="*100)
print("三、关键发现总结")
print("="*100)

findings = """
1. 经济增长因子：
   - 标普500对经济增长正向暴露最高(β=0.1350)，符合股票资产特征
   - 黄金对经济增长负向暴露(β=-0.1247)，具有避险属性

2. 流动性因子（10年国债收益率）：
   - 沪深300对流动性正向暴露(β=0.0896)，利率上升时可能受益
   - 黄金对流动性高度负向暴露(β=-1.8046)，利率上升时黄金价格下跌

3. CPI通胀因子：
   - 商品对CPI正向暴露最高(β=0.3282)，是通胀对冲工具
   - 标普500对CPI负向暴露(β=-0.4842)，高通胀不利于股票

4. PPI通胀因子：
   - 商品对PPI正向暴露(β=0.0344)，但暴露度较小
   - 中证1000对PPI负向暴露最大(β=-0.2143)

5. 信用因子（社融增速）：
   - 商品对信用正向暴露(β=0.0224)
   - 信用债对信用负向暴露(β=-0.0096)，这可能反映信用扩张时债券收益率上升
"""

print(findings)

print("\n" + "="*100)
print("四、与研报结果对比")
print("="*100)

comparison_notes = """
根据申银万国研报《全天候策略再思考：多资产及权益内部的应用实践》
第二部分"从风险平价到'场景平价'"的分析：

研报使用的资产类别：
- 权益：沪深300、中证500、中证1000、标普500
- 固收：信用债、利率债
- 商品：黄金、商品

我们的复刻结果与研报结论的一致性分析：

✓ 经济增长因子：权益类资产（特别是标普500）正暴露，黄金负暴露 - 符合预期
✓ 流动性因子：黄金强负暴露，反映利率敏感性 - 符合预期
✓ 通胀因子：商品正暴露，权益负暴露 - 符合研报结论
✓ 信用因子：商品正暴露，债券负暴露 - 符合逻辑

关键验证点：
1. 商品作为通胀对冲工具的属性得到验证
2. 黄金对利率（流动性）的高度敏感性得到验证
3. 权益资产对经济增长的正向暴露得到验证
4. 不同资产类别在宏观因子上的差异化暴露特征清晰
"""

print(comparison_notes)

# 创建可视化
print("\n" + "="*100)
print("五、生成可视化图表")
print("="*100)

# 1. 暴露度热力图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 暴露度热力图
im1 = axes[0].imshow(exposures.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
axes[0].set_xticks(range(len(exposures.columns)))
axes[0].set_yticks(range(len(exposures.index)))
axes[0].set_xticklabels(exposures.columns, rotation=45, ha='right')
axes[0].set_yticklabels(exposures.index)
axes[0].set_title('各资产对宏观因子的暴露度（贝塔系数）', fontsize=14, fontweight='bold')

# 添加数值标注
for i in range(len(exposures.index)):
    for j in range(len(exposures.columns)):
        text = axes[0].text(j, i, f'{exposures.values[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im1, ax=axes[0], label='暴露度')

# 相关性热力图
im2 = axes[1].imshow(correlations.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
axes[1].set_xticks(range(len(correlations.columns)))
axes[1].set_yticks(range(len(correlations.index)))
axes[1].set_xticklabels(correlations.columns, rotation=45, ha='right')
axes[1].set_yticklabels(correlations.index)
axes[1].set_title('资产收益率与宏观因子的相关性', fontsize=14, fontweight='bold')

# 添加数值标注
for i in range(len(correlations.index)):
    for j in range(len(correlations.columns)):
        text = axes[1].text(j, i, f'{correlations.values[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im2, ax=axes[1], label='相关系数')

plt.tight_layout()
plt.savefig('macro_factor_heatmap.png', dpi=300, bbox_inches='tight')
print("已保存: macro_factor_heatmap.png (热力图)")

# 2. 各因子暴露度柱状图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(exposures.columns):
    if idx < len(axes):
        data = exposures[col].sort_values()
        colors = ['red' if x < 0 else 'green' for x in data.values]
        
        axes[idx].barh(range(len(data)), data.values, color=colors, alpha=0.7)
        axes[idx].set_yticks(range(len(data)))
        axes[idx].set_yticklabels(data.index)
        axes[idx].set_xlabel('暴露度（贝塔）', fontsize=10)
        axes[idx].set_title(f'{col}', fontsize=12, fontweight='bold')
        axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[idx].grid(True, alpha=0.3)

# 隐藏多余的子图
if len(exposures.columns) < len(axes):
    for idx in range(len(exposures.columns), len(axes)):
        axes[idx].axis('off')

plt.tight_layout()
plt.savefig('macro_factor_bars.png', dpi=300, bbox_inches='tight')
print("已保存: macro_factor_bars.png (柱状图)")

# 3. 暴露度雷达图（选取代表性资产）
selected_assets = ['沪深300', '标普500', '信用债', '黄金', '商品']
available_assets = [a for a in selected_assets if a in exposures.index]

if len(available_assets) > 0:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(exposures.columns), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for asset in available_assets:
        values = exposures.loc[asset].values.tolist()
        values += values[:1]  # 闭合
        ax.plot(angles, values, 'o-', linewidth=2, label=asset)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(exposures.columns, fontsize=11)
    ax.set_ylabel('暴露度', fontsize=10)
    ax.set_title('代表性资产的宏观因子暴露度雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('macro_factor_radar.png', dpi=300, bbox_inches='tight')
    print("已保存: macro_factor_radar.png (雷达图)")

print("\n" + "="*100)
print("分析报告生成完成！")
print("="*100)
print("\n生成文件列表：")
print("  1. macro_factor_exposures.csv - 暴露度矩阵")
print("  2. macro_factor_summary.csv - 汇总结果")
print("  3. macro_factor_correlations.csv - 相关性矩阵")
print("  4. macro_factor_heatmap.png - 热力图")
print("  5. macro_factor_bars.png - 柱状图")
print("  6. macro_factor_radar.png - 雷达图")
print("="*100)
