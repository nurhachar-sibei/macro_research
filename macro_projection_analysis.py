"""
宏观变量投影分析
使用Lasso回归将标准化的宏观变量向主成分进行投影
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """
    读取宏观数据和主成分数据
    """
    # 读取宏观数据
    macro_data = pd.read_csv('macro_data_cal.csv', index_col=0)
    
    # 读取主成分数据
    pca_data = pd.read_csv('pca_series.csv', index_col=0)
    
    print("宏观数据形状:", macro_data.shape)
    print("主成分数据形状:", pca_data.shape)
    print("\n宏观数据列名:")
    print(macro_data.columns.tolist())
    
    return macro_data, pca_data

def standardize_macro_variables(macro_data):
    """
    对5个宏观变量进行标准化处理（均值0，标准差0.1）
    """
    # 提取5个宏观变量
    macro_variables = macro_data.copy()
    
    # 重命名列以便理解
    column_mapping = {
        macro_data.columns[0]: '经济',
        macro_data.columns[1]: '流动性', 
        macro_data.columns[2]: 'CPI',
        macro_data.columns[3]: 'PPI',
        macro_data.columns[4]: '信用'
    }
    macro_variables = macro_variables.rename(columns=column_mapping)
    
    print("原始宏观变量统计:")
    print(macro_variables.describe())
    
    # 标准化处理：均值0，标准差0.1
    standardized_data = pd.DataFrame(index=macro_variables.index)
    
    for col in macro_variables.columns:
        # 计算原始数据的均值和标准差
        mean_val = macro_variables[col].mean()
        std_val = macro_variables[col].std()
        
        # 标准化为均值0，标准差1，然后缩放到标准差0.1
        standardized_data[col] = ((macro_variables[col] - mean_val) / std_val) * 0.1
    
    print("\n标准化后宏观变量统计:")
    print(standardized_data.describe())
    
    return standardized_data

def lasso_projection(standardized_macro, pca_data):
    """
    使用Lasso回归将标准化的宏观变量向6个主成分进行投影
    """
    # 确保数据索引对齐
    common_index = standardized_macro.index.intersection(pca_data.index)
    macro_aligned = standardized_macro.loc[common_index]
    pca_aligned = pca_data.loc[common_index]
    
    print(f"\n对齐后的数据长度: {len(common_index)}")
    
    # 存储投影结果
    projected_macro = pd.DataFrame(index=common_index, columns=macro_aligned.columns)
    
    # 存储每个变量的最优alpha值和交叉验证分数
    results_summary = {}
    
    # 对每个宏观变量分别进行Lasso回归
    for macro_var in macro_aligned.columns:
        print(f"\n处理宏观变量: {macro_var}")
        
        # 目标变量（宏观变量）
        y = macro_aligned[macro_var].values
        
        # 特征变量（6个主成分）
        X = pca_aligned.values
        
        # 使用LassoCV进行交叉验证选择最优alpha
        # alphas参数设置一个范围，让算法自动选择最优值
        alphas = np.logspace(-4, 1, 50)  # 从0.0001到10的对数空间
        
        lasso_cv = LassoCV(
            alphas=alphas,
            cv=15,  # 15折交叉验证
            random_state=500,
            max_iter=2000
        )
        
        # 拟合模型
        lasso_cv.fit(X, y)
        
        # 获取最优alpha和对应的系数
        best_alpha = lasso_cv.alpha_
        coefficients = lasso_cv.coef_
        
        # 计算投影值（预测值）
        projected_values = lasso_cv.predict(X)
        projected_macro[macro_var] = projected_values
        
        # 计算交叉验证分数
        cv_scores = cross_val_score(lasso_cv, X, y, cv=5, scoring='neg_mean_squared_error')
        mean_cv_score = -cv_scores.mean()
        
        # 存储结果
        results_summary[macro_var] = {
            'best_alpha': best_alpha,
            'coefficients': coefficients,
            'cv_mse': mean_cv_score,
            'non_zero_components': np.sum(np.abs(coefficients) > 1e-6)
        }
        
        print(f"  最优alpha: {best_alpha:.6f}")
        print(f"  交叉验证MSE: {mean_cv_score:.6f}")
        print(f"  非零主成分数量: {results_summary[macro_var]['non_zero_components']}")
        print(f"  系数: {coefficients}")
    
    return projected_macro, results_summary

def save_results(projected_macro, results_summary):
    """
    保存投影结果和分析摘要
    """
    # 保存投影后的宏观变量矩阵F̂
    projected_macro.to_csv('projected_macro_variables.csv')
    print(f"\n投影后的宏观变量矩阵已保存到: projected_macro_variables.csv")
    
    # 保存分析摘要
    summary_df = pd.DataFrame(results_summary).T
    summary_df.to_csv('lasso_analysis_summary.csv')
    print(f"Lasso分析摘要已保存到: lasso_analysis_summary.csv")
    
    # 打印最终结果统计
    print("\n投影后宏观变量统计:")
    print(projected_macro.describe())
    
    print("\nLasso回归分析摘要:")
    for var, summary in results_summary.items():
        print(f"{var}:")
        print(f"  最优正则化参数: {summary['best_alpha']:.6f}")
        print(f"  交叉验证MSE: {summary['cv_mse']:.6f}")
        print(f"  选择的主成分数量: {summary['non_zero_components']}/6")

def main():
    """
    主函数：执行完整的宏观变量投影分析流程
    """
    print("=== 宏观变量投影分析 ===")
    
    # 1. 读取数据
    print("\n1. 读取数据...")
    macro_data, pca_data = load_data()
    
    # 2. 标准化宏观变量
    print("\n2. 标准化宏观变量...")
    standardized_macro = standardize_macro_variables(macro_data)
    
    # 3. Lasso回归投影
    print("\n3. 使用Lasso回归进行投影...")
    projected_macro, results_summary = lasso_projection(standardized_macro, pca_data)
    
    # 4. 保存结果
    print("\n4. 保存结果...")
    save_results(projected_macro, results_summary)
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()