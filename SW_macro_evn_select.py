#package
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from WindPy import w
from Util_Fin.PCAanalysis import PCAAnalyzer
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# follow 20241118-上海申银万国证券研究所-数说资产配置系列之十一：盈利、情绪和需求预期：市场信息对宏观量化模型的修正
# follow 20250609-上海申银万国证券研究所-全天候策略再思考：多资产及权益内部的应用实践

w.start()

def lasso_projection(Y,X):
    common_vars = Y.index.intersection(X.index)
    
    Y = Y.loc[common_vars]
    X = X.loc[common_vars]
    print(f"\n对齐后的数据长度:{len(Y)}")

    #存储投影结果
    projected_Y = pd.DataFrame(index=common_vars,columns=Y.columns)
    results_summary = {}
    # 对每个宏观变量进行Lasso投影
    for macro_var in Y.columns:
        print(f"\n处理宏观变量: {macro_var}")
        y = Y[macro_var].values
        X_lasso = X.values
        lasso_cv = LassoCV(alphas=np.logspace(-4, 4, 1000), 
                           cv=10, 
                           random_state=100,
                           max_iter=1000)
        lasso_cv.fit(X_lasso, y)

        best_alpha = lasso_cv.alpha_
        coefficients = lasso_cv.coef_
        if np.sum(np.abs(coefficients)) > 0:
            projected_values = lasso_cv.predict(X_lasso)
        else:
            projected_values = y
        projected_Y[macro_var] = projected_values
        # 计算交叉验证分数
        cv_scores = cross_val_score(lasso_cv, X_lasso, y, cv=5, scoring='neg_mean_squared_error')
        mean_cv_score = -cv_scores.mean()
        results_summary[macro_var] = {
            'best_alpha': best_alpha,
            'coefficients': coefficients,
            'cv_mse': mean_cv_score,
            'non_zero_components': np.sum(np.abs(coefficients) > 1e-6)
        }

        # print(f"  最优alpha: {best_alpha:.6f}")
        # print(f"  交叉验证MSE: {mean_cv_score:.6f}")
        # print(f"  非零主成分数量: {results_summary[macro_var]['non_zero_components']}")
        # print(f"  系数: {coefficients}")
    return projected_Y, results_summary

def create_ranked_names_df(df):
    # 创建空DataFrame存放结果
    max_rank = len(df)
    dfB = pd.DataFrame(
        index=range(1, max_rank+1),
        columns=df.columns, 
    )
    dfB.index.name  = 'Rank'
    
    # 处理每个科目
    for factor in df.columns: 
        factor_data = df.copy()    
        factor_data['Name'] = factor_data.index  
        factor_data = factor_data.sort_values( 
            by=[factor, 'Name'],
            ascending=[False, True]
        )
        # 生成排名索引 
        factor_data.index  = range(1, len(factor_data)+1)
        
        # 填充到结果DataFrame
        dfB[factor] = factor_data['Name']
    
    return dfB

#1.资产选择
'''
stock:
沪深300 - 000300.SH
中证500 - 000905.SH
中证1000 - 000852.SH

跨境股票:
标普500 - SPX.GI

债券:
中债7-10年政策性金融债 - CBA08202.CS
中债1-3年国债 - 931552.CSI

商品或其他:
黄金(Au9999) - AU9999.SGE
大商所豆粕期货 - M.DCE
华夏中国交建REIT - 508018.SH #剔除
'''
investment_pool = ['000300.SH','000905.SH','000852.SH','SPX.GI','CBA08202.CS','931552.CSI','AU9999.SGE','M.DCE']
str = ','.join(investment_pool)
history_data=w.wsd(str,
                   "pct_chg", 
                   "2011-01-01", 
                   "2025-9-30",  
                   'Period=M',    
                   usedf=True) 
price_data = history_data[1]
price_data.to_csv('ret_data.csv')
ret_data = pd.read_csv('D:/program_learning/GD_security_working/asset_alloc/price_data/micro_research_price_data.csv',index_col=0,parse_dates=True)
ret_data.dropna(inplace=True)
ret_data = ret_data/100

cum12_ret_data= ret_data.rolling(window=12).apply(lambda x: (1+x).prod()-1).iloc[11:].loc[:"2021-01-01"]
cum12_ret_data_standardize = (cum12_ret_data-cum12_ret_data.mean())/cum12_ret_data.std()

pca_analyzer = PCAAnalyzer(standardize=True,n_components=6)
pca_analyzer.fit(cum12_ret_data_standardize)
pca_ret = pd.DataFrame(pca_analyzer.transform(),index=cum12_ret_data_standardize.index)
pca_ret.to_csv("pca_series.csv")

'''
获取宏观数据
经济:  工业增加值当月同比的滚动 3 个月平均  # M0000545-规模以上工业企业增加值当月同比
流动性: 10 年期国债到期收益率  # S0059749-中债10年期国债到期收益率 
通胀: CPI同比  # M0000612-CPI当月同比
通胀: PPI同比 # M0001227-PPI当月同比 
信用: 累计12个月新增社融同比增速 # M5206730 社融:当月值
'''
macro_data = w.edb("M0061675,M0000545,S0059749,M1001654,M0061676,M0000612,M0061677,M0001227,M5206730",
                   "2009-01-01",
                   "2025-10-27",
                   "Fill=Previous",
                   'Period=M',
                   usedf=True)
macro_data = macro_data[1]
macro_data.to_csv("macro_data.csv")
macro_data = pd.read_csv("D:/program_learning/GD_security_working/asset_alloc/price_data/macro_social_data.csv", index_col=0,parse_dates=True)
macro_data = macro_data.resample('M').last().dropna().loc[:'2021-01-01']

macro_data['新增社融'] = macro_data['M5206730']
macro_data['S0059749'] = 1/macro_data['S0059749']
macro_data['累计12个月新增社融'] = macro_data['新增社融'].rolling(12).sum()
macro_data['累计12个月新增社融同比增速'] = macro_data['累计12个月新增社融'].pct_change(12)
macro_data['工业增加值同比增速三月平均'] = macro_data['M0000545'].rolling(3).mean()

macro_data_cal = macro_data[['工业增加值同比增速三月平均','S0059749','M0000612','M0001227', '累计12个月新增社融同比增速']]
macro_data_cal.columns = ['工业增加值同比增速三月平均','10年期国债到期收益率','CPI同比','PPI同比','社融同比增速']
macro_data_cal = macro_data_cal.loc['2012-12-30':]
macro_data_cal.index = pca_ret.index
macro_data_cal.to_csv("macro_data_cal.csv")
#标准化
macro_data_cal_1 = ((macro_data_cal - macro_data_cal.mean()) / macro_data_cal.std())*0.1

#LASSO
print("%===正在就行LASSO回归")
projected_macro, results_summary = lasso_projection(macro_data_cal_1, pca_ret)
print("%===LASSO回归完成")
exposures_df = pd.DataFrame(index=cum12_ret_data_standardize.columns, columns=projected_macro.columns)
p_values = pd.DataFrame(index=cum12_ret_data_standardize.columns, columns=projected_macro.columns)
for asset in cum12_ret_data_standardize.columns:
    for macro_var in projected_macro.columns:
        Y = cum12_ret_data_standardize[asset].values
        X = projected_macro[macro_var].values
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        exposures_df.loc[asset, macro_var] = model.params[1]/ret_data[asset].std()
        # exposures_df.loc[asset, macro_var] = (np.linalg.inv(X.T@X)@X.T@Y)[1]/ret_data[asset].std()
        p_values.loc[asset, macro_var] = model.pvalues[1]

rankdf = create_ranked_names_df(exposures_df)
print("%===宏观暴露排名如下")
print(rankdf)

#2.策略回测s
Env = {'Growth_up':["SPX.GI","000300.SH"],
       'Growth_down':["931552.CSI","AU9999.SGE"],
       'Inflaction_up':['AU9999.SGE','931552.CSI'],
       'Inflaction_down':['000905.CSI','000300.CSI'],
       'fluent_up':['931552.CSI','CBA08202.CS'],
       'fluent_down':['SPX.GI']}
