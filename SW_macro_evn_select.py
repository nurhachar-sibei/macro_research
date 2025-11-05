#package
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from WindPy import w
# follow 20241118-上海申银万国证券研究所-数说资产配置系列之十一：盈利、情绪和需求预期：市场信息对宏观量化模型的修正
# follow 20250609-上海申银万国证券研究所-全天候策略再思考：多资产及权益内部的应用实践

w.start()

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
# investment_pool = ['000300.SH','000905.SH','000852.SH','SPX.GI','CBA08202.CS','931552.CSI','AU9999.SGE','M.DCE']
# str = ','.join(investment_pool)
# history_data=w.wsd(str,
#                    "pct_chg", 
#                    "2011-01-01", 
#                    "2025-9-30",  
#                    'Period=M',    
#                    usedf=True) 
# price_data = history_data[1]
# price_data.to_csv('D:/program_learning/GD_security_working/asset_alloc/price_data/micro_research_price_data.csv')
ret_data = pd.read_csv('D:/program_learning/GD_security_working/asset_alloc/price_data/micro_research_price_data.csv',index_col=0,parse_dates=True)
ret_data.dropna(inplace=True)
ret_data = ret_data/100

cum12_ret_data= ret_data.rolling(window=12).apply(lambda x: (1+x).prod()-1).iloc[11:].loc[:"2021-01-01"]

