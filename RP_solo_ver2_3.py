import importlib
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib import style

import scipy.optimize as sco
from tqdm import  tqdm#看进度条

from sklearn.decomposition import PCA
import tkinter as tk  
from tkinter import ttk  

from tabulate import tabulate
from arch import arch_model
from pypfopt.risk_models  import semicovariance

from sklearn.covariance import LedoitWolf
from scipy import stats

import logging
from datetime import datetime

import traceback

from Util_Fin import Wind_util #wind数据读取模块
from Util_Fin import Volatility_util #读取波动率模块
from Util_Fin import Eval_util  #评价指标模块
from Util_Fin import Position_util #仓位模块
from Util_Fin import logger_util #日志模块

#日志模块
# date_current = datetime.now().strftime('%Y%m%d')
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(f'./log/RP_model_{date_current}.log', encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)
logger = logger_util.setup_logger('RP_model')

# #数据导入
# code_list = ['CBA00301.CS','000300.SH','AU9999.SGE'] #'AU9999.SGE'
# start_date = '2003-01-01'
# end_date = '2025-10-14'
# price_df = Wind_util.get_hfq_price(code_list,start_date,end_date)
# price_df.to_excel("price.xlsx")
# #数据更新
# # logger.info('将数据更新至最新日期')
# # Wind_util.update_hfq_price('./price_data/price.xlsx')
# # logger.info('数据更新完成')
# #数据读取
# logger.info("数据读取")
# price_df = Wind_util.get_workspace_data('./price_data/price.xlsx',code_list,start_date)
# ret_ = Wind_util.get_return_df(price_df).iloc[:-1]
# logger.info("数据读取完成")

#模型搭建
class RPmodel():

    def __init__(self,
                 #基本数据
                 ret_,
                 #回测参数
                 start_date='2021-07-01',
                 end_date='2025-06-30',
                 change_time_delta=5,
                 cal_windows=120,
                 #模型选择参数
                 cov_matrix_method = 'EWMA_SEMI',#协方差选择
                 risk_alloc = 'EWRCP', #'EWRCP'纯正的风险平价-等权分配风险 ，'SSWRCP' 按夏普率分配风险
                 risk_budget_objective='naive_risk_parity',
                 #优化器参数
                 optimizer = 'SLSQP',#优化器选择
                 optimization_initial_position = [0.03,0.03,0.02,0.07,0.05,0.04,0.76], 
                 optimization_montca = [False,1], #是否起始点随机，重复模拟次数 False时建议选1加快速度
                 #杠杆参数
                 leverage_switch = 0, #0是不增加杠杆，1是事前杠杆模式一，2是事前杠杆模式二
                 leverage_post_ratio = 1, #整体杠杆缩放
                 frequency_calcov = 252
                ):
        #基础信息导入
        self.start_date = start_date#回测开始区间
        self.end_date =end_date#回测结束区间
        self.change_time_delta = change_time_delta#调仓周期
        self.total_df=ret_ #收益率数据 - 总表格
        self.cal_windows = cal_windows#计算指标所使用的历史数据周期
        self.stock_names = self.total_df.columns.tolist() #资产名称
        self.noa = self.total_df.shape[1] #资产数量
        self.cov_matrix_method = cov_matrix_method #协方差计算方式，参考协方差相关函数
        self.frequency_calcov = frequency_calcov #波动率的频率
        self.optimizer = optimizer #优化器选择 参考sco官方函数
        self.optimization_initial_position = optimization_initial_position #指定起始点时所采用的向量
                                                #不建议一开始使用，可以考虑随机排起始点后找到较优点再使用
        self.optimization_montca = optimization_montca[0] #是否选择随机的起始点开始 
        self.montca_count = optimization_montca[1] #起始点选择次数
        self.risk_alloc = risk_alloc #'EWRCP'纯正的风险平价-等权分配风险 ，'SSWRCP' 按夏普率分配风险
        self.risk_budget_objective = risk_budget_objective #优化目标函数 RP下有naive,pca,lasso 
        
        #杠杆设置
        self.leverage_switch = leverage_switch
        self.fix_income_asset = ['CBA00301.CS']
        self.leverage_cost = {"equity":0.05,'fix_income':0.025}
        if self.leverage_switch==1:
            self.leverage_ratios = self.leverage_ratio_cal()
        elif self.leverage_switch ==0:
            self.leverage_ratios = {key: 1 for key in self.stock_names}
        elif self.leverage_switch ==2:
            leverage_setting = [3,1]
            self.leverage_ratios = {self.stock_names[i]:leverage_setting[i] for i in range(len(self.stock_names))}
        self.leverage_ratios = {key:value for key,value in self.leverage_ratios.items() }
        self.leverage_post_ratios = {key:1*leverage_post_ratio for key in self.stock_names}
        
        #中间变量
        self.cal_df = pd.DataFrame() #中间变量 计算模型相关指标所使用的历史收益率数据 -> 每次调仓日均会发生动态变动 
        self.cal_cov_matrix = 0#中间变量 计算期的协方差

        #存储器（备用）
        self.std_set = [] #波动率储存器
        self.weight_set = [] #权重储存器
        self.res_set = {} #单期结果储存器，包含：时间，权重，协方差，波动率，optv
        
        # 回测相关信息
        #bench_mark设定
        self.benchmark = {'801010':[0.8,0.1,0.1]}
        #持仓日信息获取
        # self.position_get(1,1)
          #获取结果 self.position_df,self.change_position_df,self.change_position_date
        #回测结果
        # self.Backtest()
          #获取结果 self.result_rp,self.result_benchmark 
        # #回测评价指标
        # self.get_eval()
          #获取结果 self.eval_df
        self.ewma_lambda = 0.94
          

    def position_get(self,M,D):
        self.position_class = Position_util.Position_info(self.total_df,self.start_date,self.end_date,self.change_time_delta,M,D)
        try:
            self.position_df,self.change_position_df,self.change_position_date = self.position_class.position_information()
            if len(self.change_position_date)>=3:
                logger.info(f"="*50+"\n"+f"读取调仓日成功，调仓日期:\ndate_example:{self.change_position_date[0]}\ndate_example:{self.change_position_date[1]}\ndate_example:{self.change_position_date[2]}..."+"\n"+"="*50)
            else:
                logger.info(f"="*50+"\n"+f"读取调仓日成功，调仓日期:\ndate_example:{self.change_position_date}+"+"\n"+"="*50)
        except Exception as e:
            logger.error(f"调仓日读取失败，请确认输入调仓参数是否正确:{e}")
            print(traceback.format_exc())

    ###Part1:RP公式计算
    ##A.普通RP
    def calculate_risk_contribution(self,weight,cov_matrix):
        weight = np.matrix(weight)
        #先计算RC[N*1]的值
        #portfolio的sigma: sqrt(w'Σw)
        sigma = np.sqrt(weight*cov_matrix*weight.T)
        #边际风险贡献 MRC [N*1]
        MRC = cov_matrix*weight.T/sigma
        #风险贡献RC 
        RC = np.multiply(MRC,weight.T)
        return RC
    #1.sse
    #利用RC最小化资产之间的风险贡献差，定义优化问题的目标函数
    def naive_risk_parity(self,weight,parameters):
        '''
        weight:待求解的资产权重
        parameters:参数列表
             [0]:协方差矩阵
             [1]:风险平价下的目标风险贡献度向量
        '''
        weight = np.matrix(weight)
        cov_matrix = parameters[0]
        RC_target_ratio = parameters[1]
        #RC_target为风险平价下的目标风险贡献，一旦参数传递以后，RC_target就是一个常数，不随迭代而改变
        portfolio_sigma = np.sqrt(weight*cov_matrix*weight.T) # 组合波动率
        RC_target = np.asmatrix(np.multiply(portfolio_sigma,RC_target_ratio))# 目标风险贡献
        # RC_real是 每次迭代以后最新的真实风险贡献，随迭代而改变
        RC_real = self.calculate_risk_contribution(weight,cov_matrix)
        SSE = sum(np.square(RC_real-RC_target.T))[0,0]
        return SSE
    #B:PCA RP
    #2.1rc
    def calculate_risk_contribution_pca(self,weight,cov_matrix):
        '''
        pca RC计算
        '''
        weight = np.matrix(weight)
        sigma = np.sqrt(weight*cov_matrix*weight.T)
        #PCA分解
        a = E*weight.T
        b = E*(cov_matrix*weight.T)
        RC=np.multiply(a,b)/sigma
        return RC
    #2.2sse
    def pca_risk_parity(self,weight,parameters):
        '''
        weight:待求解的资产权重
        parameters:参数列表
             [0]:协方差矩阵
             [1]:风险平价下的目标风险贡献度向量
        '''
        cov_matrix = parameters[0]
        RC_target_ratio = parameters[1]
        weight = np.matrix(weight)
        portfolio_sigma = np.sqrt(weight*cov_matrix*weight.T)
        RC_target = np.asmatrix(np.multiply(portfolio_sigma,RC_target_ratio))
        RC_real = self.calculate_risk_contribution_pca(weight,cov_matrix)
        SSE = sum(np.square(RC_real-RC_target.T))[0,0]
        return SSE
    #C:Lasso_risk_parity
    def lasso_risk_parity(self,weight,parameters):
        '''
        weight:待求解的资产权重
        parameters:参数列表
             [0]:协方差矩阵
             [1]:风险平价下的目标风险贡献度向量
        '''
        weight = np.matrix(weight)
        cov_matrix = parameters[0]
        RC_target_ratio = parameters[1]
        #RC_target为风险平价下的目标风险贡献，一旦参数传递以后，RC_target就是一个常数，不随迭代而改变
        portfolio_sigma = np.sqrt(weight*cov_matrix*weight.T) # 组合波动率
        RC_target = np.asmatrix(np.multiply(portfolio_sigma,RC_target_ratio))# 目标风险贡献
        # RC_real是 每次迭代以后最新的真实风险贡献，随迭代而改变
        RC_real = self.calculate_risk_contribution(weight,cov_matrix)
        SSE = sum(np.square(RC_real-RC_target.T)+np.fabs(weight)*6)[0,0]
        return SSE
    
    ##随机起点设置
    def random_initial(self):
        random_points = np.random.uniform(0,  1, self.noa - 1)
        random_points.sort() 
        points = np.concatenate([[0],  random_points, [1]])
        values = np.diff(points) 
        np.random.shuffle(values)
        return(values)
    ##杠杆计算
    #杠杆和资金成本部分需要在实践中进行改进
    #事前杠杆
    def leverage_ratio_cal(self):
        cal_df = self.total_df
        #找到当前时期内回报率最高的资产
        cumulative_returns = (1+cal_df).cumprod()-1
        final_returns = cumulative_returns.iloc[-1]
        #遭到汇报最高的资产
        max_return_asset = final_returns.idxmax()
        max_return = final_returns[max_return_asset]
        leverage_ratios = {}
        for asset in cal_df.columns:
            asset_return = final_returns[asset]
            if asset_return > 0:
                leverage_ratio = max_return / asset_return
            else:
                leverage_ratio = 1  # 如果资产收益为负，设置杠杆为1
            leverage_ratios[asset] = leverage_ratio
        return leverage_ratios
    #事后杠杆
    def leverage_model_1_ret_df(self,ret_,leverage_ratios):
        hold_df = ret_
        leveraged_returns_df = pd.DataFrame()
        for asset in hold_df.columns:
            leverage_ratio = leverage_ratios[asset]
            if asset in self.fix_income_asset:
                annual_cost=0.025
            else:
                annual_cost = 0.05
            daily_cost = annual_cost/252
            
            leveraged_returns_df[asset]= hold_df[asset]*leverage_ratio -daily_cost*(leverage_ratio-1)
        return leveraged_returns_df
    #扣除资金成本
    def cash_cost_div_ret_df(self,ret_):
        hold_df=ret_
        ccd_returns_df = pd.DataFrame()
        for asset in hold_df.columns:
            if asset in self.fix_income_asset:
                annual_cost=0.025
            else:
                annual_cost = 0.05*0.15
            daily_cost = annual_cost/252
            
            ccd_returns_df[asset]= hold_df[asset] -daily_cost
        return ccd_returns_df 

    
    ###Part2:单期RP计算
    def RP(self,i,cov_method='ALL',risk_budget_objective='naive_risk_parity',risk_alloc = 'EWRCP'):
        ret_df =self.total_df
        p_d = self.change_position_date[i]
        try:
            next_p_d = self.change_position_date[i+1]
        except:
            next_p_d = self.end_date
        self.cal_df = ret_df.loc[:p_d].iloc[-self.cal_windows-1:-1]
        self.cal_df = self.leverage_model_1_ret_df(self.cal_df,self.leverage_ratios)
        def calculate_period_return(daily_returns):  
            return (1 + daily_returns).prod() - 1 
        if self.frequency_calcov == 252:
            self.cal_df = self.cal_df #modify
        elif self.frequency_calcov == 52:
            self.cal_df = self.cal_df.resample('W-FRI').apply(calculate_period_return)
        elif self.frequency_calcov == 12:
            self.cal_df = self.cal_df.resample('M').apply(calculate_period_return)
        elif self.frequency_calcov == 4:
            self.cal_df = self.cal_df.resample('Q').apply(calculate_period_return)

        cov_type = Volatility_util.Cov_Matrix(self.cal_df,cov_method)
        asset_cov = np.matrix(cov_type.calculate_cal_cov_matrix(frequency=self.frequency_calcov,lambda_=self.ewma_lambda))
        
        hold_df_original = ret_df.loc[p_d:next_p_d].iloc[:-1]
        hold_df = self.leverage_model_1_ret_df(hold_df_original,self.leverage_ratios)
        hold_df = self.leverage_model_1_ret_df(hold_df,self.leverage_post_ratios)
        rp_mode = {'naive_risk_parity':self.naive_risk_parity,
                   'pca_risk_parity':self.pca_risk_parity,
                   'lasso_risk_parity':self.lasso_risk_parity}
        risk_budget_objective = rp_mode[risk_budget_objective]
        
        num = self.noa

        x0 = np.array(self.optimization_initial_position)
        bounds = tuple((0,1) for _ in range(num)) #取值范围（0，1）
        #风险平价下每个资产的目标风险贡献度相等
        if risk_alloc=='EWRCP':
            RC_set_ratio = np.array([1.0/num for _ in range(num)])
        elif risk_alloc == 'SSWRCP':
            mu_ = self.cal_df.mean()*self.frequency_calcov
            std_ = self.cal_df.std()*np.sqrt(self.frequency_calcov)
            sharp_ratio = mu_/std_
            sharp_ratio_squre = sharp_ratio**2
            ssw = sharp_ratio_squre/sharp_ratio_squre.sum()
            RC_set_ratio = np.array(ssw)
            
            
            
        res_dict = {}
        #以随机起始点的方式开始找到最优点，最后得到是所有随机起始点计算结果中SSE最好的那个
        for i in range(self.montca_count):
            if self.optimization_montca == False:
                x0 = x0
            else:
                x0 = self.random_initial()
            #约束条件：
            #权重为1
            cons_1 = ({"type":"eq","fun": lambda x: sum(x) - 1},)
            optv = sco.minimize(risk_budget_objective,
                                x0,
                                args=[asset_cov,RC_set_ratio],
                                method=self.optimizer,
                                bounds=bounds,
                                constraints=cons_1)
            weight_rp = np.matrix(optv['x']).T
            res_ = {'optv':optv,
                    'RC_set_ratio':RC_set_ratio,
                    "cov_matrix_method":cov_method,
                    "weight_rp":weight_rp,
                    "SSE":abs(optv['fun']),
                    "portfolio_sigma":np.sqrt(weight_rp.T*asset_cov*weight_rp),
                    "asset_cov":asset_cov
                    }
            res_dict[i]=res_
        best_SSE_id = min(res_dict.items(),  key=lambda x: x[1]['SSE'])[0]
        res_ = res_dict[best_SSE_id]
        optv = res_['optv']
        weight_rp = res_['weight_rp']
        return p_d,weight_rp,hold_df,hold_df_original,optv,res_
    
    ###Part3:多期RP_test
    ##banchmark设计
    #benchmark_重置
    def benchmark_default(self):
        print("当前benchmark设置为:")
        print(self.benchmark)
        judge = input("是否重置benchmark(Y/N):")
        if judge=='Y':
            self.benchmark = {}
            benchmark_name = input('请输入基准名称')
            benchmark_weight = input('请直接输入基准权重,用空格分隔：')
            self.benchmark[benchmark_name] = [float(item) for item in benchmark_weight.split()]
            print("设置成功")
        else:
            print("benchmark不进行重置")
    #可以增加多个benchmark用于比较
    def benchmark_add(self):
        print("当前benchmark设置为:")
        print(self.benchmark)
        benchmark_name = input('请输入基准名称')
        benchmark_weight = input('请直接输入基准权重,用空格分隔：')
        self.benchmark[benchmark_name] = [float(item) for item in benchmark_weight.split()]
        print("添加成功")
    #删除benchmark
    def benchmark_del(self):
        print("当前benchmark设置为:")
        print(self.benchmark)
        benchmark_name = input('请输入要删除的基准名称')
        del self.benchmark[benchmark_name]
        print("删除成功")

    ##动态权重计算模块
    def calculate_dynamic_weight(self,hold_df,init_weight,init_date):
        '''
        计算因为价格变动而产生的动态权重
        hold_df: 持有期内所有资产的收益率表格
        init_weight:第一天所持有的权重
        init_date:第一天所对应的日期
        '''
        weights_df = pd.DataFrame(index=hold_df.index,columns=hold_df.columns).fillna(0)
        weights_df.loc[init_date] = init_weight.T.tolist()[0]
        prev_weights = weights_df.loc[init_date].values
        for i in range(1,len(weights_df)):
            daily_ret = hold_df.iloc[i].values
            #计算组合总收益率
            port_return = np.dot(prev_weights,daily_ret)
            new_weights = prev_weights*(1+daily_ret)/(1+port_return)
            weights_df.iloc[i]  = new_weights
            prev_weights = new_weights
        return weights_df

    ##回测部分     
    def Backtest(self):
        change_position_date = self.change_position_date

        position_df_original = self.position_df
        position_df = self.leverage_model_1_ret_df(position_df_original,self.leverage_ratios)
        position_df = self.leverage_model_1_ret_df(position_df,self.leverage_post_ratios)
        position_df_cash_remove = self.cash_cost_div_ret_df(position_df)
        

        cov_matrix_method = self.cov_matrix_method
        total_weights_df_benchmark= {}
        for i in range(len(change_position_date)):
            p_d,rp_weight,hold_df,hold_df_original,optv,res_ = self.RP(i,
                                                     cov_method=self.cov_matrix_method,
                                                     risk_budget_objective=self.risk_budget_objective,
                                                     risk_alloc=self.risk_alloc)
            logger.info(f"回测，日期为：{p_d}，产品为：{self.stock_names}，权重为：{rp_weight.T.tolist()[0]}")
            self.res_set[p_d]=res_

            def weight_adjust_ret(position_df,total_weights_df):
                position_df = position_df[position_df.index.isin(total_weights_df.index)]
                ret_adjust = position_df*total_weights_df
                return ret_adjust
            #策略输出权重和策略收益率净值
            weights_df = self.calculate_dynamic_weight(hold_df,rp_weight,p_d)
            if i == 0:
                total_weights_df_rp = weights_df
            else:
                total_weights_df_rp = pd.concat([total_weights_df_rp,weights_df])

            asset_rp_ret = weight_adjust_ret(position_df,total_weights_df_rp)
            portfolio_rp = asset_rp_ret.sum(axis=1)
            rp_pv = (portfolio_rp+1).cumprod()
            rp_dict = {"rp_weight_df":total_weights_df_rp,
                       "rp_asset_ret":asset_rp_ret,
                       "rp_portfolio_ret":portfolio_rp,
                       "rp_pv":rp_pv}

            #benchmark动态权重和相关净值收益率
            benchmark_results = {}
            if len(self.benchmark) == 0:
                print("未找到benchmark")
                print("跳过相关计算")
            else:
                try:
                    for benchmark_name,benchmark_weight in self.benchmark.items():
                        benchmark_weight = np.matrix([benchmark_weight]).T
                        weights_df_benchmark = self.calculate_dynamic_weight(hold_df_original,benchmark_weight,p_d)
                        if i == 0:
                            total_weights_df_benchmark[benchmark_name] = weights_df_benchmark
                        else:
                            total_weights_df_benchmark[benchmark_name] = pd.concat([total_weights_df_benchmark[benchmark_name],weights_df_benchmark])
                        #净值对比
                        asset_benchmark_ret = weight_adjust_ret(position_df_original,total_weights_df_benchmark[benchmark_name])
                        portfolio_benchmark = asset_benchmark_ret.sum(axis=1)
                        _benchmark_pv = (portfolio_benchmark+1).cumprod()
                        _benchmark_dict = {benchmark_name+"_weight_df":total_weights_df_benchmark[benchmark_name],  
                                benchmark_name+"_asset_ret":asset_benchmark_ret,
                                benchmark_name+"_portfolio_ret":portfolio_benchmark,
                                benchmark_name+"_pv":_benchmark_pv}
                        benchmark_results[benchmark_name] = _benchmark_dict
                except Exception as e:
                    print(f'benchmark设定有误,错误信息为{e},请检查benchmark长度以及其他信息')



        
        # asset_rp_ret_cash_cost_remove = weight_adjust_ret(position_df_cash_remove,total_weights_df_rp)
        # portfolio_rp_ccr = asset_rp_ret_cash_cost_remove.sum(axis=1)
        # rp_ccr_pv = (portfolio_rp_ccr+1).cumprod()
        # rp_ccr_dict = {"rp_weight_df":total_weights_df_rp,
        #            "rp_ccr_asset_ret":asset_rp_ret_cash_cost_remove,
        #            "rp_ccr_portfolio_ret":portfolio_rp_ccr,
        #            "rp_ccr_pv":rp_ccr_pv}
        
        self.result_rp = rp_dict
        self.result_benchmark = benchmark_results
        logger.info(f"回测完成")
        # return rp_dict,benchmark_results,rp_ccr_dict
    
    ###Part4:作图评价部分
    ##A.作图部分
    #权重变动
    def plot_weight_change(self,df):
        df001 = df
        labels = df001.columns.to_list()
        fig, ax = plt.subplots(figsize=(10,5))
        #variable_name = list(dict(df=df).keys())[0]
        #variable_name = variable_name.split("_")[-1]
        plt.title("RP weight time series")
        plt.yticks(np.arange(0, 1, step=0.2))
        ax.stackplot(df001.index, df001.T,baseline='zero',labels=labels)
        ax.legend(loc='upper left')
        plt.show()    
    #净值曲线绘制
    def plot_pv(self,pv_dict,label_name=['8020_pv','9010_pv','rp_pv']):
        sns.set(style="whitegrid")
        plt.figure(figsize = (15,6))
        # sns.lineplot(data=pd.DataFrame({"ew":ew_dict['ew_pv'],"rp_pv":rp_dict['rp_pv']}),  linewidth=2.5)
        sns.lineplot(data=pd.DataFrame({label_name[i]:pv_dict[i][label_name[i]] for i in range(len(label_name))}),  linewidth=2.5)
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.legend(loc='upper left')
        plt.title('各模型净值变化')
        plt.show()

    def get_eval(self):
        self.rp_eval = Eval_util.get_eval_portfolio(self.result_rp['rp_pv'],(self.cal_windows,self.change_time_delta,'rp'))
        self.benchmark_eval = pd.concat([Eval_util.get_eval_portfolio(self.result_benchmark[i][i+'_pv'],i) for i in self.benchmark.keys()],axis=1)
        self.eval_df = pd.concat([self.rp_eval,self.benchmark_eval],axis=1)
        logger.info(f"策略相关指标,年化收益率:{round(self.rp_eval.iloc[1,0],4)},波动率:{round(self.rp_eval.iloc[2,0],4)},最大回撤:{round(self.rp_eval.iloc[3,0],4)},夏普比:{round(self.rp_eval.iloc[4,0],4)}")

def main_1():
    lambda_pd = pd.read_csv('./excel/lambda_max_windows.csv')
    lambda_pd['max_windows'] = lambda_pd['max_windows']*5
    test_rp = RPmodel(
                    ret_,
                    start_date='2006-01-01',
                    end_date='2025-08-27',
                    change_time_delta='Y',
                    cal_windows=600,
                    cov_matrix_method = 'SEMI',#协方差选择
                    risk_alloc = 'EWRCP', #'EWRCP'纯正的风险平价-等权分配风险 ，'SSWRCP' 按夏普率分配风险
                    risk_budget_objective='naive_risk_parity',
                    optimizer = 'SLSQP',#优化器选择
    #                  optimization_part_showsFalse,
                    optimization_initial_position = [0.06,0.94],
    #                 optimization_initial_position = [0.14,0.14,0.14,0.14,0.14,0.16,0.14],
                    optimization_montca = [True,500],
                    leverage_switch=0,
                    leverage_post_ratio=1,
                    frequency_calcov = 252,
                    )
    if test_rp.cov_matrix_method != None:
        for i in range(len(lambda_pd)):
            lambda_ = lambda_pd.iloc[i,0]
            max_windows = lambda_pd.iloc[i,1]
            print(f"start_lambda_:{lambda_},max_windows:{max_windows}:%%%%%%%%%%%%")
            test_rp.cal_windows = max_windows #600天计算风险平价
            test_rp.calcov = 52 #按周计算波动率
            test_rp.ewma_lambda = lambda_
            test_rp.position_get(1,1)
            test_rp.Backtest()
            # test_rp.get_eval()
            # #年度评价
            Annual_df = Eval_util.Year_analysis(test_rp.result_rp['rp_portfolio_ret'],dafult_VaR_year_windows=5,file_name=f'lambda{lambda_*100}')
            # print(Annual_df)

            # #作图
            # test_rp.plot_weight_change(test_rp.result_rp['rp_weight_df'])
            # print("="*60)
            # list_pv = [test_rp.result_rp]
            # label_pv = ['rp_pv']
            # for i in test_rp.result_benchmark.keys():
            #     list_pv.append(test_rp.result_benchmark[i])
            #     label_pv.append(i+'_pv')
            # test_rp.plot_pv(list_pv,label_name=label_pv)

            print("="*60)
            ##与benchmark对比
            # eval_df = pd.concat([test_rp.rp_eval,test_rp.benchmark_eval],axis=1)
            # print(round(eval_df,4))
            print("="*60)

def main_2():
    test_rp = RPmodel(
                ret_,
                start_date='2010-01-01',
                end_date='2025-10-14',
                change_time_delta='Y',
                cal_windows=600,
                cov_matrix_method = 'EWMA',#协方差选择
                risk_alloc = 'SSWRCP', #'EWRCP'纯正的风险平价-等权分配风险 ，'SSWRCP' 按夏普率分配风险
                risk_budget_objective='naive_risk_parity',
                optimizer = 'SLSQP',#优化器选择
#                  optimization_part_showsFalse,
                optimization_initial_position = [0.06,0.94],
#                 optimization_initial_position = [0.14,0.14,0.14,0.14,0.14,0.16,0.14],
                optimization_montca = [True,500],
                leverage_switch=0,
                leverage_post_ratio=1,
                frequency_calcov = 252,
                )
    test_rp.calcov = 52 #按周计算波动率
    test_rp.ewma_lambda = 0.91
    test_rp.cal_windows = 250 #600天计算风险平价    
    test_rp.position_get(1,1)
    test_rp.Backtest()
    Annual_df = Eval_util.Year_analysis(test_rp.result_rp['rp_portfolio_ret'],dafult_VaR_year_windows=5,file_name=f'夏普分配')
    test_rp.plot_weight_change(test_rp.result_rp['rp_weight_df'])
    print("="*60)
    list_pv = [test_rp.result_rp]
    label_pv = ['rp_pv']
    for i in test_rp.result_benchmark.keys():
        list_pv.append(test_rp.result_benchmark[i])
        label_pv.append(i+'_pv')
    test_rp.plot_pv(list_pv,label_name=label_pv)
if __name__ == '__main__':
    main_2()