"""
Code for generating data for Experiment 1 in Project 8
"""

import StrategyLearner as sl
import ManualStrategy as ms
import indicators as ind
import marketsimcode as msc
import numpy as np
import pandas as pd
from util import get_data
import matplotlib.pyplot as plt
import datetime as dt
import warnings
import time
warnings.filterwarnings("ignore")

def author():
    return 'ascott97'


def get_stats(df_trades, trader_name, symbol = "JPM",sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), verbose=False):

    portvals = msc.compute_portvals(df_trades, symbol)

    sf = 252
    rfr = 0
    daily_rets = (portvals / portvals.shift(1)) - 1
    daily_rets = daily_rets[1:]
    cum_ret = (portvals[-1] / portvals[0]) - 1
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sharpe_ratio = np.sqrt(sf) * np.mean(avg_daily_ret - rfr) / std_daily_ret
                                                                                    
    # Print Stats
    if verbose:                                                                                
        print(f"Date Range: {sd} to {ed}")                                                                                    
        print()                                                                                     
        print(f"Sharpe Ratio of {trader_name}: {np.round(sharpe_ratio,4)}")                                                                                      
        print()                                                                                     
        print(f"Cumulative Return of {trader_name}: {np.round(cum_ret,4)}")                                                                                                                                                                            
        print()                                                                                     
        print(f"Standard Deviation of {trader_name}: {np.round(std_daily_ret,4)}")                                                                                                                                                                      
        print()                                                                                     
        print(f"Average Daily Return of {trader_name}: {np.round(avg_daily_ret,4)}")                                                                                                                                                                         
        print()                                                                                     
        print(f"Final Portfolio Value of {trader_name}: {np.round(portvals[-1],4)}")


def plot_data(df_trades_man, df_trades_strat, df_trades_base, trader_name, symbol = "JPM"):

    portvals_man = msc.compute_portvals(df_trades_man, symbol)
    portvals_strat = msc.compute_portvals(df_trades_strat, symbol)
    portvals_base = msc.compute_portvals(df_trades_base, symbol)

    plt.plot(portvals_man/portvals_man[0], color="red")
    plt.plot(portvals_strat/portvals_strat[0], color="orange")
    plt.plot(portvals_base/portvals_base[0], color="green")
    # for index, val in df_trades.iterrows():
    #     # print(index)
    #     # print(val['Trades'])
    #     if val['Trades'] > 0:
    #         plt.axvline(index,color='blue')
    #     if val['Trades'] < 0:
    #         plt.axvline(index,color='black')
    plt.xticks(rotation=30)
    plt_name = "Experiment 1 - Normalized Returns " + trader_name
    plt.legend(["Manual", "Strategy", "Benchmark"], loc="bottom left")
    plt.xlabel("Date")
    plt.ylabel("Normalized Returns")
    plt.title(plt_name)
    plt.savefig(plt_name + '.png', bbox_inches="tight")
    plt.close()

def baselinePolicy(symbol= "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000):

    alldates = pd.date_range(sd,ed)
    stocks = [symbol]
    df_prices = get_data(stocks, alldates, addSPY=True)
    del df_prices['SPY']
    df_trades = df_prices.copy()
    df_trades['Trades'] = 0.0
    del df_trades[symbol]
    df_trades.iloc[0] = 1000

    return df_trades

def run_exp1(verbose=False):
    Symbols = ["ML4T-220", "AAPL", "UNH", "SINE_FAST_NOISE", "JPM", "V_new"]
    Trader = ["Benchmark", "Manual", "Learner"]
    idx = 5

    for j in range(2):
        # print("here")
    

        if j == 0:
            sd = dt.datetime(2020,7,27)
            ed = dt.datetime(2020,12,31)
            trader_name = "(In Sample)"
        else:
            sd = dt.datetime(2021,1,4)
            ed = dt.datetime(2021,7,26)
            trader_name = "(Out of Sample)"

        # Benchmark
        df_trades_base_in = baselinePolicy(symbol = Symbols[idx], sd=sd, ed=ed, sv = 100000)

        # Manual Strategy
        manual_trader = ms.ManualStrategy(verbose = False, impact = 0.005, commission=9.95)
        df_trades_man_in = manual_trader.testPolicy(symbol = Symbols[idx], sd=sd, ed=ed, sv = 100000)


        # Strategy Learner
        if j == 0:
            learner = sl.StrategyLearner(verbose = False, impact = 0.005, commission=9.95) # constructor
            learner.add_evidence(symbol = Symbols[idx], sd=sd, ed=ed, sv = 100000) # training phase
        df_trades_strat_in = learner.testPolicy(symbol = Symbols[idx], sd=sd, ed=ed, sv = 100000) # testing phase

        plot_data(df_trades_man_in, df_trades_strat_in, df_trades_base_in, trader_name, Symbols[idx])


        # if verbose:
        #     print(" ****** Strategy Learner Statistics (Experiment #1) ****** ")
        #     print(" **                                                     ** ")
        #   # Get Stats for Each Trader
        #   for i in range(3):
        #       if i == 0:
        #           df_trades = df_trades_base_in
        #       elif i == 1:
        #           df_trades = df_trades_man_in
        #       elif i == 2:
        #           df_trades = df_trades_strat_in

        #       get_stats(df_trades, Trader[i], symbol = Symbols[idx],sd=sd, ed=ed, verbose=True)
        #     print(" **                                                     ** ")
        #     print(" ****** Strategy Learner Statistics (Experiment #1) ****** ")




# def run_exp1(verbose=False):

# 	Symbols = ["ML4T-220", "AAPL", "UNH", "SINE_FAST_NOISE", "JPM"]
# 	Trader = ["Benchmark", "Manual", "Learner"]
# 	idx = 4

#     for i in range(2):
#         print("here")
    

#  #    if j == 0:
#  #        sd = dt.datetime(2008,1,1)
#  #        ed = dt.datetime(2009,12,31)
#  #        trader_name = "(In Sample)"
#  #    else:
#  #        sd = dt.datetime(2010,1,1)
 #        ed = dt.datetime(2011,12,31)
 #        trader_name = "(Out of Sample)"

	# # Benchmark
	# df_trades_base_in = tos.baselinePolicy(symbol = Symbols[idx], sd=sd, ed=ed, sv = 100000)

	# # Manual Strategy
	# manual_trader = ms.ManualStrategy(verbose = False, impact = 0.005, commission=9.95)
	# df_trades_man_in = manual_trader.testPolicy(symbol = Symbols[idx], sd=sd, ed=ed, sv = 100000)


	# # Strategy Learner
	# learner = sl.StrategyLearner(verbose = False, impact = 0.005, commission=9.95) # constructor
	# learner.add_evidence(symbol = Symbols[idx], sd=sd, ed=ed, sv = 100000) # training phase
	# df_trades_strat_in = learner.testPolicy(symbol = Symbols[idx], sd=sd, ed=ed, sv = 100000) # testing phase


 #    if verbose:
 #        print(" ****** Strategy Learner Statistics (Experiment #1) ****** ")
 #        print(" **                                                     ** ")
 #    	# Get Stats for Each Trader
 #    	for i in range(3):
 #    		if i == 0:
 #    			df_trades = df_trades_base_in
 #    		elif i == 1:
 #    			df_trades = df_trades_man_in
 #    		elif i == 2:
 #    			df_trades = df_trades_strat_in

 #    		get_stats(df_trades, Trader[i], symbol = Symbols[idx],sd=sd, ed=ed, verbose=True)
 #        print(" **                                                     ** ")
 #        print(" ****** Strategy Learner Statistics (Experiment #1) ****** ")

	# plot_data(df_trades_man_in, df_trades_strat_in, df_trades_base_in, trader_name, Symbols[idx])