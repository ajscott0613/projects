"""
Code for generating data for Experiment 2 in Project 8
"""

import StrategyLearner as sl
import ManualStrategy as ms
import indicators as ind
import marketsimcode as msc
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import warnings
import time
warnings.filterwarnings("ignore")

def author():
	return 'ascott97'

def get_stats(df_trades, trader_name, symbol = "JPM",sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), 
	commission = 0.0, impact=0.005, verbose=False):

    portvals = msc.compute_portvals(df_trades, symbol, impact=impact, commission=0.0)

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


def plot_data(df_trades_1, df_trades_2, df_trades_3, df_trades_4, impacts, symbol = "JPM"):

    portvals_1 = msc.compute_portvals(df_trades_1, symbol, impact=impacts[0], commission=0.0)
    portvals_2 = msc.compute_portvals(df_trades_2, symbol, impact=impacts[1], commission=0.0)
    portvals_3 = msc.compute_portvals(df_trades_3, symbol, impact=impacts[2], commission=0.0)
    portvals_4 = msc.compute_portvals(df_trades_4, symbol, impact=impacts[3], commission=0.0)

    plt.plot(portvals_1/portvals_1[0], color="red")
    plt.plot(portvals_2/portvals_2[0], color="orange")
    plt.plot(portvals_3/portvals_3[0], color="green")
    plt.plot(portvals_4/portvals_4[0], color="blue")

    plt.xticks(rotation=30)
    plt_name = "Experiment 2 - Normalized Returns (In Sample)"
    plt.legend(["Impact = " + str(impacts[0]), "Impact = " + str(impacts[1]), "Impact = " + str(impacts[2]), 
    	"Impact = " + str(impacts[3])], loc="bottom left")
    plt.xlabel("Date")
    plt.ylabel("Normalized Returns")
    plt.title(plt_name)
    plt.savefig(plt_name + '.png', bbox_inches="tight")
    plt.close()


def run_exp2(verbose=False):

	Symbols = ["ML4T-220", "AAPL", "UNH", "SINE_FAST_NOISE", "JPM"]
	Trader = ["Benchmark", "Manual", "Learner"]
	idx = 4

	# Strategy Learner
	impacts = [0.0, 0.005, 0.01, 0.05]
	learner1 = sl.StrategyLearner(verbose = False, impact=impacts[0], commission=0.0) # constructor
	learner1.add_evidence(symbol = Symbols[idx], sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase
	df_trades_1 = learner1.testPolicy(symbol = Symbols[idx], sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # testing phase

	learner2 = sl.StrategyLearner(verbose = False, impact=impacts[1], commission=0.0) # constructor
	learner2.add_evidence(symbol = Symbols[idx], sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase
	df_trades_2 = learner2.testPolicy(symbol = Symbols[idx], sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)

	learner3 = sl.StrategyLearner(verbose = False, impact=impacts[2], commission=0.0) # constructor
	learner3.add_evidence(symbol = Symbols[idx], sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase
	df_trades_3 = learner3.testPolicy(symbol = Symbols[idx], sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)

	learner4 = sl.StrategyLearner(verbose = False, impact=impacts[3], commission=0.0) # constructor
	learner4.add_evidence(symbol = Symbols[idx], sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase
	df_trades_4 = learner4.testPolicy(symbol = Symbols[idx], sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)

	plot_data(df_trades_1, df_trades_2, df_trades_3, df_trades_4, impacts, Symbols[idx])

	# Get Stats for Each Trader
	if verbose:
		print(" ****** Strategy Learner Statistics (Experiment #2) ****** ")
		print(" **                                                     ** ")
		for i in range(4):
			if i == 0:
				df_trades = df_trades_1
			elif i == 1:
				df_trades = df_trades_2
			elif i == 2:
				df_trades = df_trades_3
			elif i == 3:
				df_trades = df_trades_4

			get_stats(df_trades, "Impact = " + str(impacts[i]), symbol = Symbols[idx],sd=dt.datetime(2008,1,1), 
				ed=dt.datetime(2009,12,31), commission=0.0, impact=impacts[i], verbose=True)
		print(" **                                                     ** ")
		print(" ****** Strategy Learner Statistics (Experiment #2) ****** ")