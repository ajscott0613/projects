
import StrategyLearner as sl
import ManualStrategy as ms
import indicators as ind
import marketsimcode as msc
import experiment1 as exp1
import experiment2 as exp2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data
import datetime as dt
import warnings
import time
import sys
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

def plot_data(df_trades, df_trades_base, trader_name, symbol = "JPM",sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31)):

    portvals = msc.compute_portvals(df_trades, symbol)
    base_port_val = msc.compute_portvals(df_trades_base, symbol)

    plt.plot(portvals/portvals[0], color="red")
    plt.plot(base_port_val/base_port_val[0], color="green")
    for index, val in df_trades.iterrows():
        # print(index)
        # print(val['Trades'])
        if val['Trades'] > 0:
            plt.axvline(index,color='blue')
        if val['Trades'] < 0:
            plt.axvline(index,color='black')
    plt.xticks(rotation=30)
    plt_name = trader_name + " - Normalized " + symbol + " Returns"
    plt.legend([trader_name, "Benchmark"], loc="bottom left")
    plt.xlabel("Date")
    plt.ylabel("Normalized Returns")
    plt.title(plt_name)
    plt.savefig(plt_name + '.png', bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

        # Check arguments
        verbose_flg = False
        if len(sys.argv) > 2:
                error("Too Many arguemnts!")
        if len(sys.argv) > 1:
                if sys.argv[1] == '-verbose':
                        verbose_flg = True

Symbols = ["ML4T-220", "AAPL", "UNH", "SINE_FAST_NOISE", "V"]
idx = 4


## MANUAL LEARNER ANALYSIS

# # Baseline Policy
# df_trades_base_in = baselinePolicy(symbol = Symbols[idx], sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # In Sample
# df_trades_base_out = baselinePolicy(symbol = Symbols[idx], sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000) # Out of Sample

# # Manual Strategy
# manual_trader = ms.ManualStrategy(verbose = False, impact = 0.005, commission=9.95)
# df_trades_man_in = manual_trader.testPolicy(symbol = Symbols[idx], sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # In Sample
# df_trades_man_out = manual_trader.testPolicy(symbol = Symbols[idx], sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000) # Out of Sample
# # print(df_trades_man)


# # Plot Manual and Learner Traders Against Baseline - In Sample
# # tos.plot_data(df_trades, df_trades_base_in, "Learner", symbol = Symbols[idx],sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31))
# plot_data(df_trades_man_in, df_trades_base_in, "Manual (In Sample)", symbol = Symbols[idx],sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31))

# # Plot Manual and Learner Traders Against Baseline - Out of Sample
# # tos.plot_data(df_trades, df_trades_base_out, "Learner", symbol = Symbols[idx],sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31))
# plot_data(df_trades_man_out, df_trades_base_out, "Manual (Out Sample)", symbol = Symbols[idx],sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31))

# # Get Stats for Manual Trader With In and Out of Sample Data
# if verbose_flg:
#         print(" ****** Manual Trader Statistics ****** ")

#         get_stats(df_trades_man_in, "Manual In Sample", symbol = Symbols[idx],sd=dt.datetime(2008,1,1), 
#                                 ed=dt.datetime(2009,12,31), commission=9.95, impact=0.005, verbose=True)

#         get_stats(df_trades_base_in, "Benchmark In Sample", symbol = Symbols[idx],sd=dt.datetime(2008,1,1), 
#                                 ed=dt.datetime(2009,12,31), commission=9.95, impact=0.005, verbose=True)

#         get_stats(df_trades_man_out, "Manual Out of Sample", symbol = Symbols[idx],sd=dt.datetime(2010,1,1), 
#                                 ed=dt.datetime(2011,12,31), commission=9.95, impact=0.005, verbose=True)

#         get_stats(df_trades_base_out, "Benchmark Out of Sample", symbol = Symbols[idx],sd=dt.datetime(2008,1,1), 
#                                 ed=dt.datetime(2009,12,31), commission=9.95, impact=0.005, verbose=True)

#         print(" ****** Manual Trader Statistics ****** ")



## MANUAL LEARNER ANALYSIS
# Run Experiment 1
exp1.run_exp1()


# Run Experiment 2
# exp2.run_exp2(verbose=verbose_flg)