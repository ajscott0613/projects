import pandas as pd
import datetime as dt
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from util import get_data

def author(): 
    return 'ascott97'

def get_price(symbol, dates):

    df_prices = get_data([symbol], dates, addSPY=True, colname='Adj Close')
    del df_prices['SPY']
    df_prices.fillna(method='ffill', inplace=True)
    df_prices.fillna(method='bfill', inplace=True)
    return df_prices


def get_SMA(prices, lookback):
    SMA = prices.rolling(window=lookback,center=False).mean()
    return SMA

def get_SMA_Cross(st, lt, prices):
    SMA_st = get_SMA(prices, st)
    SMA_lt = get_SMA(prices, lt)
    SMA_ind = SMA_st/SMA_lt - 1
    return SMA_st, SMA_lt, SMA_ind


def get_BB(prices, lookback):
    SMA = get_SMA(prices, lookback)
    mov_std = prices.rolling(window=lookback,center=False).std()
    upper = SMA + (2 * mov_std)
    lower = SMA - (2 * mov_std)
    # bb_indicator = (prices - SMA) / (2 * mov_std)
    # return upper, lower, bb_indicator
    bb_indicator = (prices - lower) / (upper - lower)
    return upper, lower, bb_indicator

def get_momentum(prices, lookback):
    momentum = prices/prices.shift(lookback) - 1
    return momentum

def get_CCI(prices, lookback):
    tp = (prices.rolling(window=lookback,center=False).max() + 
        prices.rolling(window=lookback,center=False).median() + 
        prices.rolling(window=lookback,center=False).min())/3
    mad = tp.mad()
    CCI = (tp - tp.mean()) / (0.015*mad)
    return CCI

def get_MACD(prices, st=12, lt=26):
    SMA_st= get_SMA(prices, st)
    SMA_lt = get_SMA(prices,lt)
    macd_line = SMA_st - SMA_lt
    signal_line = get_SMA(macd_line, 9)
    macd_indicator = macd_line - signal_line
    return macd_line, signal_line, macd_indicator



def plot_indicators():

    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    alldates = pd.date_range(sd,ed)
    symbol = 'JPM'

    prices = get_price(symbol, alldates)
    SMA = get_SMA(prices, 14)
    SMA_st, SMA_lt, SMA_ind = get_SMA_Cross(14, 100, prices)
    upper_bb, lower_bb, bb_indicator = get_BB(prices, 14) # 14 day
    momentum = get_momentum(prices, 14)
    cci_indicator = get_CCI(prices, 14)
    macd_line, signal_line, macd_indicator = get_MACD(prices)

    # figure 1 - SMA Indicators
    plt.plot(prices, color="red")
    plt.plot(SMA_st, color="green")
    plt.plot(SMA_lt, color="blue")
    plt.xticks(rotation=30)
    plt_name = "Simple Moving Average"
    plt.legend(["Price", "SMA (14 Day)", "SMA (100 Day)"], loc="lower left")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(plt_name)
    plt.savefig('Indicator_1_SMA_Cross_Overplot.png', bbox_inches="tight")
    plt.close()

    plt.plot(SMA_ind, color="purple")
    plt.axhline(y=0, color="grey", linestyle="--", alpha=0.5)
    plt_name = "SMA Indicator JPM"
    plt.xticks(rotation=30)
    plt.xlabel("Date")
    plt.ylabel("SMA Indicator Value")
    plt.title(plt_name)
    plt.savefig('Indicator_1_SMA_Cross.png', bbox_inches="tight")
    plt.close()



    # figure 2 - Momentum Indicator
    plt.plot(momentum, color="purple")
    plt.axhline(y=-0.25, color="grey", linestyle="--", alpha=0.5)
    plt.axhline(y=0, color="grey", linestyle="-", alpha=0.5)
    plt.axhline(y=0.25, color="grey", linestyle="--", alpha=0.5)
    plt_name = "Momentum JPM"
    plt.xticks(rotation=30)
    plt.xlabel("Date")
    plt.ylabel("Momentum")
    plt.title(plt_name)
    plt.savefig('Indicator_2_Momentum.png', bbox_inches="tight")
    plt.close()



    # figure 3 - Bollinger Bands Indicator
    plt.plot(prices, color="blue")
    plt.plot(SMA, color="red", alpha=0.5)
    plt.plot(upper_bb, color="green", alpha=0.5)
    plt.plot(lower_bb, color="orange", alpha=0.5)
    plt.xticks(rotation=30)
    plt_name = "Bollinger Bands JPM" 
    plt.legend(["Price", "SMA (14 Day)", "Upper BB", "Lower BB"], loc="lower left")
    plt.xlabel("Date")
    plt.ylabel("Price (Adjusted Close $)")
    plt.title(plt_name)
    plt.savefig('Indicator_3_BB_Overplot.png', bbox_inches="tight")
    plt.close()

    plt.plot(bb_indicator, color="purple")
    # plt.axhline(y=-1, color="grey", linestyle="--", alpha=0.5)
    plt.axhline(y=0.8, color="grey", linestyle="--", alpha=0.5)
    plt.axhline(y=0.2, color="grey", linestyle="--", alpha=0.5)
    plt_name = "Bollinger Bands Inidcator JPM"
    plt.xticks(rotation=30)
    plt.xlabel("Date")
    plt.ylabel("Indicator Value")
    plt.title(plt_name)
    plt.savefig('Indicator_3_BB.png', bbox_inches="tight")
    plt.close()


    # figure 4 - Commodity Channel Index Indicator
    plt.plot(cci_indicator, color="purple")
    plt.axhline(y=-100, color="grey", linestyle="--", alpha=0.5)
    plt.axhline(y=0, color="grey", linestyle="-", alpha=0.5)
    plt.axhline(y=100, color="grey", linestyle="--", alpha=0.5)
    plt_name = "Commdity Channel Indicator JPM"
    plt.xticks(rotation=30)
    plt.xlabel("Date")
    plt.ylabel("Indicator Value")
    plt.title(plt_name)
    plt.savefig('Indicator_4_CCI.png', bbox_inches="tight")
    plt.close()


    # figure 5 - Moving Average Convergence Divergence Indicator
    plt.plot(macd_line, color="green")
    plt.plot(signal_line, color="red")
    plt.legend(["MACD", "Signal Line"], loc="lower right")
    plt_name = "Moving Average Convergence Divergence JPM"
    plt.xticks(rotation=30)
    plt.xlabel("Date")
    plt.ylabel("Price Difference")
    plt.title(plt_name)
    plt.savefig('Indicator_5_MACD_Overplot.png', bbox_inches="tight")
    plt.close()

    plt.plot(macd_indicator, color="purple")
    # plt.axhline(y=-100, color="grey", linestyle="--", alpha=0.5)
    plt.axhline(y=0, color="grey", linestyle="--", alpha=0.5)
    # plt.axhline(y=100, color="grey", linestyle="--", alpha=0.5)
    plt_name = "MACD Indicator JPM"
    plt.xticks(rotation=30)
    plt.xlabel("Date")
    plt.ylabel("Indicator Value")
    plt.title(plt_name)
    plt.savefig('Indicator_5_MACD.png', bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_indicators()