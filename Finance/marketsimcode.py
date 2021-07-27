""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""MC2-P1: Market simulator.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		   	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			 	 	 		 		 	
or edited.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Student Name: Tucker Balch (replace with your name)  		  	   		   	 			  		 			 	 	 		 		 	
GT User ID: tb34 (replace with your User ID)  		  	   		   	 			  		 			 	 	 		 		 	
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
import datetime as dt  		  	   		   	 			  		 			 	 	 		 		 	
import os  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
import numpy as np  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
import pandas as pd  		  	   		   	 			  		 			 	 	 		 		 	
from util import get_data, plot_data

def author():
    return 'ascott97'  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
def compute_portvals(  		  	   		   	 			  		 			 	 	 		 		 	
    df_trades_in,
    symbol="AAPL",  		  	   		   	 			  		 			 	 	 		 		 	
    start_val=100000,  		  	   		   	 			  		 			 	 	 		 		 	
    commission=9.95,  		  	   		   	 			  		 			 	 	 		 		 	
    impact=0.005,  		  	   		   	 			  		 			 	 	 		 		 	
):  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    Computes the portfolio values.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    :param orders_file: Path of the order file or the file object  		  	   		   	 			  		 			 	 	 		 		 	
    :type orders_file: str or file object  		  	   		   	 			  		 			 	 	 		 		 	
    :param start_val: The starting value of the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
    :type start_val: int  		  	   		   	 			  		 			 	 	 		 		 	
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		   	 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		   	 			  		 			 	 	 		 		 	
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		   	 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		   	 			  		 			 	 	 		 		 	
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		   	 			  		 			 	 	 		 		 	
    :rtype: pandas.DataFrame  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    # this is the function the autograder will call to test your code  		  	   		   	 			  		 			 	 	 		 		 	
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		   	 			  		 			 	 	 		 		 	
    # code should work correctly with either input  		  	   		   	 			  		 			 	 	 		 		 	
    # TODO: Your code here  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # In the template, instead of computing the value of the portfolio, we just  		  	   		   	 			  		 			 	 	 		 		 	
    # read in the value of IBM over 6 months  		  	   		   	 			  		 			 	 	 		 		 	

    df_orders = df_trades_in.copy()
    df_orders.index.name = 'Date'

    # print(df_orders)

    # sort oders df and find start and end dates
    df_orders = df_orders.sort_values(by='Date',ascending=True)
    # df_orders = df_orders.sort_values(by=index,ascending=True)
    start_date = df_orders.index[0]
    end_date = df_orders.index[-1]

    # create prices dataframe
    stocks = [symbol]
    alldates = pd.date_range(start_date, end_date)
    df_prices = get_data(stocks, alldates, addSPY=True)
    df_prices['Cash'] = 1.0
    del df_prices['SPY']

    # create trades dataframe
    df_trades = df_prices.copy()
    df_trades[:] = 0

    # create trades dataframe
    for index, row in df_orders.iterrows():

        shares = row['Trades']
        price = df_prices.loc[index][symbol]

        # if row['Order'] == 'BUY':
        #     cash = -1 * price * shares * (1 + impact) - commission
        # else:
        #     cash = price * shares * (1 - impact) - commission
        #     shares *= -1

        if np.abs(shares) > 0:
            cash = -1 * price * shares * (1 - impact) - commission
        else:
            cash = 0
        # if shares > 0:
        #     shares *= -1

        df_trades.loc[index][symbol] += shares
        df_trades.loc[index]['Cash'] += cash


    # create holdings dataframe
    df_holdings = df_trades.copy()
    df_holdings.loc[start_date]['Cash'] += start_val
    df_holdings = df_holdings.cumsum()
    # print(df_holdings)


    # create values dataframe and portvals array
    df_values = df_prices * df_holdings
    portvals = df_values.sum(axis=1)
                                                                                
    return portvals     	  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
def test_code(df_trades, symbol = "AAPL"):  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    Helper function to test code  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    # this is a helper function you can use to test your code  		  	   		   	 			  		 			 	 	 		 		 	
    # note that during autograding his function will not be called.  		  	   		   	 			  		 			 	 	 		 		 	
    # Define input parameters  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 	 		  	   		   	 			  		 			 	 	 		 		 			  	   		   	 			  		 			 	 	 		 		 	
    # Process orders  		  	   		   	 			  		 			 	 	 		 		 	
    # portvals = compute_portvals(orders_file=of, start_val=sv)
    portvals = compute_portvals(df_trades, symbol)
    port_val = portvals  		  	   		   	 			  		 			 	 	 		 		 	
    if isinstance(portvals, pd.DataFrame):  		  	   		   	 			  		 			 	 	 		 		 	
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		   	 			  		 			 	 	 		 		 	
    else:  		  	   		   	 			  		 			 	 	 		 		 	
        "warning, code did not return a DataFrame"  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # Get portfolio stats  		  	   		   	 			  		 			 	 	 		 		 	
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		   	 			  		 			 	 	 		 		 	
    start_date = dt.datetime(2008, 1, 1)  		  	   		   	 			  		 			 	 	 		 		 	
    end_date = dt.datetime(2008, 6, 1)  		  	   		   	 			  		 			 	 	 		 		 	
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [  		  	   		   	 			  		 			 	 	 		 		 	
        0.2,  		  	   		   	 			  		 			 	 	 		 		 	
        0.01,  		  	   		   	 			  		 			 	 	 		 		 	
        0.02,  		  	   		   	 			  		 			 	 	 		 		 	
        1.5,  		  	   		   	 			  		 			 	 	 		 		 	
    ]  		  	   		   	 			  		 			 	 	 		 		 	
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		   	 			  		 			 	 	 		 		 	
        0.2,  		  	   		   	 			  		 			 	 	 		 		 	
        0.01,  		  	   		   	 			  		 			 	 	 		 		 	
        0.02,  		  	   		   	 			  		 			 	 	 		 		 	
        1.5,  		  	   		   	 			  		 			 	 	 		 		 	
    ]

    sf = 252
    rfr = 0
    daily_rets = (port_val / port_val.shift(1)) - 1
    daily_rets = daily_rets[1:]
    cum_ret = (port_val[-1] / port_val[0]) - 1
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sharpe_ratio = np.sqrt(sf) * np.mean(avg_daily_ret - rfr) / std_daily_ret

  		  	   		   	 			  		 			 	 	 		 		 	
    # Compare portfolio against $SPX  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Date Range: {start_date} to {end_date}")  		  	   		   	 			  		 			 	 	 		 		 	
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		   	 			  		 			 	 	 		 		 	
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		   	 			  		 			 	 	 		 		 	
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		   	 			  		 			 	 	 		 		 	
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		   	 			  		 			 	 	 		 		 	
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		   	 			  		 			 	 	 		 		 	
    test_code()  		  	   		   	 			  		 			 	 	 		 		 	
