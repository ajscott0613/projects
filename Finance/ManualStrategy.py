""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
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
import random                                                                                   
         
import numpy as np                                                                          
import pandas as pd                                                                                     
import util as ut
import indicators as ind
import marketsimcode as msc
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
class ManualStrategy(object):  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 			  		 			 	 	 		 		 	
        If verbose = False your code should not generate ANY output.  		  	   		   	 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		   	 			  		 			 	 	 		 		 	
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		   	 			  		 			 	 	 		 		 	
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    # constructor  		  	   		   	 			  		 			 	 	 		 		 	
    def __init__(self, verbose=False, impact=0.05, commission=9.95):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Constructor method  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        self.verbose = verbose  		  	   		   	 			  		 			 	 	 		 		 	
        self.impact = impact  		  	   		   	 			  		 			 	 	 		 		 	
        self.commission = commission
        self.lookback = 14

    def author(): 
        return 'ascott97'

    def take_action(self, position, action):
        """
        : action = 0 --> BUY
        : action = 1 --> SELL
        : action = 2 --> HOLD
        """

        new_position = position
        
        if position == -1000:
            if action == 0:
                new_position = 1000
        elif position == 0:
            if action == 0:
                new_position = 1000
            elif action == 1:
                new_position = -1000
        elif position == 1000:
            if action == 1:
                new_position = -1000

        return new_position  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # this method should use the existing policy and test it against new data  		  	   		   	 			  		 			 	 	 		 		 	
    def testPolicy(  		  	   		   	 			  		 			 	 	 		 		 	
        self,  		  	   		   	 			  		 			 	 	 		 		 	
        symbol="IBM",  		  	   		   	 			  		 			 	 	 		 		 	
        sd=dt.datetime(2009, 1, 1),  		  	   		   	 			  		 			 	 	 		 		 	
        ed=dt.datetime(2010, 1, 1),  		  	   		   	 			  		 			 	 	 		 		 	
        sv=10000,  		  	   		   	 			  		 			 	 	 		 		 	
    ):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Tests your learner using data outside of the training data  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        :param symbol: The stock symbol that you trained on on  		  	   		   	 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		   	 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		   	 			  		 			 	 	 		 		 	
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 			  		 			 	 	 		 		 	
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 			  		 			 	 	 		 		 	
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 			  		 			 	 	 		 		 	
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 			  		 			 	 	 		 		 	
        :rtype: pandas.DataFrame  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        dates = pd.date_range(sd,ed)
        df_prices = ind.get_price(symbol, dates)

        daily_rets = (df_prices / df_prices.shift(1)) - 1
        daily_rets = daily_rets[1:]


        sd_older = sd - dt.timedelta(days=365)
        dates_older = pd.date_range(sd_older,ed)
        df_prices_older = ind.get_price(symbol, dates_older)
        sd_key = df_prices.index[0]
        sd_index = df_prices_older.index.get_loc(sd_key)


        df_holdings = df_prices.copy()
        df_holdings['Holdings'] = np.nan
        del df_holdings[symbol]
        # print(df_holdings)


        # Get Indicator Values
        _,_,ind1 = ind.get_BB(df_prices_older, self.lookback)
        ind2 = ind.get_CCI(df_prices_older, self.lookback)
        _,_,ind3 = ind.get_SMA_Cross(self.lookback, 100, df_prices_older)
        ind4 = ind.get_momentum(df_prices_older, self.lookback)
        _,_,ind5 = ind.get_MACD(df_prices_older)
        BB = ind1.iloc[sd_index:].values
        CCI = ind2.iloc[sd_index:].values
        SMA_Cross = ind3.iloc[sd_index:].values
        Momentum = ind4.iloc[sd_index:].values
        MACD = ind5.iloc[sd_index:].values

        df_holdings.iloc[0]['Holdings'] = 0
        action = 2

        # BB_threshold = 1.0
        # MACD_threshold = 0.15
        # Momentum_threshold = 0.1

        # BB_threshold = 0.
        BB_threshold = 0.0
        MACD_threshold = 0.0
        Momentum_threshold = 0


        for day_idx in range(1,daily_rets.shape[0]):

            # BB Logic
            BB_sell = BB[day_idx-1] >= 1-BB_threshold and BB[day_idx] < 1-BB_threshold
            BB_buy = BB[day_idx-1] <= BB_threshold and BB[day_idx] > BB_threshold

            # MACD Logic
            # MACD_sell = MACD[day_idx] > MACD[day_idx-1] and MACD[day_idx] > 0 and MACD[day_idx] < MACD_threshold
            # MACD_buy = MACD[day_idx] < MACD[day_idx-1] and MACD[day_idx] < 0 and MACD[day_idx] > -1*MACD_threshold
            MACD_sell = MACD[day_idx] >= 0 and MACD[day_idx-1] < 0
            MACD_buy = MACD[day_idx] <= 0 and MACD[day_idx-1] > 0

            # Momentum Logic
            Momentum_sell = Momentum[day_idx] > Momentum_threshold
            Momentum_buy = Momentum[day_idx] < -1*Momentum_threshold

            # SMA Cross Logic
            SMA_sell = SMA_Cross[day_idx] <= 0 and SMA_Cross[day_idx-1] > 0
            SMA_buy = SMA_Cross[day_idx] >= 0 and SMA_Cross[day_idx-1] < 0

            # Momentum_sell = True
            # Momentum_buy = True
            # Momentum_buy = Momentum[day_idx] > Momentum_threshold
            # Momentum_sell = Momentum[day_idx] < -1*Momentum_threshold


            action = 2
            if (BB_sell and Momentum_sell) or (MACD_sell and Momentum_sell):
                action = 1 # Sell signal
            if (BB_buy and Momentum_buy) or (MACD_buy and Momentum_buy):
                action = 0 # Buy Signal





            df_holdings.iloc[day_idx]['Holdings'] = self.take_action(df_holdings.iloc[day_idx-1]['Holdings'], action)  

        df_holdings.iloc[-1]['Holdings'] = 0
        df_trades = df_holdings.diff()
        df_trades['Trades'] = df_trades['Holdings']
        del df_trades['Holdings']
        df_trades.iloc[0]['Trades'] = 0
        return df_trades	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		   	 			  		 			 	 	 		 		 	
    print("One does not simply think up a strategy")  		  	   		   	 			  		 			 	 	 		 		 	
