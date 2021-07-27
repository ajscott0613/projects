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

class QLearner(object):                                                                                     
    """                                                                                     
    This is a Q learner object.                                                                                     
                                                                                    
    :param num_states: The number of states to consider.                                                                                    
    :type num_states: int                                                                                   
    :param num_actions: The number of actions available..                                                                                   
    :type num_actions: int                                                                                      
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.                                                                                      
    :type alpha: float                                                                                      
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.                                                                                      
    :type gamma: float                                                                                      
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.                                                                                      
    :type rar: float                                                                                    
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.                                                                                   
    :type radr: float                                                                                   
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.                                                                                     
    :type dyna: int                                                                                     
    :param verbose: If “verbose” is True, your code can print out information for debugging.                                                                                    
    :type verbose: bool                                                                                     
    """                                                                                     
    def __init__(                                                                                   
        self,                                                                                   
        num_states=100,                                                                                     
        num_actions=4,                                                                                      
        alpha=0.2,                                                                                      
        gamma=0.9,                                                                                      
        rar=0.5,                                                                                    
        radr=0.99,                                                                                      
        dyna=0,                                                                                     
        verbose=False,                                                                                      
    ):                                                                                      
        """                                                                                     
        Constructor method                                                                                      
        """                                                                                     
        self.verbose = verbose                                                                                      
        self.num_actions = num_actions                                                                                      
        self.s = 0                                                                                      
        self.a = 0
        self.Q = np.zeros((num_states, num_actions)) 

        self.num_states = num_states
        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna

        # Initialize Dyna Q Values
        self.T_c = 0.00001*np.ones((num_states, num_actions, num_states))
        self.R = np.zeros((num_states, num_actions))

    def author(self):
        return 'ascott97'                                                                                   
                                                                                    
    def querysetstate(self, s):                                                                                     
        """                                                                                     
        Update the state without updating the Q-table                                                                                   
                                                                                    
        :param s: The new state                                                                                     
        :type s: int                                                                                    
        :return: The selected action                                                                                    
        :rtype: int                                                                                     
        """                                                                                     
        self.s = s                                                                                      

        if np.random.random() < self.rar:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.Q[s])

        if self.verbose:                                                                                      
            print(f"s = {s_prime}, a = {action}, r={r}")  

        return action                                                                                   
                                                                                    
    def query(self, s_prime, r):                                                                                    
        """                                                                                     
        Update the Q table and return an action                                                                                     
                                                                                    
        :param s_prime: The new state                                                                                   
        :type s_prime: int                                                                                      
        :param r: The immediate reward                                                                                      
        :type r: float                                                                                      
        :return: The selected action                                                                                    
        :rtype: int                                                                                     
        """                                                                                     

        a = self.a
        s = self.s

        # print(self.Q)
        # print(self.Q.shape)
        # print(s_prime)
        # Update Q-Table
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + \
            self.alpha * (r + self.gamma * self.Q[s_prime,np.argmax(self.Q[s_prime])])



        # Dyna-Q Code
        if self.dyna > 0:
            self.T_c[s, a, s_prime] += 1
            self.R[s,a] = (1 - self.alpha)*self.R[s,a] + self.alpha*r


            for i in range(self.dyna):
                # Hallucinate Experience
                s_rand = np.random.randint(self.num_states)
                a_rand = np.random.randint(self.num_actions)
                s_prime_dyna = np.argmax(self.T_c[s_rand, a_rand])
                r_dyna = self.R[s_rand, a_rand]

                # Update Q-Table
                self.Q[s_rand, a_rand] = (1 - self.alpha) * self.Q[s_rand, a_rand] + \
                    self.alpha * (r_dyna + self.gamma * self.Q[s_prime_dyna, np.argmax(self.Q[s_prime_dyna])])


        # Select Action
        if np.random.random() < self.rar:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.Q[s_prime])

        self.a = action
        self.s = s_prime
        # self.rar *= self.radr

        if self.verbose:                                                                                      
            print(f"s = {s_prime}, a = {action}, r={r}")      

        return action 		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
class StrategyLearner(object):  		  	   		   	 			  		 			 	 	 		 		 	
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
    def __init__(self, verbose=False, impact=0.005, commission=9.95):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Constructor method  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        self.verbose = verbose  		  	   		   	 			  		 			 	 	 		 		 	
        self.impact = impact  		  	   		   	 			  		 			 	 	 		 		 	
        self.commission = commission
        self.lookback = 14
        self.bins = np.array(list(range(10)))

    def author(): 
        return 'ascott97'

    def take_action(self, position, action, cur_price, next_price):
        """
        : Takes one of the following actions and returns the updated
        : holdings value and reward value
        : action = 0 --> BUY
        : action = 1 --> SELL
        : action = 2 --> HOLD
        """

        reward = 0
        new_position = position
        # daily_return *= 100
        # print(position)
        # print(action)
        
        if position == -1000:
            if action == 0:
                new_position = 1000
                reward = -1*self.impact*(next_price)*2*new_position - self.commission
            # else:
            #     reward = -1*daily_return
        elif position == 0:
            if action == 0:
                new_position = 1000
                reward = 1*self.impact*(next_price)*new_position - self.commission
            elif action == 1:
                new_position = -1000
                reward = -1*self.impact*(next_price)*new_position - self.commission

        elif position == 1000:
            if action == 1:
                new_position = -1000
                reward = 1*self.impact*(next_price)*2*new_position - self.commission
            # else:
            #     reward = daily_return

        # print("reward check = ", reward)
        reward += 1*(next_price - cur_price) * new_position
        # print("final Reward: ", reward)


        return new_position, reward

  		  	   		   	 			  		 			 	 	 		 		 	
    # this method should create a QLearner, and train it for trading  		  	   		   	 			  		 			 	 	 		 		 	
    def add_evidence(  		  	   		   	 			  		 			 	 	 		 		 	
        self,  		  	   		   	 			  		 			 	 	 		 		 	
        symbol="IBM",  		  	   		   	 			  		 			 	 	 		 		 	
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 			  		 			 	 	 		 		 	
        ed=dt.datetime(2009, 1, 1),  		  	   		   	 			  		 			 	 	 		 		 	
        sv=10000,  		  	   		   	 			  		 			 	 	 		 		 	
    ):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Trains your strategy learner over a given time frame.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        :param symbol: The stock symbol to train on  		  	   		   	 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		   	 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        # add your code to do learning here

        converged = False
        x = np.zeros((3,1))
        dates = pd.date_range(sd,ed)
        df_prices = ind.get_price(symbol, dates)

        daily_rets = (df_prices / df_prices.shift(1)) - 1
        daily_rets = daily_rets[1:]


        sd_older = sd - dt.timedelta(days=365)
        dates_older = pd.date_range(sd_older,ed)
        df_prices_older = ind.get_price(symbol, dates_older)
        sd_key = df_prices.index[0]
        sd_index = df_prices_older.index.get_loc(sd_key)

        num_bins = len(self.bins)
        max_state_idx = num_bins + num_bins*10 + num_bins*100

        # Call Q-Learner Constructor
        self.learner = QLearner(                                                                                   
            num_states=(max_state_idx + 1),                                                                                     
            num_actions=3,                                                                                      
            alpha=0.01,                                                                                      
            gamma=0.0,                                                                                      
            rar=0.98,                                                                                    
            radr=0.9995,                                                                                      
            dyna=0,                                                                                   
            verbose=False,                                                                                      
        )

        # df_trades = df_prices.copy()
        df_holdings = df_prices.copy()
        df_holdings['Holdings'] = np.nan
        del df_holdings[symbol]        # print(df_holdings)

        # Initlialize Vars
        cum_ret_prev = 0
        iters = 0
        conv_counter = 0
        Q_prev = np.copy(self.learner.Q)

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
        _,self.x0bins = pd.qcut(BB[:,0], num_bins,labels=False,retbins=True)
        _,self.x1bins = pd.qcut(CCI[:,0],num_bins,labels=False,retbins=True)
        _,self.x2bins = pd.qcut(SMA_Cross[:,0],num_bins,labels=False,retbins=True)
        _,self.x3bins = pd.qcut(Momentum[:,0],num_bins,labels=False,retbins=True)
        _,self.x4bins = pd.qcut(MACD[:,0],num_bins,labels=False,retbins=True)
        x_0 = np.digitize(BB[:,0], self.x0bins[1:-1])
        x_1 = np.digitize(CCI[:,0], self.x1bins[1:-1])
        x_2 = np.digitize(SMA_Cross[:,0], self.x2bins[1:-1])
        x_3 = np.digitize(Momentum[:,0], self.x3bins[1:-1])
        x_4 = np.digitize(MACD[:,0], self.x4bins[1:-1])
        state = x_0 + x_3*10 + x_4*100

    
        while not converged:

            action = self.learner.querysetstate(state[0])
            daily_return = daily_rets.iloc[0][symbol]
            cur_price = df_prices.iloc[0][symbol]
            next_price = df_prices.iloc[1][symbol]
            df_holdings.iloc[0]['Holdings'], reward = self.take_action(0, action, cur_price, next_price)



            for day_idx in range(1,daily_rets.shape[0]):


                daily_return = daily_rets.iloc[day_idx][symbol]
                cur_price = df_prices.iloc[day_idx-1][symbol]
                next_price = df_prices.iloc[day_idx][symbol]
                df_holdings.iloc[day_idx]['Holdings'], reward = self.take_action(df_holdings.iloc[day_idx-1]['Holdings'], action, cur_price, next_price)
                action = self.learner.query(state[day_idx], reward)

            df_holdings.iloc[-1]['Holdings'] = 0
            df_trades = df_holdings.diff()
            df_trades['Trades'] = df_trades['Holdings']
            del df_trades['Holdings']
            df_trades.iloc[0]['Trades'] = 0


            portvals = msc.compute_portvals(                                                                                   
                    df_trades,
                    symbol,                                                                                      
                    sv,                                                                                   
                    self.commission,                                                                                     
                    self.impact,                                                                                     
                )

            cum_ret = (portvals[-1] / portvals[0]) - 1
            Q_diff = np.abs(self.learner.Q - Q_prev)
            Q_max_diff = Q_diff.max()

            if iters > 20:

                # if abs(cum_ret - cum_ret_prev) < 0.0001:
                if Q_max_diff < 0.001:
                    conv_counter += 1
                else:
                    conv_counter = 0

                if conv_counter > 5 or iters > 20000:
                    converged = True
            # if iters > 100:
            #     if iters % 100 == 0:
            #         print("Iteration #", iters)
            print("----------------------------------------------")
            print("--                                          --")
            print("Iteration #", iters)
            print("Error = ", abs(cum_ret - cum_ret_prev))
            print("Q Diff: ", Q_max_diff)
            print("Epsilon: ", self.learner.rar)

            cum_ret_prev = cum_ret
            Q_prev = np.copy(self.learner.Q)
            iters += 1
            self.learner.rar *= self.learner.radr
        # print("Iters = ", iters)
        print("Mode Trained in ", iters, " iterations!")
        np.savetxt('Q_Table.csv', self.learner.Q, delimiter=',')
	  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # this method should use the existing policy and test it against new data  		  	   		   	 			  		 			 	 	 		 		 	
    def testPolicy(  		  	   		   	 			  		 			 	 	 		 		 	
        self,  		  	   		   	 			  		 			 	 	 		 		 	
        symbol="jpm",  		  	   		   	 			  		 			 	 	 		 		 	
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 			  		 			 	 	 		 		 	
        ed=dt.datetime(2009, 12, 31),  		  	   		   	 			  		 			 	 	 		 		 	
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

        cum_ret_prev = 0
        iters = 0


        num_bins = len(self.bins)

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
        _,self.x0bins = pd.qcut(BB[:,0], num_bins,labels=False,retbins=True)
        _,self.x1bins = pd.qcut(CCI[:,0],num_bins,labels=False,retbins=True)
        _,self.x2bins = pd.qcut(SMA_Cross[:,0],num_bins,labels=False,retbins=True)
        _,self.x3bins = pd.qcut(Momentum[:,0],num_bins,labels=False,retbins=True)
        _,self.x4bins = pd.qcut(MACD[:,0],num_bins,labels=False,retbins=True)
        x_0 = np.digitize(BB[:,0], self.x0bins[1:-1])
        x_1 = np.digitize(CCI[:,0], self.x1bins[1:-1])
        x_2 = np.digitize(SMA_Cross[:,0], self.x2bins[1:-1])
        x_3 = np.digitize(Momentum[:,0], self.x3bins[1:-1])
        x_4 = np.digitize(MACD[:,0], self.x4bins[1:-1])
        state = x_0 + x_3*10 + x_4*100





        self.learner.rar = 0

        action = self.learner.querysetstate(state[0])

        daily_return = daily_rets.iloc[0][symbol]
        df_holdings.iloc[0]['Holdings'] = 0


        for day_idx in range(1,daily_rets.shape[0]):

            # implement action
            cur_price = df_prices.iloc[day_idx-1][symbol]
            next_price = df_prices.iloc[day_idx][symbol]
            action = self.learner.querysetstate(state[day_idx])
            df_holdings.iloc[day_idx]['Holdings'],_ = self.take_action(df_holdings.iloc[day_idx-1]['Holdings'], action, cur_price, next_price)


        df_holdings.iloc[-1]['Holdings'] = 0
        df_trades = df_holdings.diff()
        df_trades['Trades'] = df_trades['Holdings']
        del df_trades['Holdings']
        df_trades.iloc[0]['Trades'] = 0
        return df_trades



    def compute_returns(        
        self,
        df_trades,                                                                                   
        symbol="IBM",                                                                                   
        sd=dt.datetime(2009, 1, 1),                                                                                     
        ed=dt.datetime(2010, 1, 1),                                                                                     
        sv=10000,   
        ):

        portvals = msc.compute_portvals(                                                                                   
        df_trades,
        symbol,                                                                                      
        sv,                                                                                   
        self.commission,                                                                                     
        self.impact,                                                                                     
        )
        # print(portvals)
        cum_ret = (portvals[-1] / portvals[0]) - 1

        return cum_ret		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		   	 			  		 			 	 	 		 		 	
    print("One does not simply think up a strategy")  		  	   		   	 			  		 			 	 	 		 		 	
