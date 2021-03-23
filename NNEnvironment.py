import numpy as np
import pandas as pd
import itertools
import os
import gym
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import StandardScaler


def get_data():
    df = pd.read_csv('close_pricesABTMMMMSFT.csv', parse_dates=True, index_col=0)
    df.reset_index(drop=True, inplace=True)
    return df.values


def get_data1():
    df = pd.read_csv('close_pricesGOGLAMZNAAPL.csv', parse_dates=True, index_col=0)
    df.reset_index(drop=True, inplace=True)
    return df.values


# CHANGED FROM THE PROGRAM
def get_scaler(env):
    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env._step(action)
        states.append(state)
        if done:
            break
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def get_scaler1(env):
    low = [0] * (env.n_stock * 2 + 1)

    high = []
    max_price = env.stock_price_history.max(axis=1)
    min_price = env.stock_price_history.min(axis=1)
    max_cash = env.initial_investment * 3  # 3 is a magic number...
    max_stock_owned = max_cash // min_price
    for i in max_stock_owned:
        high.append(i)
    for i in max_price:
        high.append(i)
    high.append(max_cash)

    scaler = StandardScaler()
    scaler.fit([low, high])
    return scaler


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class MultiEnv:
    """
      A 3-stock (ABT, MMM, MSFT) trading environment.

      State: [# of stock owned, current stock prices, cash in hand]
        - array of length n_stock * 2 + 1
        - price is discretized (to integer) to reduce state space +FALSE
        - use close price for each stock
        - cash in hand is evaluated at each step based on action performed

      Action: sell (0), hold (1), and buy (2)
        (3^4 possibilities)
        - when selling, sell all the shares
        - when buying, buy as many as cash in hand allows
        - if buying multiple stock, equally distribute cash in hand and then utilize the balance
      """

    def __init__(self, data, initial_investment=20000):
        # data
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape

        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        # action space
        self.action_space = np.arange(3 ** self.n_stock)

        # action_permutations
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

        # calculate size of state
        self.state_dim = self.n_stock * 2 + 1

        # seed and start
        self._reset()

    def _reset(self):
        # self.cur_step = 0
        # self.stock_owned = [0] * self.n_stock
        # self.stock_price = self.stock_price_history[:, self.cur_step]
        # self.cash_in_hand = self.initial_investment
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def _step(self, action):
        assert action in self.action_space

        prev_val = self._get_val()

        # update price===go to the next day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]  # update price

        # perform the trade
        self._trade(action)

        # get the new value after
        cur_val = self._get_val()

        # reward is the Portfolio Value
        reward = cur_val - prev_val

        # done if we run out of data
        done = self.cur_step == self.n_step - 1

        # store the current portfolio value
        info = {'cur_val': cur_val}

        # confirm to the gym API
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        # obs = []
        # obs.extend(self.stock_owned)
        # obs.extend(list(self.stock_price))
        # obs.append(self.cash_in_hand)
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2 * self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        # all combo to sell(0), hold(1), or buy(2) stocks
        action_vec = self.action_list[action]

        # one pass to get sell/buy index
        sell_index = []
        buy_index = []
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # two passes: sell first, then buy; might be naive in real-world settings
        if sell_index:
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1  # buy one share
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False


class TradingEnv(gym.Env):
    """
      A 4-stock (ABT, MMM, MSFT, AAPL) trading environment.

      State: [# of stock owned, current stock prices, cash in hand]
        - array of length n_stock * 2 + 1
        - price is discretized (to integer) to reduce state space +FALSE
        - use close price for each stock
        - cash in hand is evaluated at each step based on action performed

      Action: sell (0), hold (1), and buy (2)
        (3^4 possibilities)
        - when selling, sell all the shares
        - when buying, buy as many as cash in hand allows
        - if buying multiple stock, equally distribute cash in hand and then utilize the balance
      """

    def __init__(self, data, initial_investment=20000):
        # data
        self.stock_price_history = data
        self.n_stock, self.n_step = self.stock_price_history.shape

        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        # action space
        self.action_space = spaces.Discrete(3 ** self.n_stock)

        # observation space: give estimates in order to sample and build scaler
        # stock_max_price = self.stock_price_history.max(axis=1)
        # stock_range = [[0, initial_investment * 2 // mx] for mx in stock_max_price]
        # price_range = [[0, mx] for mx in stock_max_price]
        # cash_in_hand_range = [[0, initial_investment * 2]]
        # self.observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range)
        self.state_dim = self.n_stock * 2 + 1

        # seed and start
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.cur_step = 0
        self.stock_owned = [0] * self.n_stock
        self.stock_price = self.stock_price_history[:, self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def _step(self, action):
        assert self.action_space.contains(action)

        prev_val = self._get_val()

        # update price===go to the next day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[:, self.cur_step]  # update price

        # perform the trade
        self._trade(action)

        # get the new value after
        cur_val = self._get_val()

        # reward is the Portfolio Value
        reward = cur_val - prev_val

        # done if we run out of data
        done = self.cur_step == self.n_step - 1

        # store the current portfolio value
        info = {'cur_val': cur_val}

        # confirm to the gym API
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = []
        obs.extend(self.stock_owned)
        obs.extend(list(self.stock_price))
        obs.append(self.cash_in_hand)
        # obs = np.empty(self.state_dim)
        # obs[:self.n_stock] = self.stock_owned
        # obs[self.n_stock:2 * self.n_stock] = self.stock_price
        # obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self):
        return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        # all combo to sell(0), hold(1), or buy(2) stocks
        action_combo = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))  # changed
        action_vec = action_combo[action]

        # one pass to get sell/buy index
        sell_index = []
        buy_index = []
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # two passes: sell first, then buy; might be naive in real-world settings
        if sell_index:
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1  # buy one share
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False
