import matplotlib.pyplot as plt
# import argparse
import pickle
import pandas as pd
from collections import OrderedDict
# import time
# import re

# from NNAgent import DQNNAgent

# parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
# args = parser.parse_args()


# df = pd.read_csv('close_pricesGOOGLAMZNAAPL.csv', parse_dates=True, index_col=0)
# df = df[1006:]
# # print(df_new.head())
# # print(df_new.tail())
# print(df.head())
# print(df.tail())


def uniform_portfolio(x, y, z):
    return 287*x + 77*y + 239*z + 20


def uniform_portfolio1(x, y, z):
    return 19.4*x + 69.2*y + 952*z


def plotting():
    tickers = ['1', '10', '100', '200', '400', '600', '800', '1000', '1200', '1400', '1600', '1800', '2000']
    a = OrderedDict()
    left = pd.DataFrame()

    for ticker in tickers:
        a[ticker] = pickle.load(open("episode_pnl/202102101241-train-{}.p".format(ticker), "rb"))
        if left.empty:
            left = pd.DataFrame({'ep{}'.format(ticker): a[ticker]})
            print(left)
        else:
            right = pd.DataFrame({'ep{}'.format(ticker): a[ticker]})
            print(right)
            left = left.join(right)
            print(left)
    print(left)
    plt.plot(left)
    plt.legend(['ep1', 'ep10', 'ep100', 'ep200', 'ep400', 'ep600', 'ep800', 'ep1000',
                'ep1200', 'ep1400', 'ep1600', 'ep1800', 'ep2000'])
    plt.show()


# plotting()


def uniformplotting():
    df = pd.read_csv('close_pricesABTMMMMSFT.csv', parse_dates=True, index_col=0)

    df = df[756:]

    df['add'] = df.apply(lambda row: uniform_portfolio(row['ABT'], row['MMM'], row['MSFT']), axis=1)

    left = df['add'].to_frame()

    s = pickle.load(open("episode_pnl/202102030048-test-1-e0.1.p", "rb"))

    right = pd.DataFrame({'TEST': s})
    right.index = left.index
    print(right)

    left = left.join(right)
    print(left)

    plt.plot(left)
    plt.legend(['Buy & Hold', '64x64-lr0.1-e0.1'])
    plt.ylabel('Portfolio Value')
    plt.xticks(rotation=45)
    plt.show()


# uniformplotting()

def uniformplotting1():
    df = pd.read_csv('close_pricesGOGLAMZNAAPL.csv', parse_dates=True, index_col=0)

    # plt.plot(df)
    # plt.show()

    df = df[756:]

    # plt.plot(df)
    # plt.show()

    df['add'] = df.apply(lambda row: uniform_portfolio1(row['GOOGL'], row['AMZN'], row['AAPL']), axis=1)

    left = df['add'].to_frame()

    # plt.plot(left)
    # plt.show()

    s = pickle.load(open("episode_pnl/202102030048-test-1-GAA-e0.1.p", "rb"))

    # plt.plot(s)
    # plt.show()

    right = pd.DataFrame({'TEST': s})
    right.index = left.index
    print(right)

    left = left.join(right)
    print(left)

    plt.plot(left)
    plt.legend(['Buy & Hold', '64x64-lr0.1-e0.1'])
    plt.ylabel('Portfolio Value')
    plt.xticks(rotation=45)
    plt.show()


uniformplotting1()


def densityplotting():
    s = pickle.load(open("portfolio_val/202102030048-test-GAA-e0.1.p", "rb"))
    plt.hist(s, bins=25)
    plt.ylabel('No. of Episodes')
    plt.xlabel('Portfolio Value')
    plt.show()


densityplotting()


def metrics():
    left = pd.DataFrame(pickle.load(open("episode_pnl/202102030048-test-1-e0.1.p", "rb"))).pct_change().fillna(0)
    print(left)
    print('MEAN=', left.mean())
    print('STD=', left.std())
    sharpe = (755 ** 0.5) * (left.mean() / left.std())

    print('Sharpe Ratio=', sharpe)

    t = pd.DataFrame(pickle.load(open("portfolio_val/202102030048-test-e0.1.p", "rb")))
    print('Average Reward=', t.mean())


# metrics()
