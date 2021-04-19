import datetime as dt
import pandas as pd
import pandas_datareader.data as web
from collections import OrderedDict

# set the dates
start = dt.datetime(2005, 1, 1)
end = dt.datetime(2012, 12, 31)

tickers = ['CBS', 'MMM', 'MSFT', 'VIAC']
for ticker in tickers:
    df = web.DataReader('GOOGL', 'yahoo', start, end)
    df.to_csv('googl.csv')
    df = web.DataReader('AMZN', 'yahoo', start, end)
    df.to_csv('amzn.csv')
    df = web.DataReader('AAPL', 'yahoo', start, end)
    df.to_csv('aapl.csv')

closedata = OrderedDict()

resultclose = pd.DataFrame()
tickers = ['GOOGL', 'AMZN', 'AAPL']

for ticker in tickers:
    csvData = pd.read_csv('{}.csv'.format(ticker), index_col=0, parse_dates=['Date'])
    # data[ticker] =csvData[['Open','High','Low','Close','Volume', 'Adj Close']]
    closedata[ticker] = csvData[['Close']]

    if resultclose.empty:
        resultclose = closedata[ticker].rename(columns={'Close': ticker})
    else:
        resultclose = resultclose.join(closedata[ticker].rename(columns={'Close': ticker}))

resultclose.to_csv('close_pricesGOGLAMZNAAPL.csv')
