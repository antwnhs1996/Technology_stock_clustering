# Technology stocks clustering 



## Introduction
The tech sector is filled with secular growth stories, which profit from disruptive long-term trends, instead of cyclical ones that are tightly tethered to macroeconomic cycles. Trends like 5G communication networks, cloud computing, artificial intelligence, cybersecurity, driverless cars, e-commerce, social networks, and augmented reality are all long-term secular growth stories that could disrupt older markets. Therefore its crusial for modern investors to allocate a porpotion of their portfolio to these sector.


### what is clustering?
Cluster analysis is a technique used to group sets of objects that share similar characteristics. It is common in statistics. Investors will use cluster analysis to develop a cluster trading approach that helps them build a diversified portfolio. Stocks that exhibit high correlations in returns fall into one basket, those slightly less correlated in another, and so on, until each stock is placed into a category.

### Objective
The goal of this project is to group the top 60 technology stocks according to their average return and volatility.  Those clusters will help investors  choosing the stocks that suit better to their investing. style


## Libraries And Tools
```python
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import yfinance as yf
import datetime

```
### Ticker selection
We will create a ticker list of our stocks according to top 60 technology stocks in Yahoo Finance.
```python
tickers = ['AAPL', 'MSFT', 'NVDA', 'TSM', 'ASML', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'ACN', 'INTC', 'AVGO', 'SHOP', 'SAP', 'TXN', 'INTU', 'QCOM', 'SONY', 'SQ', 'AMAT', 'IBM', 'SNOW', 'TEAM', 'INFY', 'ADI', 'UBER', 'DELL', 'LRCX', 'MU', 'WDAY', 'ADSK', 'DOCU', 'NXPI', 'FTNT', 'KLAC', 'PLTR', 'SNPS', 'DDOG', 'PANW', 'WIT', 'TEL', 'CDNS', 'APH', 'XLNX', 'STM', 'ZS', 'U', 'DIDI', 'MSI', 'MCHP', 'CTSH','TTD','ERIC','OKTA','HUBS','EPAM','APP','HPQ','MDB','NOK']
```

### Collecting the metrics of each stock
First we will create a list of metrics that we need for our clustering. I chose to collect 19 of the major metrics that describe a stock.
```python
keys = ['shortName', 'currentPrice', 'marketCap', 'sector', 'industry', 'profitMargins', 'grossMargins', 'revenueGrowth', 'grossProfits', 'returnOnAssets', 'debtToEquity', 'returnOnEquity', 'totalDebt', 'totalCash', 'totalRevenue', 'exchange', 'market', 'bookValue', 'priceToBook']
```
In order to collect the values of those metrics we will use a library called yfinance:
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict

conda search yfinance --channel conda-forge
```
This library offers a reliable method of downloading market data from Yahoo Finance by utilizing its API. (please refer to this article for further details. [yfinance](https://aroussi.com/post/python-yahoo-finance "yfinance")). The steps for collecting the metrics data are described below.
**STEP 1**
```python
# Creation of the dataframe that we will store our metrics
df_stocks = pd.DataFrame(columns =['ticker'] + keys)
```
**STEP 2**
```python
# we will iterate through every ticker in our list in order to find our metrics. We will utilize the ticker object that returns all the metrics of a specific stock
for ticker in tqdm(tickers):
    
    try:
        #creating the object
        stock = yf.Ticker(ticker.lower())
        stats = [stock.info[key] for key in keys]
        print('{} stats extracted'.format(ticker))
        df_stocks.loc[len(df_stocks)] = [ticker] + stats
        print('{} info extracted'.format(ticker))
    except:
        print('{} raised an exception'.format(ticker))
        continue
```
### Collecting the historical data of each stock
**STEP 1**
```python
# we are difining the timeframe of our research
start = datetime.datetime(2011,1,1)
end = datetime.datetime(2021,1,1)
```
**STEP 2**
```python
# we are difining the timeframe of our research
start = datetime.datetime(2011,1,1)
end = datetime.datetime(2021,1,1)
```
```python
# create empty dataframe
df_history = pd.DataFrame()
# iterate over each ticker
for ticker in tqdm(tickers):  
    # print the symbol which is being downloaded
    print( str(tickers.index(ticker)) + str(' : ') + ticker, sep=',', end=',', flush=True)  
    
    try:
        # download the stock price 
        stock = []
        stock = yf.download(ticker,start=start, end=end, progress=False)
        
        # append the individual stock prices 
        if len(stock) == 0:
            None
        else:
            stock['Name']=ticker
            df_history = df_history.append(stock,sort=False)
    except Exception:
        None
```
Later in this project we will compare the returns of the above stocks with some of the most well known indexes for tech stocks. Those are  Nasdaq-100 and S&P500. For this purpose we will also collect the historical data of those indexes.

### S&P 500
```python
#collecting the historical data of s&p 500 index
df_sp500 = pd.DataFrame()
    
try:
        # download the stock price 
    sp500 = []
    sp500 = yf.download('^GSPC',start=start, end=end, progress=False)
    df_sp500 = df_sp500.append(sp500,sort=False)
        
except Exception:
    None
```
### NASDAQ 100
```python
#collecting the historical data of nasdaq index
df_nasdaq = pd.DataFrame()
    
try:
        # download the stock price 
    nasdaq = []
    nasdaq = yf.download('^NDX',start=start, end=end, progress=False)
    df_nasdaq = df_nasdaq.append(nasdaq,sort=False)
        
except Exception:
    None
```
### Exporting the data
```python
df_stocks.to_csv('stock_data')

df_history.to_csv('stock_history_data')

df_nasdaq.to_csv('nasdaq100_data')

df_sp500.to_csv('sp500_data')
```
