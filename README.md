# Technology stocks clustering 



## Introduction
The tech sector is filled with secular growth stories, which profit from disruptive long-term trends, instead of cyclical ones that are tightly tethered to macroeconomic cycles. Trends like 5G communication networks, cloud computing, artificial intelligence, cybersecurity, driverless cars, e-commerce, social networks, and augmented reality are all long-term secular growth stories that could disrupt older markets. Therefore its crusial for modern investors to allocate a porpotion of their portfolio to these sector.


### what is clustering?
Cluster analysis is a technique used to group sets of objects that share similar characteristics. It is common in statistics. Investors will use cluster analysis to develop a cluster trading approach that helps them build a diversified portfolio. Stocks that exhibit high correlations in returns fall into one basket, those slightly less correlated in another, and so on, until each stock is placed into a category.

### Objective
The goal of this project is to group the top 500 technology stocks according to their average return,volatility and other financial metrics.  Those clusters will help investors  choosing the stocks that suit better to their investing style.


## Libraries And Tools
```python
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import yfinance as yf
import datetime

```
## Data collection
### Ticker selection
We will create a ticker list of our stocks according to top 500 technology stocks in Yahoo Finance.
```python
tickers_df = pd.read_csv('stock_tickers.csv') #a list of tickers
tickers_df = tickers_df.rename(columns = {';;;;;;;;;;':'tickers'})
tickers_df['tickers'] = tickers_df['tickers'].apply(str)
tickers_df['tickers'] = tickers_df['tickers'].apply(lambda x : re.sub(r'[^\w\s]','',x))
tickers_df = tickers_df.drop(tickers_df.index[0])
# Tickers of top 500 companies according to market capitalization
tickers = tickers_df['tickers'].tolist()

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
start = datetime.datetime(2015,12,1)
end = datetime.datetime(2020,12,31)
```
**STEP 2**
```python
df_history = pd.DataFrame()

for ticker in tqdm(tickers):
    ticker = ticker
    period1 = int(time.mktime(datetime.datetime(2015, 12, 1, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2020, 12, 31, 23, 59).timetuple()))
    interval = '1d'

    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    try:
        df = pd.read_csv(query_string)
        df['ticker'] = ticker
        print('{} extracted'.format(ticker))
    except:
        continue
    df_history = df_history.append(df)
    sleep(1.5)
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

## Data analysis
### Libraires & Tools
```python
import pandas as pd
import numpy as np
```
```python
data = pd.read_csv('../data/stock_history_data')
stock_metrics = pd.read_csv('../data/stock_data')
```
First we will perform some feature engineering in order to create our final dataframe. The first step is to calculate daily return and Cumulative return  of each stock.

### Daily return
Since we are only interested in specif columns we will keep only the ones we need and give them simpler names. It is always easier to work with lowercase column names and column names that don't contain special characters.
```python
cleaned_df = data[['Date','ticker','Adj Close']]
cleaned_df.columns = ['date','ticker','price']

cleaned_df = cleaned_df.pivot_table(index=['date'], columns=['ticker'], values=['price'])
# flatten columns multi-index, `date` will become the dataframe index
cleaned_df.columns = [col[1] for col in cleaned_df.columns.values]
```
![Screenshot 2021-11-10 at 10 14 18 PM](https://user-images.githubusercontent.com/83364833/141186784-762c83d5-f4e1-42bc-97ca-b063ec589abe.png)

```python
# compute daily returns using pandas pct_change()
df_daily_returns = cleaned_df.pct_change()
# skip first row with NA 
df_daily_returns = df_daily_returns[1:]
df_daily_returns
```
![Screenshot 2021-11-10 at 10 16 52 PM](https://user-images.githubusercontent.com/83364833/141186978-0e8c1e39-141e-4885-b953-380e3a61d695.png)

### Cumulative return

```python
# Calculate the cumulative daily returns
df_cum_daily_returns = (1 + df_daily_returns).cumprod() - 1
df_cum_daily_returns = df_cum_daily_returns.reset_index()
df_cum_daily_returns
```
![Screenshot 2021-11-10 at 10 17 32 PM](https://user-images.githubusercontent.com/83364833/141187051-b327a49c-95dd-4416-9f71-183fd33898e0.png)

Now that we have calculated the cumulative return of each stock we will create a new column in ```stock_metrics``` dataset and fill it with the value of the last row of df_cum_daily_returns for each stock. This row represents the total return of each stock for the above period.

```python
total_return = pd.DataFrame(df_cum_daily_returns.iloc[df_cum_daily_returns.index.tolist()[-1]][1:]).reset_index()

#merging the total return dataframe with the stock metrics dataframe
stock_metrics = stock_metrics.merge(total_return, left_on = 'ticker', right_on = 'index', how = 'outer')

stock_metrics['total return'] = stock_metrics[1279]

# dropping the columns that we dont need
stock_metrics = stock_metrics.drop(columns = [1279,'index','Unnamed: 0'])
```
### Stock Volatility

```python
#At first we will calculate the mean return of each of our stocks .

mean_return = {}
for ticker in df_daily_returns.keys().tolist():
    mean_return[ticker] = df_daily_returns[ticker].mean()

#The second step is to calculate the deviation from the mean for each stock

df_dev = df_daily_returns.copy()

for ticker in df_dev.keys().tolist():
    df_dev[ticker] = df_dev[ticker] - mean_return[ticker]

#In the next step we will square our deviations and find the sum of them for each individual stock. At last, in order to calculate the variance, we will divide those numbers with the total number of days minus one .

df_dev = df_dev.apply(lambda x: x**2)

dev_sum = {}
for ticker in df_dev.keys().tolist():
    dev_sum[ticker] = df_dev[ticker].sum()/(len(df_dev.index)-1)

#And finally, the volatility is the square root of those numbers

#creating the dataframe
df_volatility = pd.DataFrame.from_dict(dev_sum, orient = 'index', columns =['volatility']).reset_index()

#calculating the square root
df_volatility['volatility'] = df_volatility['volatility'].apply(lambda x:x**0.5) 

# merging with the main dataframe
stock_metrics = stock_metrics.merge(df_volatility, left_on = 'ticker', right_on = 'index', how = 'outer')

stock_metrics = stock_metrics.drop(columns = ['index'])

stock_metrics.dropna(inplace = True)
```

### S&P 500 Cumulative return

```python
df_sp500 = pd.read_csv("../Data/sp500_data")

def cumulative_index(df):
    ''' A function that returns the cumulative return of an index'''
    #sellecting the data that we need
    df = df[['Date','Adj Close']]
    #renaming the columns
    df.columns = ['date','price']
    df = df.pivot_table(index = ['date'], values = ['price'])
    #calculating the daily returns of the index
    df = df.pct_change()
    df = df[1:]
    df = (1+df).cumprod() -1
    df = df.reset_index()
    return df

df_sp500 = cumulative_index(df_sp500)
```
### Nasdaq 100  Cumulative return
```python
df_nasdaq = pd.read_csv("../Data/nasdaq100_data")

df_nasdaq = cumulative_index(df_nasdaq)
```
### Exporting the data
```python
df_nasdaq.to_csv('nasdaq_vis')

df_sp500.to_csv('sp500_vis')

df_cum_daily_returns.to_csv('stock_vis')

stock_metrics.to_csv('clustering_data')
```

## Data modelling
### Libraires & Tools
```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer
```

```python
data = pd.read_csv('../data/clustering_data')
```
### Introduction to Clustering
Clustering is defined as dividing data points or population into several groups such that similar data points are in the same groups. The aim to segregate groups based on similar traits.Clustering is defined as dividing data points or population into several groups such that similar data points are in the same groups. The aim to segregate groups based on similar traits.

### K-Means Clustering
The most common clustering covered in machine learning for beginners is K-Means. The first step is to create c new observations among our unlabelled data and locate them randomly, called centroids. The number of centroids represents the number of output classes. The first step of the iterative process for each centroid is to find the nearest point (in terms of Euclidean distance) and assign them to its category. Next, for each category, the average of all the points attributed to that class is computed. The output is the new centroid of the class.

With every iteration, the observations can be redirected to another centroid. After several reiterations, the centroid’s change in location is less critical as initial random centroids converge with real ones—the process ends when there is no change in centroids’ position. Many methods can be employed for the task, but a common one is ‘elbow method’. A low level of variation is needed within the clusters measured by the within-cluster sum of squares. The number of centroids and observations are inversely proportional. Thus, setting the highest possible number of centroids would be inconsistent.

### Hierarchical Clustering
Two techniques are used by this algorithm- Agglomerative and Divisive. In HC, the number of clusters K can be set precisely like in K-means, and n is the number of data points such that n>K. The agglomerative HC starts from n clusters and aggregates data until K clusters are obtained. The divisive starts from only one cluster and then splits depending on similarities until K clusters are obtained. The similarity here is the distance among points, which can be computed in many ways, and it is the crucial element of discrimination. It can be computed with different approaches:

1. Min: Given two clusters C1 and C2 such that point a belongs to C1 and b to C2. The similarity between them is equal to the minimum of distance

2. Max: The similarity between points a and b is equal to the maximum of distance

3. Average: All the pairs of points are taken, and their similarities are computed. Then the average of similarities is the similarity between C1 and C2.

###K-Means vs Hierarchical
As we know, clustering is a subjective statistical analysis, and there is more than one appropriate algorithm for every dataset and type of problem. So how to choose between K-means and hierarchical?

1. If there is a specific number of clusters in the dataset, but the group they belong to is unknown, choose K-means.

2. If the distinguishes are based on prior beliefs, hierarchical clustering should be used to know the number of clusters.

3. With a large number of variables, K-means compute faster.

4. The result of K-means is unstructured, but that of hierarchal is more interpretable and informative.

5. It is easier to determine the number of clusters by hierarchical clustering’s dendrogram.

In our case, we will use Hierarchical and KMeans clustering to create our model. but first, we will drop several columns of our dataset. the columns that we will keep are:

1. Revenue Growth: the percentage that the revenue of a company has grown

2. Profit Margin: profit margin is a measure of a company’s earnings (or profits) relative to its revenue

3. Return on assets: Return on assets (ROA) is an indicator of how well a company utilizes its assets in terms of profitability.

4. Return on equity: Return on equity (ROE) is a measure of a company's financial performance, calculated by dividing net income by shareholders' equity.

5. Debt to equity: The debt-to-equity (D/E) ratio compares a company’s total liabilities to its shareholder equity and can be used to evaluate how much leverage a company is using.

6. Volatility

7. total return

```python
model_data = data[['ticker','revenueGrowth','profitMargins','returnOnAssets','returnOnEquity','debtToEquity','total return', 'volatility']]

X = model_data.drop(columns = ['ticker'])
```
### Data scaling
```python
scaler = StandardScaler()

scaler.fit(X)

scaled_df = scaler.transform(X)
```
### Hierarchical clustering
```python
'''generate the linkage matrix'''
Z = linkage(scaled_df, 'ward')


'''plot the dendogram'''
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Stock')
plt.ylabel('Distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=311,                 
    leaf_rotation=90.,      # rotates the x axis labels
    leaf_font_size=8.,      # font size for the x axis labels
)
plt.axhline(y=19, c='k')
plt.rcParams.update({'text.color': "white",
                     'axes.labelcolor': "white"})
plt.show()
```
![Screenshot 2021-11-10 at 10 28 13 PM](https://user-images.githubusercontent.com/83364833/141188466-5935a086-11a3-4168-894b-c58b70770077.png)

We can see that the blackline, which cut the dendogram in such a way that it cuts the tallest vertical line, cut the dendogram five times. This means that we will use the AgglomerativeClustering algorithm with n_clusters = 4.

```python
'''agglomerative cluster data with n=5'''
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
cluster.fit_predict(scaled_df)
model_data['HC clusters'] = cluster.labels_
```

### K-means clustering
```python
'''k means - determine optimal k'''
K = range(1,10)
kmeans = [KMeans(n_clusters=k) for k in K]
score = [kmeans[k].fit(scaled_df).score(scaled_df) for k in range(len(kmeans))]


# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))
plt.figure(figsize=(15, 10))
visualizer.fit(scaled_df)        # Fit the data to the visualizer
visualizer.poof()        # Draw/show/poof the data
plt.show()
```

![Screenshot 2021-11-10 at 10 30 00 PM](https://user-images.githubusercontent.com/83364833/141188665-9a1f71e9-27e8-4e9c-9201-ceb39d853a8f.png)

```python
# This gives a perspective into the density and separation of the formed clusters
# Instantiate the clustering model and visualizer
model = KMeans(n_clusters=4, random_state=42)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
plt.figure(figsize=(15, 10))
visualizer.fit(scaled_df)        # Fit the data to the visualizer
visualizer.poof()  # show the data
```

![Screenshot 2021-11-10 at 10 30 52 PM](https://user-images.githubusercontent.com/83364833/141188764-3a8a1bbc-295e-4d23-a461-d215dd602287.png)

So, now we will create our model with four clusters

```python
model = KMeans(
        n_clusters=4,
        init='k-means++',
        random_state=42)

model = model.fit(scaled_df)

predicts = model.predict(scaled_df)

model_data['KM clusters'] = predicts
```
### PCA decomposition

```python
pca = PCA(n_components=3)

pca.fit(scaled_df)

reduced_arr = pca.transform(scaled_df)

reduced_df = pd.DataFrame(reduced_arr, columns = ['pca1','pca2','pca3'])

reduced_df['HC clusters'] = model_data['HC clusters']

reduced_df['KM clusters'] = predicts

reduced_df['ticker'] = model_data['ticker']

reduced_df['return']= model_data['total return']

reduced_df['volatility']= model_data['volatility']
```

### hierarchical clustering visualisation

```python
fig = px.scatter_3d(reduced_df, x='pca1', y='pca2', z='pca3',
                    color='HC clusters', symbol='KM clusters',opacity=0.7)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
```
![Screenshot 2021-11-10 at 10 35 55 PM](https://user-images.githubusercontent.com/83364833/141189464-c1cd5bce-bafa-46fe-9117-080942338ba5.png)

```python
fig = px.scatter(reduced_df, x='pca1', y='pca2', color='HC clusters', opacity = 0.7)
fig.show()
```
![Screenshot 2021-11-10 at 10 36 20 PM](https://user-images.githubusercontent.com/83364833/141189545-b0a99efb-46c3-4c13-9ebe-a22ae5cd7259.png)

###Kmeans Clustering visualisation

![Screenshot 2021-11-10 at 10 38 37 PM](https://user-images.githubusercontent.com/83364833/141189753-1269f4b2-e8be-4593-add7-46fa515d189b.png)


![Screenshot 2021-11-10 at 10 39 04 PM](https://user-images.githubusercontent.com/83364833/141189816-481892f5-37e1-473b-afd6-349dc16d10ab.png)

Although with both algorithms we found that the optimal number of clusters is four, the two algorithms proceed with different clustering. Now we will create a table with mean volatility and total return for each cluster and algorithm.

```python
metrics_df = pd.DataFrame(columns = ['cluster','Avg return','Avg volatility','method'])

for method in ['HC clusters','KM clusters']:    
    for cluster in range(0,4):
        filter_df = reduced_df[reduced_df[method] == cluster]
        append_dict = {'cluster': cluster+1,'Avg return':filter_df['return'].mean(),'Avg volatility':filter_df['volatility'].mean(),'method':method}
        metrics_df = metrics_df.append(append_dict,ignore_index = True)

 metrics_df['Avg volatility'] =  metrics_df['Avg volatility']*100 

 metrics_df
```
![Screenshot 2021-11-10 at 10 40 18 PM](https://user-images.githubusercontent.com/83364833/141189978-d27de171-4ec1-45f3-9c3d-ea341d62cd48.png)

As we can see, both algorithms have created one cluster that has the highest average return, two custers that have similar return rates and one that has a return rate below 0.

### Conclusion
Now we have created four buckets of stocks, with different attributes, thus, every investor has some more insights for his stock peaking procedure. for further detail about each stock and its metrics please visit the app.

