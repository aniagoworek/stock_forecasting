import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy.stats import kruskal
# from tqdm import tqdm_notebook
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.regression.linear_model import yule_walker

from itertools import product

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize']= (20,8)

df = pd.read_csv('NKE.csv', header=0)
# print(df.head())
# print(df.describe())
df = df.drop(['open', 'high', 'low','volume'], axis=1)
# print(df.head())

print(df.shape)
df = df.loc[10300:11095]
print(df.head())

df['date']=pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

df['log_close'] = np.log(df['close'])
# print(df.head())

plt.figure(figsize=(10, 6))
sns.distplot(df['close'],bins=20, hist=True, kde=True, color='#1a488f')
plt.title('Histogram of Nike Stock Prices', fontsize=16)
plt.xlabel('Nike Stock Prices', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5, color='grey')
plt.tight_layout()
plt.savefig('nike_stock_histogram.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.distplot(df['log_close'],bins=20, hist=True, kde=True, color='#1a488f')
plt.title('Histogram of Log-transformed Nike Stock Prices', fontsize=16)
plt.xlabel('Log-transformed Nike Stock Prices', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5, color='grey')
plt.tight_layout()
plt.savefig('nike_log_stock_histogram.png')
plt.close()

plt.figure(figsize=(10, 8))
sns.scatterplot(df['close'])
plt.title('Nike Stocks from 2021 to 2024', fontsize=16)
plt.ylabel('Stocks', fontsize=14)
plt.xlabel('Months', fontsize=14)
plt.xticks(rotation=90)
plt.savefig('nike_scatterplot.png')
plt.close()

df = df.dropna()

# Dekompozycja szeregu czasowego
decomposition = seasonal_decompose(df['log_close'], model='additive', period=12)

# Wykres dekompozycji
plt.figure(figsize=(20, 8))
plt.subplot(411)
plt.plot(decomposition.observed, label='Observed')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='orange')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonality', color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals', color='red')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('time_series_decomposition.png')
plt.close()

# Wykresy korelogramu (ACF i PACF)
plt.figure(figsize=(12, 6))
plot_acf(df['log_close'].dropna(), lags=40, ax=plt.gca())
plt.savefig('acf_plot.png')
plt.close()

plt.figure(figsize=(12, 6))
plot_pacf(df['log_close'].dropna(), lags=40, ax=plt.gca())
plt.savefig('pacf_plot.png')
plt.close()

# Test stacjonarności - Augmented Dickey-Fuller (ADF)
def adfuller_test(sales):
    result = adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','#Observation Used']
    for value,label in zip(result,labels):
        print(label  + ': ' + str(value))
    if result[1]<=0.05:
        print('Strong evidence against the null hypothesis, Hence REJECT Ho. and The series is Stationary')
    else:
        print('week evidence against null hypothesis, Hence ACCEPT Ho. that the series is not stationary.')

adfuller_test(df['log_close'])

# Test na obecność sezonowości - Kruskal-Wallis
df['month'] = df.index.month
seasonality_test = kruskal(*[df['log_close'][df['month'] == m] for m in range(1, 13)])
print("Kruskal-Wallis Statistic:", seasonality_test.statistic)
print("p-value:", seasonality_test.pvalue)
if seasonality_test.pvalue < 0.05:
    print("The series shows significant seasonality.")
else:
    print("The series does not show significant seasonality.")

# roznicowanie - zrobic inne/ pomyslec nad tym aby osiagnac stacjonarnosc
df['log_diff'] = df['log_close'].diff()
adfuller_test(df['log_diff'].dropna())

# Wykresy korelogramu (ACF i PACF) po roznicowaniu
plt.figure(figsize=(12, 6))
plot_acf(df['log_diff'].dropna(), lags=40, ax=plt.gca())
plt.savefig('acf_plot2.png')
plt.close()

plt.figure(figsize=(12, 6))
plot_pacf(df['log_diff'].dropna(), lags=40, ax=plt.gca())
plt.savefig('pacf_plot2.png')
plt.close()
