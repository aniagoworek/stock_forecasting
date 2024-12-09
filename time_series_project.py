import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
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

# roznicowanie
df['log_diff'] = df['log_close'].diff()
adfuller_test(df['log_diff'].dropna())

# Wykresy korelogramu (ACF i PACF) po roznicowaniu
plt.figure(figsize=(12, 6))
plot_acf(df['log_diff'].dropna(), lags=80, ax=plt.gca())
plt.savefig('acf_plot2.png')
plt.close()

plt.figure(figsize=(12, 6))
plot_pacf(df['log_diff'].dropna(), lags=80, ax=plt.gca())
plt.savefig('pacf_plot2.png')
plt.close()

# Dopasowanie modelu ARIMA (1, 1, 0)
model = ARIMA(df['log_close'], order=(1, 1, 0))
model_fit = model.fit()
print(model_fit.summary())
model_fit.plot_diagnostics(figsize=(10, 8))
plt.savefig('ARIMA.png')
residuals = model_fit.resid

# Test ADF na resztach
adf_test = adfuller(residuals)
print("ADF Test Statistic:", adf_test[0])
print("p-value:", adf_test[1])

if adf_test[1] < 0.05:
    print("Reszty są stacjonarne.")
else:
    print("Reszty nie są stacjonarne.")

# test na autokorelacje reszt
ljung_box_test = acorr_ljungbox(residuals, lags=10)
print("Test Ljunga-Boxa:")
print(ljung_box_test)

if all(ljung_box_test['lb_pvalue'] > 0.05):
    print("Reszty są nieskorelowane.")
else:
    print("Reszty są skorelowane.")


# prognozowanie z arima 110
# Prognoza na 30 dni
# forecast = model_fit.get_forecast(steps=30)
# forecast_ci = forecast.conf_int()

# # Wykres prognozy
# plt.figure(figsize=(10, 6))
# plt.plot(df.index, df['close'], label='History')
# plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, label='Forecast', color='red')
# plt.fill_between(forecast_ci.index, 
#                  forecast_ci.iloc[:, 0], 
#                  forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
# plt.legend()
# plt.show()

# model sarimax
# Dopasowanie modelu SARIMA(1, 1, 0)(1, 1, 0, 12)
sarima_model = SARIMAX(df['log_close'], order=(1, 1, 0), seasonal_order=(1, 1, 0, 12))
sarima_model_fit = sarima_model.fit()


print(sarima_model_fit.summary())
sarima_model_fit.plot_diagnostics(figsize=(10, 8))
plt.savefig('SARIMA_12.png')

# Test Ljunga-Boxa dla modelu SARIMA
ljung_box_test_sarima = acorr_ljungbox(sarima_model_fit.resid, lags=10)
print("Test Ljunga-Boxa dla SARIMA:")
print(ljung_box_test_sarima)