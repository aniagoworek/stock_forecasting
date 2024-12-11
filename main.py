import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

from functions import adfuller_test, optimize_ARIMA, plot_acf_pacf, kruskal_test, decompose_series

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 8)

# Load and preprocess data
df = pd.read_csv('NKE.csv', header=0)
print(df.head())
print(df.describe())
df = df.drop(['open', 'high', 'low', 'volume'], axis=1)
df = df.loc[10300:11095]
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df['log_close'] = np.log(df['close'])
df = df.dropna()

# Plot histograms
sns.distplot(df['close'], bins=20, hist=True, kde=True, color='#1a488f')
plt.title('Histogram of Nike Stock Prices')
plt.savefig('nike_stock_histogram.png')
plt.close()

sns.distplot(df['log_close'], bins=20, hist=True, kde=True, color='#1a488f')
plt.title('Histogram of Log-transformed Nike Stock Prices')
plt.savefig('nike_log_stock_histogram.png')
plt.close()

# Scatter plot
sns.scatterplot(df['close'])
plt.title('Nike Stocks from 2021 to 2024')
plt.savefig('nike_scatterplot.png')
plt.close()

# Time series decomposition
decompose_series(df['log_close'], period=12, filename='time_series_decomposition.png')

plot_acf_pacf(df['log_close'], 'acf_plot.png', 'pacf_plot.png')
adfuller_test(df['log_close'])
kruskal_test(df['log_close'], df.index)

# Differencing for stationarity
df['log_diff'] = df['log_close'].diff()
adfuller_test(df['log_diff'])
plot_acf_pacf(df['log_diff'], 'acf_plot_diff.png', 'pacf_plot_diff.png')

# Optimize ARIMA model
ps = range(0, 8, 1)
d = 1
qs = range(0, 8, 1)
parameters_list = [(p, d, q) for p in ps for q in qs]
result_df = optimize_ARIMA(df['log_close'], parameters_list)
print(result_df)

# Best fit for ARIMA model
best_model = SARIMAX(df['log_close'], order=(0, 1, 0), simple_differencing=True)
model_fit = best_model.fit(disp=False)
print(model_fit.summary())
model_fit.plot_diagnostics(figsize=(10, 8))
plt.savefig('ARIMA.png')

# ARIMA (1,1,0)
model_110 = SARIMAX(df['log_close'], order=(0, 1, 0), simple_differencing=True)
model110_fit = best_model.fit(disp=False)
print(model110_fit.summary())
model110_fit.plot_diagnostics(figsize=(10, 8))
plt.savefig('ARIMA110.png')

# Residuals analysis
residuals = model_fit.resid
adfuller_test(residuals)
ljung_box_test = acorr_ljungbox(residuals, lags=10)
print("Ljung-Box Test Results:")
print(ljung_box_test)
if all(ljung_box_test['lb_pvalue'] > 0.05):
    print("Residuals are correlated.")
else:
    print("Residuals are not correlated.")

residuals_arima110 = model110_fit.resid
adfuller_test(residuals_arima110)
ljung_box_test = acorr_ljungbox(residuals_arima110, lags=10)
print("Ljung-Box Test Results:")
print(ljung_box_test)
if all(ljung_box_test['lb_pvalue'] > 0.05):
    print("Residuals are correlated.")
else:
    print("Residuals are not correlated.")