import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import warnings

from functions import adfuller_test, optimize_arima, plot_acf_pacf, kruskal_test, decompose_series

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 8)

# Load and preprocess data
df = pd.read_csv('NKE.csv', header=0)
print(df.head())
print(df.describe())
df = df.drop(['open', 'high', 'low', 'volume'], axis=1)
df = df.loc[9354:11095] #9354
df['date'] = pd.to_datetime(df['date'])
print(df.head())
df.set_index('date', inplace=True)
df['log_close'] = np.log(df['close'])

df = df.dropna()
print(df.head())
print(df.tail())

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
plt.title('Nike Stocks from 2018 to 2024')
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


df['log_seasonal_diff'] = df['log_close'].diff(30) 
plot_acf_pacf(df['log_seasonal_diff'].dropna(), 'acf_seasonal_diff.png', 'pacf_seasonal_diff.png')

# # Optimize ARIMA model
# ps = range(0, 8, 1)
# d = 1
# qs = range(0, 8, 1)
# parameters_list = [(p, d, q) for p in ps for q in qs]
# result_df = optimize_arima(df['log_close'], parameters_list)
# print(result_df)

# # Best fit for ARIMA model
# best_model = SARIMAX(df['log_close'], order=(2,1,1), simple_differencing=False) #2,1,1 do log
# model_fit = best_model.fit(disp=False)
# print(model_fit.summary())
# model_fit.plot_diagnostics(figsize=(10, 8))
# plt.savefig('ARIMA.png')
# plt.close()

# # Residuals analysis
# residuals = model_fit.resid
# adfuller_test(residuals)
# ljung_box_test = acorr_ljungbox(residuals, lags=13) # 24, 36 wyprobowac
# print("Ljung-Box Test Results:")
# print(ljung_box_test)
# if all(ljung_box_test['lb_pvalue'] > 0.05):
#     print("Residuals are correlated.")
# else:
#     print("Residuals are not correlated.")

# # Wykres dopasowania modelu
# start_date = pd.to_datetime('2018-01-05 00:00:00') #2018-01-05
# end_date = pd.to_datetime('2024-12-05 00:00:00')
# model_predict = model_fit.predict(start=start_date, end=end_date)

# plt.figure(figsize=(10, 8))
# plt.plot(model_predict[start_date:end_date], color = "orange", label='Prediction')
# plt.plot(df['log_close'][start_date:end_date],color="blue", label='Actual')
# plt.title("Prediction vs Actual ", size = 24)
# plt.legend()
# plt.savefig('model_fit_vs_actual.png')
# plt.show()
# plt.close()

# # Generowanie wykresu z przedziałami ufności
# prediction = model_fit.get_prediction(start=start_date, end=end_date)
# confidence_intervals = prediction.conf_int()
# forecast = prediction.predicted_mean
# lower_limits = confidence_intervals.iloc[:, 0]
# upper_limits = confidence_intervals.iloc[:, 1]

# plt.figure(figsize=(10, 8))
# plt.plot(forecast, color='orange', label='Forecast')
# plt.plot(df['log_close'][start_date:end_date], color='blue', label='Actual')
# plt.fill_between(forecast.index, lower_limits, upper_limits, color='pink', alpha=0.3, label='Confidence Interval')
# plt.title("Prediction vs Actual with Confidence Intervals", size=24)
# plt.legend()
# plt.savefig('model_fit_with_confidence_intervals.png')
# plt.close()

# Model GARCH - lepszy do danych finansowcyh

# Model GARCH
returns = df['log_close'].dropna() * 100  # Przekształcenie na zmienność procentową

# Ustawienie modelu GARCH(1,1)
garch_model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='normal')

# Dopasowanie modelu
garch_fit = garch_model.fit(update_freq=5, disp="off")
print(garch_fit.summary())

# Wizualizacja wyników
plt.figure(figsize=(10, 6))
plt.plot(returns.index, returns, label="Returns")
plt.plot(returns.index, garch_fit.conditional_volatility, color='orange', label="Conditional Volatility")
plt.title("GARCH Model - Returns and Conditional Volatility")
plt.legend()
plt.savefig('garch_returns_volatility.png')
plt.close()

# GARCH
forecast_horizon = 30 
forecast = garch_fit.forecast(horizon=forecast_horizon)

# Sprawdzenie dostępnych kolumn w prognozie
print(forecast.mean.head())  # Wyświetl dostępne prognozy średniej
print(forecast.variance.head())  # Wyświetl dostępne prognozy wariancji

forecast_mean = forecast.mean.iloc[-1]  # Ostatnia prognozowana średnia
forecast_volatility = np.sqrt(forecast.variance.iloc[-1])  # Ostatnia prognozowana wariancja

# Wizualizacja prognozy
forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_horizon + 1, freq='B')[1:]

plt.figure(figsize=(10, 6))
plt.plot(forecast_dates, forecast_mean, label="Forecasted Mean", color="orange")
plt.fill_between(forecast_dates, forecast_mean - forecast_volatility, 
                 forecast_mean + forecast_volatility, color='pink', alpha=0.3, label="Confidence Interval")
plt.title("GARCH Model Forecast")
plt.legend()
plt.savefig('garch_forecast.png')
plt.close()

# Wykres historycznych danych i prognozowanych danych - naprawic
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['log_close'], label='Historical Data (Log Close)', color='blue', alpha=0.6)
plt.plot(forecast_dates, forecast_mean, label="Forecasted Mean", color="orange", linestyle='--')
plt.fill_between(forecast_dates, forecast_mean - forecast_volatility, 
                 forecast_mean + forecast_volatility, color='pink', alpha=0.3, label="Confidence Interval")

plt.title("Historical vs Forecasted Data with GARCH Model")
plt.legend()
plt.xlabel('Date')
plt.ylabel('Log Close')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('garch_forecast_vs_history.png')
plt.show()