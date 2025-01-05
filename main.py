import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

from functions import adfuller_test, plot_acf_pacf, kruskal_test, decompose_series

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 8)

# Load and preprocess data
df = pd.read_csv('NKE.csv', header=0)
print(df.head())
print(df.describe())
df = df.drop(['open', 'high', 'low', 'volume'], axis=1)
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
plt.title('Nike Stocks from 1980 to 2024')
plt.savefig('nike_scatterplot.png')
plt.close()

# Time series decomposition
decompose_series(df['close'], period=252, filename='time_series_decomposition.png')

moving_avg = df['log_close'].rolling(12).mean()
std_dev = df['log_close'].rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average')
plt.plot(std_dev, color ="black", label = "Standard Deviation")
plt.plot(moving_avg, color="red", label = "Mean")
plt.legend()
plt.savefig('moving average.png') # srednia kroczaca
plt.close()

# check stationarity after log
adfuller_test(df['close'])
adfuller_test(df['log_close'])
plot_acf_pacf(df['log_close'], 'acf_plot.png', 'pacf_plot.png')
kruskal_test(df['log_close'], df.index)

# Delete trend, seasonality, leave residuals (for stationarity)
result = seasonal_decompose(df['log_close'], model='additive', period=12)
result.plot()
plt.savefig('residuals from time series.png')
plt.close()

residuals = result.resid.dropna()
adfuller_test(residuals)

#split data into train and training set
train_data, test_data = residuals[3:int(len(residuals)*0.9)], residuals[int(len(residuals)*0.9):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(residuals, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()
plt.savefig('train and test data.png')
plt.close()

model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
print(model_autoARIMA.summary())
model_autoARIMA.plot_diagnostics(figsize=(15,8))

# Best fit for ARIMA model
model = SARIMAX(train_data, order=(3,0,0), simple_differencing=False)
model_fit = model.fit(disp=False)
print(model_fit.summary())
model_fit.plot_diagnostics(figsize=(10, 8))
plt.savefig('ARIMA (3,0,0) diagnostics.png')
plt.close()

# Residuals analysis
residual_model = model_fit.resid
adfuller_test(residuals)
ljung_box_test = acorr_ljungbox(residual_model, lags=10)
print("Ljung-Box Test Results:")
print(ljung_box_test)
if all(ljung_box_test['lb_pvalue'] > 0.05):
    print("Residuals are correlated.")
else:
    print("Residuals are not correlated.")

# Model fit for training data
start_date = pd.to_datetime('1980-12-15 00:00:00')
end_date = pd.to_datetime('2020-07-01 00:00:00')
model_predict = model_fit.predict(start=start_date, end=end_date)

prediction = model_fit.get_prediction(start=start_date, end=end_date)
confidence_intervals = prediction.conf_int()
forecast = prediction.predicted_mean
lower_limits_train = confidence_intervals.iloc[:, 0]
upper_limits_train = confidence_intervals.iloc[:, 1]

# Add components
forecast_with_components = forecast + result.trend[start_date:end_date] + result.seasonal[start_date:end_date]
forecast_with_components = forecast_with_components.dropna()
train_data_with_components = train_data + result.trend[train_data.index] + result.seasonal[train_data.index]
train_data_with_components = train_data_with_components.dropna()

plt.figure(figsize=(10, 8))
plt.plot(train_data_with_components[forecast_with_components.index], color='orange', label='Actual (Train Data)')
plt.plot(forecast_with_components, color='blue', label='Forecast')
plt.fill_between(forecast_with_components.index, lower_limits_train + result.trend[forecast_with_components.index] + result.seasonal[forecast_with_components.index],
                 upper_limits_train + result.trend[forecast_with_components.index] + result.seasonal[forecast_with_components.index],
                 color='pink', alpha=0.3, label='Confidence Interval')
plt.title("Prediction vs train data", size=24)
plt.legend()
plt.savefig('prediction vs train data.png')

# Forecast for test data
test_start_date = pd.to_datetime('2020-07-02 00:00:00')
test_end_date = pd.to_datetime('2024-11-26 00:00:00')

test_start_date = residuals.index[-len(test_data)]
test_end_date = residuals.index[-1]

test_prediction = model_fit.get_prediction(start=len(train_data), end=len(train_data) + len(test_data) - 1)
test_confidence_intervals = test_prediction.conf_int()
test_forecast = test_prediction.predicted_mean
test_lower_limits = test_confidence_intervals.iloc[:, 0]
test_upper_limits = test_confidence_intervals.iloc[:, 1]

test_forecast.index = test_data.index

# Add components to test forecast
test_forecast_with_components = test_forecast + result.trend[test_forecast.index] + result.seasonal[test_forecast.index]
test_forecast_with_components = test_forecast_with_components.dropna()
test_data_with_components = test_data + result.trend[test_data.index] + result.seasonal[test_data.index]
test_data_with_components = test_data_with_components.dropna()

# # Plot forecast vs test data
plt.figure(figsize=(10, 8))
plt.plot(test_data_with_components[test_forecast_with_components.index], color='gray', label='Actual (Test Data)')
plt.plot(test_forecast_with_components, color='blue', label='Forecast')
plt.title("Prediction vs Test Data", size=24)
plt.legend()
plt.savefig('prediction vs test data.png')
plt.close()

plt.figure(figsize=(12, 8))
plt.plot(result.trend.index, result.trend + result.seasonal, color='gray', alpha=0.5, label='Actual data')
plt.plot(test_data.index, test_data + result.trend[test_data.index] + result.seasonal[test_data.index], color='gray')
plt.plot(test_forecast.index, test_forecast_with_components, color='blue', label='Forecast')
plt.axvline(x=test_data.index[0], color='black', linestyle='--')
plt.title("Nike Stock Forecast", size=24)
plt.legend()
plt.savefig('forecast.png')
plt.close()

# Model performance
mse = mean_squared_error(test_data, test_forecast)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, test_forecast)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, test_forecast))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(test_forecast - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))