import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima

from functions import adfuller_test, plot_acf_pacf, kruskal_test, decompose_series

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 8)

# Load and preprocess data
df = pd.read_csv('NKE.csv', header=0)
print(df.head())
print(df.describe())
df = df.drop(['open', 'high', 'low', 'volume'], axis=1)
# df = df.loc[9354:11095] #9354
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
decompose_series(df['close'], period=12, filename='time_series_decomposition.png')

moving_avg = df['log_close'].rolling(12).mean()
std_dev = df['log_close'].rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average')
plt.plot(std_dev, color ="black", label = "Standard Deviation")
plt.plot(moving_avg, color="red", label = "Mean")
plt.savefig('moving average.png') # srednia kroczaca
plt.close()

adfuller_test(df['close'])
adfuller_test(df['log_close'])
plot_acf_pacf(df['log_close'], 'acf_plot.png', 'pacf_plot.png')
kruskal_test(df['log_close'], df.index)

# usuwanie sezonowosci i trendu dla stacjonarnosci, zostawienie reszt
result = seasonal_decompose(df['log_close'], model='additive', period=12)
result.plot()
plt.savefig('residuals from time series.png')
plt.close()

residuals = result.resid.dropna()
adfuller_test(residuals)

print(residuals.head())
print(residuals.tail())

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

print(train_data.head())
print(train_data.tail())
print("Test Data:", test_data.head())

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
plt.show()

# Best fit for ARIMA model
model = SARIMAX(train_data, order=(3,0,0), simple_differencing=False) #2,1,1 do log
model_fit = model.fit(disp=False)
print(model_fit.summary())
model_fit.plot_diagnostics(figsize=(10, 8))
plt.savefig('ARIMA.png')
plt.close()

# Residuals analysis
residual_model = model_fit.resid
adfuller_test(residuals)
ljung_box_test = acorr_ljungbox(residual_model, lags=13) # 24, 36 wyprobowac
print("Ljung-Box Test Results:")
print(ljung_box_test)
if all(ljung_box_test['lb_pvalue'] > 0.05):
    print("Residuals are correlated.")
else:
    print("Residuals are not correlated.")

# Dopasowanie modelu do danych treningowych
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
plt.plot(train_data_with_components[forecast_with_components.index], color='blue', label='Actual (Train Data with Components)')
plt.plot(forecast_with_components, color='orange', label='Forecast')
plt.fill_between(forecast_with_components.index, lower_limits_train + result.trend[forecast_with_components.index] + result.seasonal[forecast_with_components.index],
                 upper_limits_train + result.trend[forecast_with_components.index] + result.seasonal[forecast_with_components.index],
                 color='pink', alpha=0.3, label='Confidence Interval')
plt.title("Prediction vs train data", size=24)
plt.legend()
plt.savefig('prediction_vs_train_with_components.png')

# Forecast
test_data_with_components = test_data + result.trend[test_data.index] + result.seasonal[test_data.index]
test_data_with_components = test_data_with_components.dropna()

# Forecast
forecast_result = model_fit.get_forecast(steps=len(test_data))
fc = forecast_result.predicted_mean
conf = forecast_result.conf_int(alpha=0.05)
lower_limits_test = conf.iloc[:, 0]
upper_limits_test = conf.iloc[:, 1]

fc_series = pd.Series(fc, index=test_data.index)  # Use the same index as test_data
lower_series = pd.Series(lower_limits_test.values, index=test_data.index)
upper_series = pd.Series(upper_limits_test.values, index=test_data.index)

#problem diagnosis
print("Forecast series (fc_series):")
print(fc_series)
print("Test Data:", test_data.head())

# Plot
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(train_data, label='Training Data')
plt.plot(test_data, color='blue', label='Actual Stock Price')
plt.plot(fc_series, color='orange', label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=0.10)
plt.title('ARCH CAPITAL GROUP Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('ARCH CAPITAL GROUP Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()
