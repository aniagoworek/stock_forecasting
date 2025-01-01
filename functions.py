import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import kruskal
from tqdm import tqdm


def adfuller_test(series, verbose=True):
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    """
    result = adfuller(series.dropna())
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', '#Observation Used']
    if verbose:
        for value, label in zip(result, labels):
            print(f"{label}: {value}")
        if result[1] <= 0.05:
            print("Strong evidence against null hypothesis (stationary series).")
        else:
            print("Weak evidence against null hypothesis (non-stationary series).")
    return result


def optimize_arima(endog, order_list):
    """
    Return DataFrame with ARIMA parameters and corresponding AIC.
    """
    results = []
    for order in tqdm(order_list, desc="Optimizing ARIMA parameters"):
        try:
            model = SARIMAX(endog, order=order, simple_differencing=False).fit(disp=False)
            results.append([order, model.aic])
        except Exception:
            continue
    result_df = pd.DataFrame(results, columns=['(p, d, q)', 'AIC'])
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_df


def plot_acf_pacf(series, filename_acf, filename_pacf):
    """
    Plot and save ACF and PACF plots.
    """
    plot_acf(series.dropna(), lags=80)
    plt.savefig(filename_acf)
    plt.close()
    plot_pacf(series.dropna(), lags=80)
    plt.savefig(filename_pacf)
    plt.close()


def kruskal_test(series, index):
    """
    Perform Kruskal-Wallis test for seasonality.
    """
    months = index.month
    seasonality_test = kruskal(*[series[months == m] for m in range(1, 13)])
    print("Kruskal-Wallis Statistic:", seasonality_test.statistic)
    print("p-value:", seasonality_test.pvalue)
    if seasonality_test.pvalue < 0.05:
        print("The series shows significant seasonality.")
    else:
        print("The series does not show significant seasonality.")
    return seasonality_test


def decompose_series(series, period, filename):
    """
    Perform time series decomposition and save the plot.
    """
    decomposition = seasonal_decompose(series, model='additive', period=period)
    plt.figure(figsize=(10, 8))
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
    plt.savefig(filename)
    plt.close()
    return decomposition