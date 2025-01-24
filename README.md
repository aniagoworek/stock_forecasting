# **Nike Stock Forecasting**
## **Overview**
This project involves the analysis, modeling and forecasting of Nike stock prices using Python. It implements time series analysis techniques to predict stock trends and evaluate model performance. The process includes data preprocessing, visualization, stationarity testing, decomposition, ARIMA modeling, and forecast validation.

## **Key Features**
- **Data Preprocessing:**
  - Historical stock data (`NKE.csv`) is cleaned and transformed for analysis.
  - Log-transformation is applied to stabilize variance.
- **Exploratory Data Analysis (EDA):**
  - Visualizations of stock price distributions and trends.
  - Plots include histograms, scatterplots, and moving averages.
- **Stationarity Testing:**
  - Augmented Dickey-Fuller (ADF) test to assess stationarity.
  - Kruskal-Wallis test to check for seasonality.
- **Time Series Decomposition:**
  - Stock data is decomposed into trend, seasonal, and residual components.
- **Modeling:**
  - SARIMAX and Auto ARIMA models are implemented for forecasting.
  - Residual diagnostics and fitness evaluation of models.
- **Forecasting and Validation:**
  - Predictions compared against training and test data.
  - Model performance measured with error metrics like MSE, RMSE, MAE, and MAPE.

---

## **Project Structure**
- **Main Script (main.py):**
  - Loads and processes stock data.
  - Performs EDA, stationarity testing, and decomposition.
  - Trains and evaluates forecasting models.
## **Functions Script (functions.py):**
  Contains reusable functions for:
  - ADF testing
  - Plotting ACF and PACF
  - Kruskal-Wallis testing
  - Time series decomposition

---

## **Requirements**
  **Python Packages**
  Install the following Python libraries before running the scripts:
  - pip install numpy pandas matplotlib seaborn statsmodels pmdarima scikit-learn tqdm
  
---

## **How to Run**
**1. Prepare the Dataset:**
  - Place the NKE.csv file in the project directory.
  - Ensure the dataset includes columns like date, close, open, high, low, and volume.

**2. Run the Main Script:**
  - Execute main.py to perform the analysis and generate forecasts.

**3. Outputs:**
  Plots and model diagnostics are saved as PNG files in the project directory:
  - nike_stock_histogram.png
  - nike_scatterplot.png
  - time_series_decomposition.png
  - train and test data.png
  - prediction vs train data.png
  - forecast.png

---

## **Methodology**
  **1. Data Preprocessing**
  Log-transformations are applied for variance stabilization.
  Irrelevant columns are dropped, and null values are handled.
  
  **2. EDA and Visualization**
  Generates visualizations to understand the distribution and temporal patterns of stock prices.
  
  **3. Time Series Analysis**
  Checks for stationarity using ADF tests.
  Decomposes the series into additive components to analyze trends and seasonality.
  
  **4. Modeling**
  Auto ARIMA identifies optimal parameters for ARIMA modeling.
  SARIMAX (3,0,0) is used to fit and forecast the data.
  
  **5. Forecast Evaluation**
  Forecasts are compared with actual stock prices from the test set.
  Evaluation metrics (MSE, RMSE, MAE, MAPE) quantify performance.

---

## **Results**
- Residual Analysis: Verifies stationarity and non-correlation in model residuals.
- Forecast Accuracy: Model shows strong performance on both training and test sets.
- Error Metrics: The metrics indicate the very good performance of the model's predictions.
