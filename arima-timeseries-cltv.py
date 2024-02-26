import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from google.colab import drive

# Function to load data
def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Function to preprocess data
def preprocess_data(data):
    """Preprocess the data by dropping unnecessary columns and handling missing values."""
    if 'Country' in data.columns:
        data.dropna(subset=['Country'], inplace=True)
    data = data.drop(columns=[1, 2, 3, 7, 8, 9])
    return data

# Function to calculate CLV
def calculate_clv(data):
    """Calculate Customer Lifetime Value (CLV)."""
    data_cust = data[data['CancellationDate'].isnull()]
    data_cust = data_cust.drop(columns=['Loyalty ID', 'CancellationDate'])
    data_cust['EnrollmentDateOpening'] = pd.to_datetime(data_cust['EnrollmentDateOpening']).dt.tz_localize(None)
    clv = data_cust.groupby('EnrollmentDateOpening').mean()['Customer Lifetime Value'].reset_index()
    return clv

# Function to plot CLV
def plot_clv(clv):
    """Plot the Customer Lifetime Value (CLV)."""
    plt.figure(figsize=(10, 5))
    plt.plot(clv['EnrollmentDateOpening'], clv['Customer Lifetime Value'])
    plt.title('Customer Lifetime Value Over Time')
    plt.xlabel('Enrollment Date Opening')
    plt.ylabel('Customer Lifetime Value')
    plt.grid(True)
    plt.show()

# Function to perform HP filter
def hp_filter(ts, lamb):
    """Perform the Hodrick-Prescott (HP) filter."""
    cycle, trend = sm.tsa.filters.hpfilter(ts, lamb)
    return cycle, trend

# Function to plot decomposition
def plot_decomposition(cycle, trend):
    """Plot the decomposed time series."""
    fig, ax = plt.subplots(3, 1, figsize=(15, 15))
    ax[0].plot(ts, label='Original')
    ax[0].set_title('Customer Lifetime Value')
    ax[0].legend()
    
    ax[1].plot(trend, label='Trend', color='orange')
    ax[1].set_title('Trend Component')
    ax[1].legend()
    
    ax[2].plot(cycle, label='Cycle', color='green')
    ax[2].set_title('Cyclical Component')
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()

# Function to perform ADF test
def adf_test(series):
    """Perform the Augmented Dickey-Fuller (ADF) test."""
    dftest = adfuller(series, autolag='AIC')
    print("1. ADF Statistic: ", dftest[0])
    print("2. P-Value: ", dftest[1])
    print("3. Num Of Lags Used: ", dftest[2])
    print("4. Num Of Observations Used for ADF Regression and Critical Values Calculation: ", dftest[3])
    print("5. Critical Values:")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)

# Function to plot ACF and PACF
def plot_acf_pacf(series):
    """Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)."""
    x_diff = series.diff().dropna()
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot ACF
    lag_acf = acf(x_diff, nlags=20, fft=True)
    ax[0].plot(lag_acf)
    ax[0].axhline(y=0, linestyle='--', color='gray')
    ax[0].axhline(y=-1.96/np.sqrt(len(x_diff)), linestyle='--', color='gray')
    ax[0].axhline(y=1.96/np.sqrt(len(x_diff)), linestyle='--', color='gray')
    ax[0].set_title('Autocorrelation Function (q=1)')
    
    # Plot PACF
    lag_pacf = pacf(x_diff, nlags=20, method='ols')
    ax[1].plot(lag_pacf)
    ax[1].axhline(y=0, linestyle='--', color='gray')
    ax[1].axhline(y=-1.96/np.sqrt(len(x_diff)), linestyle='--', color='gray')
    ax[1].axhline(y=1.96/np.sqrt(len(x_diff)), linestyle='--', color='gray')
    ax[1].set_title('Partial Autocorrelation Function (p=1)')
    
    plt.tight_layout()
    plt.show()

# Function to split data into train and test sets
def split_data(series, train_size_ratio):
    """Split the time series data into training and testing sets."""
    total_size = len(series)
    train_size = int(total_size * train_size_ratio)
    train = series[:train_size]
    test = series[train_size:]
    return train, test

# Function to fit ARMA model
def fit_arma_model(train_series, order=(1, 1)):
    """Fit an ARMA model to the training data."""
    try:
        model = sm.tsa.ARIMA(train_series, order=order)
        model_fit = model.fit()
        print(model_fit.summary())
        return model_fit
    except Exception as e:
        print(f"Error fitting ARMA model: {e}")
        return None

# Function to plot residuals of the ARMA model
def plot_arma_residuals(model):
    """Plot the residuals of the fitted ARMA model."""
    if model:
        model.resid.plot(kind='kde')
        plt.title('Residual Distribution')
        plt.grid(True)
        plt.show()

# Function to predict using the ARMA model
def predict_with_arma(model, train_series, test_series):
    """Predict the test series using the fitted ARMA model."""
    if model:
        predictions = model.forecast(steps=len(test_series), start=1, end=len(train_series) + len(test_series))
        return predictions
    else:
        return None

# Function to evaluate the model
def evaluate_model(predictions, actual):
    """Evaluate the model using RMSE."""
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    print(f'RMSE: {rmse}')
    return rmse

# Function to plot train, test, and prediction
def plot_forecast(train_series, test_series, predicted_series, title='ARMA Forecast'):
    """Plot the training, testing, and predicted series."""
    plt.figure(figsize=(12, 8))
    plt.plot(train_series.index, train_series, label='Train', color='blue')
    plt.plot(test_series.index, test_series, label='Test', color='green')
    plt.plot(predicted_series.index, predicted_series, label='Predicted', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Customer Lifetime Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot weekly enrollment distribution
def plot_weekly_enrollment_distribution(data):
    """Plot the distribution of enrollments by week of year."""
    data['weekofyear'] = pd.to_datetime(data['EnrollmentDateOpening']).dt.week
    
    plt.figure(figsize=(8, 5))
    sns.histplot(data, x='weekofyear', hue='Education', kde=True)
    plt.title('Weekly Enrollment Distribution by Education Level')
    plt.xlabel('Week of Year')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Main script execution
if __name__ == "__main__":
    # Load data
    file_path = './Loyalty_card_customers.csv'
    data = load_data(file_path)
    
    if data is not None:
        # Preprocess data
        data1 = preprocess_data(data)
        
        # Calculate and plot CLV
        clv = calculate_clv(data1)
        print(clv.head())
        plot_clv(clv)
        
        # Perform HP filter
        cycle, trend = hp_filter(clv['Customer Lifetime Value'], lamb=12)
        
        # Plot decomposition
        plot_decomposition(cycle, trend)
        
        # ADF test
        adf_test(clv['Customer Lifetime Value'])
        
        # Plot ACF and PACF
        plot_acf_pacf(clv['Customer Lifetime Value'].astype(float))
        
        # Split data into train and test sets
        ts = clv.set_index('EnrollmentDateOpening')['Customer Lifetime Value']
        train, test = split_data(ts, train_size_ratio=0.7)
        
        # Fit ARMA model
        arma_model = fit_arma_model(train, order=(1, 1))
        
        # Plot residuals of the ARMA model
        if arma_model:
            plot_arma_residuals(arma_model)
        
        # Predict using the ARMA model
        predictions = predict_with_arma(arma_model, train, test)
        
        # Evaluate the model
        evaluate_model(predictions, test)
        
        # Plot forecast
        plot_forecast(train, test, predictions)
        
        # Plot weekly enrollment distribution
        enroll_data = data1.copy()
        plot_weekly_enrollment_distribution(enroll_data)
