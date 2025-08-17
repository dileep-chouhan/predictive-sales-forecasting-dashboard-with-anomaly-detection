import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
dates = pd.date_range(start='2022-01-01', periods=365)
sales = 100 + 50 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 20, 365) # Seasonal trend with noise
sales[200:210] += 100 # Simulate a temporary sales spike (anomaly)
df = pd.DataFrame({'Date': dates, 'Sales': sales})
# --- 2. Data Cleaning & Preparation ---
# In a real-world scenario, this section would involve handling missing values, outliers, etc.
# For this synthetic data, no cleaning is strictly necessary.
# --- 3. Sales Forecasting using Exponential Smoothing ---
train = df[:-30] # Use the last 30 days for testing
test = df[-30:]
model = ExponentialSmoothing(train['Sales'], trend='add', seasonal='add', seasonal_periods=7) # Assuming weekly seasonality
fitted = model.fit()
forecast = fitted.forecast(30)
# --- 4. Anomaly Detection ---
# Simple anomaly detection: Identify points significantly deviating from the forecast.
residuals = test['Sales'] - forecast
threshold = 2 * np.std(residuals) # Adjust threshold as needed
anomalies = test[np.abs(residuals) > threshold]
# --- 5. Visualization ---
# Plot Sales Data with Forecast and Anomalies
plt.figure(figsize=(12, 6))
plt.plot(train['Date'], train['Sales'], label='Training Data')
plt.plot(test['Date'], test['Sales'], label='Actual Sales')
plt.plot(test['Date'], forecast, label='Forecast')
plt.scatter(anomalies['Date'], anomalies['Sales'], color='red', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Forecast with Anomaly Detection')
plt.legend()
plt.grid(True)
plt.tight_layout()
output_filename = 'sales_forecast_anomalies.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Plot Residuals
plt.figure(figsize=(10,6))
plt.plot(residuals)
plt.title('Residuals Plot')
plt.xlabel('Day')
plt.ylabel('Residuals')
plt.grid(True)
plt.tight_layout()
output_filename2 = 'residuals_plot.png'
plt.savefig(output_filename2)
print(f"Plot saved to {output_filename2}")
# --- 6. Additional Analysis (Optional) ---
#  More sophisticated anomaly detection techniques (e.g., Isolation Forest, One-Class SVM) could be used.
#  Statistical tests could be performed to assess the significance of the forecast accuracy.