import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import datetime as dt

# Streamlit Application Title
st.title('NeuroStock: Stock Price Prediction App')

# User Input for Stock Ticker
stock = st.text_input('Enter Stock Ticker (e.g., AAPL, TSLA, etc.)', 'AAPL')

# Define date range
start = st.date_input('Select start date', dt.date(2014, 1, 1))
end = st.date_input('Select end date', dt.date(2024, 1, 1))

# Fetch data from Yahoo Finance
try:
    df = yf.download(stock, start, end)
    st.subheader('Data Summary')
    st.write(df.describe())
except Exception as e:
    st.error("Error fetching stock data. Please check the ticker symbol and internet connection.")
    st.stop()

# Visualization: Dynamic EMA Charts
st.sidebar.header("Customize Moving Averages")
ema_short = st.sidebar.slider("Short-Term EMA Span", min_value=5, max_value=50, value=20, step=1)
ema_long = st.sidebar.slider("Long-Term EMA Span", min_value=50, max_value=200, value=50, step=1)

ema_short_data = df['Close'].ewm(span=ema_short, adjust=False).mean()
ema_long_data = df['Close'].ewm(span=ema_long, adjust=False).mean()

st.subheader(f'Closing Price with {ema_short}-Day & {ema_long}-Day EMA')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], 'y', label='Closing Price')
plt.plot(ema_short_data, 'g', label=f'EMA {ema_short} Days')
plt.plot(ema_long_data, 'r', label=f'EMA {ema_long} Days')
plt.legend()
st.pyplot(fig)

# Check for Model File
model_path = 'stock_model.h5'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please upload the trained model.")
    st.stop()

# Load the Pre-trained Model
model = load_model(model_path)

# Data Preprocessing for Prediction
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_scaled = scaler.fit_transform(data_training)

# Prepare Test Data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make Predictions
y_predicted = model.predict(x_test)

# Reverse Scaling
scale_factor = 1 / scaler.scale_[0]
y_test = y_test * scale_factor
y_predicted = y_predicted * scale_factor

# Metrics
mae = mean_absolute_error(y_test, y_predicted)
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
st.subheader("Model Performance")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Root Mean Square Error (RMSE): {rmse:.2f}")

# Visualization: Predicted vs Original
st.subheader('Predicted vs Original Price')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.legend()
st.pyplot(fig2)

# Highlight Trends
st.subheader("Predicted Price Summary")
st.write(f"Maximum Predicted Price: {np.max(y_predicted):.2f}")
st.write(f"Minimum Predicted Price: {np.min(y_predicted):.2f}")
st.write(f"Average Predicted Price: {np.mean(y_predicted):.2f}")

# Download Data
st.sidebar.subheader("Download Data")
predicted_vs_actual = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_predicted.flatten()
})
csv_data = predicted_vs_actual.to_csv(index=False)
st.sidebar.download_button(
    label="Download Predictions as CSV",
    data=csv_data,
    file_name=f"{stock}_predictions.csv",
    mime='text/csv'
)
