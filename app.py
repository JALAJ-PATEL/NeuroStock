import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import datetime as dt

# Streamlit Application Title
st.title('Stock Market Price Prediction')

# User Input for Stock Ticker
stock = st.text_input('Enter Stock Ticker (e.g., AAPL, TSLA, etc.)', 'AAPL')

# Define date range
start = st.date_input('Select start date', dt.date(2011, 1, 1))
end = st.date_input('Select end date', dt.date(2023, 3, 1))

# Fetch data from Yahoo Finance
try:
    df = yf.download(stock, start, end)
    st.subheader('Data Summary')
    st.write(df.describe())
except Exception as e:
    st.error("Error fetching stock data. Please check the ticker symbol and internet connection.")
    st.stop()

# Visualization: EMA Charts
ema20 = df['Close'].ewm(span=20, adjust=False).mean()
ema50 = df['Close'].ewm(span=50, adjust=False).mean()

st.subheader('Closing Price with 20 & 50 EMA')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], 'y', label='Closing Price')
plt.plot(ema20, 'g', label='EMA 20 Days')
plt.plot(ema50, 'r', label='EMA 50 Days')
plt.legend()
st.pyplot(fig)

# Check for Model File
model_path = 'stock_dl_model.h5'
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
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)  # FIXED: Use pd.concat
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

# Visualization: Predicted vs Original
st.subheader('Predicted vs Original Price')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.legend()
st.pyplot(fig2)
