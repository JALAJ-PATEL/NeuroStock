# NeuroStock: Stock Price Prediction

**NeuroStock** is a machine learning project that predicts stock market prices using a Long Short-Term Memory (LSTM) neural network model. The project provides a web-based interface using Streamlit to visualize stock price trends, predictions, and historical data.

## Features
- Fetch stock data using Yahoo Finance API
- Preprocess data for LSTM-based predictions
- Train a deep learning model to predict future stock prices
- Interactive Streamlit dashboard for stock data visualization and predictions

## Technologies Used
- Python
- TensorFlow & Keras
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Yahoo Finance (yfinance)

## Installation
Follow these steps to set up the project locally:

1. Clone this repository:
   ```bash
   git clone https://github.com/JALAJ-PATEL/NeuroStock.git

2. Navigate to the project directory:
    ```bash
    cd NeuroStock

3. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv

4. Activate the virtual environment:

    - On Windows:
        ```bash
        .\.venv\Scripts\activate

    - On macOS/Linux:
        ```bash
        source .venv/bin/activate
    
5. Install the required dependencies:

    ```bash
    pip install -r requirements.txt


## Running the App
To run the Streamlit app locally and view the stock price predictions:
    
    ```bash
    streamlit run app.py

## Project Structure

    NeuroStock/
    ├── .venv/                   # Virtual environment folder (optional)
    ├── app.py                   # Streamlit application for stock predictions
    ├── stock_price_predict.ipynb # Jupyter notebook for model training
    ├── stock_dl_model.h5        # Trained LSTM model
    ├── requirements.txt         # Project dependencies
    ├── README.md                # Project description
    └── .gitignore               # Files to exclude from version control

## Model Overview

The project uses an **LSTM (Long Short-Term Memory)** model to predict future stock prices based on historical data. The model is trained using stock price data obtained from **Yahoo Finance**. It uses the following approach:

1. **Data Preprocessing**: 
   - Data is scaled using `MinMaxScaler` and split into training and testing datasets.
   
2. **Model Training**:
   - An LSTM model is trained on the past 100 days of stock data to predict future prices.
   
3. **Model Evaluation**:
   - The model's predictions are evaluated against actual stock prices, and accuracy is visualized.

## Predictions & Visualizations

The Streamlit app allows users to:

- Enter a stock ticker (e.g., **AAPL**, **TSLA**, etc.)
- View historical stock data
- Visualize stock prices with **Exponential Moving Averages (EMA)**
- Compare predicted stock prices with actual prices

## Contributing

If you'd like to contribute to this project, feel free to **fork** the repository and submit a **pull request**. Please ensure that your contributions align with the project's goal of providing accurate stock price predictions using **LSTM models**.

## License

This project is licensed under the **MIT License** - see the [MIT License](https://opensource.org/licenses/MIT) for details.
