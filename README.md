# NeuroStock: Stock Price Prediction

**NeuroStock** is a machine learning project that predicts stock market prices using a Long Short-Term Memory (LSTM) neural network model. The project provides a web-based interface using Streamlit to visualize stock price trends, predictions, and historical data.

## Features
- Fetch stock data using Yahoo Finance API
- Preprocess data for LSTM-based predictions
- Train a deep learning model (now enhanced with technical indicators and Bidirectional LSTM) to predict future stock prices
- Interactive Streamlit dashboard for stock data visualization and predictions
- Dynamic customization of EMA spans and downloadable prediction data

## Python Libraries Used
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

    streamlit run app.py

## Project Structure

    NeuroStock/
    ├── .venv/                   # Virtual environment  folder (optional)
    ├── app.py                   # Streamlit application for stock predictions
    ├── model.ipynb              # Jupyter notebook for model training
    ├── stock_model.h5           # Trained LSTM model
    ├── requirements.txt         # Project dependencies
    ├── README.md                # Project description
    └── .gitignore               # Files to exclude from version control


## Model Overview

The project uses an **LSTM (Long Short-Term Memory)** model to predict future stock prices based on historical data. The model is enhanced with technical indicators and uses a **Bidirectional LSTM** architecture for better performance. The model is trained using stock price data obtained from **Yahoo Finance**. It uses the following approach:

1. **Data Preprocessing**: 
   - Data is scaled using `MinMaxScaler` and split into training and testing datasets.
   - Added features: Moving Averages (MA20, MA50), Relative Strength Index (RSI), and Bollinger Bands.
   
2. **Model Training**:
   - A Bidirectional LSTM model is trained on the past 150 days of stock data.
   - Dropout layers are added to prevent overfitting.
   
3. **Model Evaluation**:
    - Metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are used to evaluate the model's accuracy.
    - The model's predictions are visualized alongside actual stock prices.

## Predictions & Visualizations

The Streamlit app provides:

- **Dynamic EMA Visualization:** Customize short-term and long-term EMA spans.
- **Detailed Performance Metrics:** Display Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Predicted Price Summary:** Maximum, minimum, and average predicted prices.
- **Downloadable Data:** Export predicted vs. actual stock prices as a CSV file.

## Contributing

If you'd like to contribute to this project, feel free to **fork** the repository and submit a **pull request**. Please ensure that your contributions align with the project's goal of providing accurate stock price predictions using **LSTM models**.

## License

This project is licensed under the **MIT License** - see the [MIT License](https://opensource.org/licenses/MIT) for details.
