<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroStock: Stock Price Prediction</title>
</head>
<body>

    <h1>NeuroStock: Stock Price Prediction</h1>

    <p><strong>NeuroStock</strong> is a machine learning project that predicts stock market prices using a Long Short-Term Memory (LSTM) neural network model. The project provides a web-based interface using Streamlit to visualize stock price trends, predictions, and historical data.</p>

    <h2>Features</h2>
    <ul>
        <li>Fetch stock data using Yahoo Finance API</li>
        <li>Preprocess data for LSTM-based predictions</li>
        <li>Train a deep learning model to predict future stock prices</li>
        <li>Interactive Streamlit dashboard for stock data visualization and predictions</li>
    </ul>

    <h2>Technologies Used</h2>
    <ul>
        <li>Python</li>
        <li>TensorFlow & Keras</li>
        <li>Streamlit</li>
        <li>NumPy</li>
        <li>Pandas</li>
        <li>Matplotlib</li>
        <li>Scikit-learn</li>
        <li>Yahoo Finance (yfinance)</li>
    </ul>

    <h2>Installation</h2>
    <p>Follow these steps to set up the project locally:</p>
    <ol>
        <li>Clone this repository:</li>
        <pre><code>git clone https://github.com/JALAJ-PATEL/NeuroStock.git</code></pre>

        <li>Navigate to the project directory:</li>
        <pre><code>cd NeuroStock</code></pre>

        <li>Create a virtual environment (optional but recommended):</li>
        <pre><code>python -m venv .venv</code></pre>

        <li>Activate the virtual environment:</li>
        <pre><code>.\.venv\Scripts\activate</code></pre>

        <li>Install the required dependencies:</li>
        <pre><code>pip install -r requirements.txt</code></pre>
    </ol>

    <h2>Running the App</h2>
    <p>To run the Streamlit app locally and view the stock price predictions:</p>
    <pre><code>streamlit run app.py</code></pre>

    <h2>Project Structure</h2>
    <pre><code>
    NeuroStock/
    ├── .venv/                   # Virtual environment folder (optional)
    ├── app.py                   # Streamlit application for stock predictions
    ├── stock_price_predict.ipynb # Jupyter notebook for model training
    ├── stock_dl_model.h5        # Trained LSTM model
    ├── requirements.txt         # Project dependencies
    ├── README.md                # Project description
    └── .gitignore               # Files to exclude from version control
    </code></pre>

    <h2>Model Overview</h2>
    <p>The project uses an LSTM (Long Short-Term Memory) model to predict future stock prices based on historical data. The model is trained using stock price data obtained from Yahoo Finance. It uses the following approach:</p>
    <ol>
        <li>Data Preprocessing: Data is scaled using MinMaxScaler and split into training and testing datasets.</li>
        <li>Model Training: An LSTM model is trained on the past 100 days of stock data to predict future prices.</li>
        <li>Model Evaluation: The model's predictions are evaluated against actual stock prices, and accuracy is visualized.</li>
    </ol>

    <h2>Predictions & Visualizations</h2>
    <p>The Streamlit app allows users to:</p>
    <ul>
        <li>Enter a stock ticker (e.g., AAPL, TSLA, etc.)</li>
        <li>View historical stock data</li>
        <li>Visualize stock prices with Exponential Moving Averages (EMA)</li>
        <li>Compare predicted stock prices with actual prices</li>
    </ul>

    <h2>Contributing</h2>
    <p>If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. Please ensure that your contributions align with the project's goal of providing accurate stock price predictions using LSTM models.</p>

    <h2>License</h2>
    <p>This project is licensed under the MIT License - see the <a href="https://opensource.org/licenses/MIT">MIT License</a> for details.</p>

    <h2>Acknowledgements</h2>
    <ul>
        <li>Special thanks to the contributors of the <a href="https://www.tensorflow.org/">TensorFlow</a> and <a href="https://streamlit.io/">Streamlit</a> communities.</li>
        <li>Data source: <a href="https://finance.yahoo.com/">Yahoo Finance</a></li>
    </ul>

</body>
</html>
