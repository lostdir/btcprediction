from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from datetime import date, timedelta

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    # Get the forecast value from the form data
    days = int(request.form['days'])

    # Fetch BTC price data from Yahoo Finance
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")
    end_date = d1
    d2 = date.today() - timedelta(days=1200)
    d2 = d2.strftime("%Y-%m-%d")
    start_date = d2
    btc_data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)

    # Preprocess the data
    btc_data = btc_data[['Close']].dropna()

    # Experiment with different ARIMA model orders (p, d, q)
    orders = [(3, 1, 2), (5, 1, 0), (2, 1, 1), (3,1,3)]  # Example orders to try

    best_mae = float('inf')
    best_forecast = None
    best_order = None

    for order in orders:
        try:
            # Fit the ARIMA model
            model = ARIMA(btc_data, order=order)
            model_fit = model.fit()

            # Make the forecast
            forecast = model_fit.forecast(steps=days)

            # Calculate MAE
            actual_prices = btc_data['Close'].values[-days:]
            mae = mean_absolute_error(actual_prices, forecast)

            # Check if the current model order improves MAE
            if mae < best_mae:
                best_mae = mae
                best_forecast = forecast
                best_order = order

        except:
            # Skip the order if it fails to converge
            continue

    # Return the forecast result and accuracy as JSON
    return jsonify({'forecast': best_forecast.tolist(), 'mae': best_mae, 'order': best_order})

if __name__ == '__main__':
    app.run()
