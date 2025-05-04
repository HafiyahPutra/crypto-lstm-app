import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
from io import StringIO
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Important for server environment
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Load environment variables
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', 'CG-7pi9DCcf6E6PmCFBLrwvGtZT')

# Global variables
model = None
scaler = MinMaxScaler(feature_range=(0, 1))
look_back = 60

def initialize_model():
    """Initialize or load the LSTM model"""
    global model
    try:
        # Try to load pre-trained model if exists
        model = load_model('final_model.h5')
        print("Loaded pre-trained model from disk")
    except:
        # Build new model if no saved model found
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        print("Created new model")

def fetch_ohlc_data(coin_id='bitcoin', vs_currency='usd', days=7):
    """Fetch OHLC data from CoinGecko API"""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency={vs_currency}&days={days}"
    headers = {
        "Accepts": "application/json",
        "X-CG-Pro-API-Key": COINGECKO_API_KEY
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def train_model():
    """Train the LSTM model with fresh data"""
    global model, scaler
    
    df = fetch_ohlc_data()
    if df is None:
        return False, "Failed to fetch data"
    
    # Preprocess data
    dataset = df['close'].values.reshape(-1, 1)
    scaler.fit(dataset)
    dataset = scaler.transform(dataset)
    
    # Create training data
    X, y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[i + look_back, 0])
    
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Train model
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    model.save('model.h5')
    
    return True, "Model trained successfully"

def predict_future(days=7):
    """Predict future prices"""
    global model, scaler
    
    df = fetch_ohlc_data()
    if df is None:
        return None, None, "Failed to fetch data"
    
    # Prepare data for prediction
    dataset = df['close'].values.reshape(-1, 1)
    dataset = scaler.transform(dataset)
    
    predictions = []
    current_batch = dataset[-look_back:].reshape(1, look_back, 1)
    
    for i in range(days):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Create future dates
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['close'], label='Historical Data')
    plt.plot(future_dates, predictions, 'r-', label='Predictions')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return future_dates, predictions.flatten().tolist(), plot_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    days = request.args.get('days', default=7, type=int)
    dates, prices, plot = predict_future(days)
    
    if dates is None:
        return jsonify({"error": "Prediction failed"}), 500
    
    return jsonify({
        "dates": [d.strftime('%Y-%m-%d') for d in dates],
        "prices": prices,
        "plot": plot
    })

@app.route('/train', methods=['POST'])
def train():
    success, message = train_model()
    if success:
        return jsonify({"status": "success", "message": message})
    else:
        return jsonify({"status": "error", "message": message}), 500

if __name__ == '__main__':
    initialize_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
