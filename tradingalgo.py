import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from scipy.ndimage import gaussian_filter1d
import alpaca_trade_api as tradeapi
import logging
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from alpaca.data.timeframe import TimeFrame
import time

# Initialize logging
logging.basicConfig(filename='trade_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')


API_KEY = ''
API_SECRET = ''
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Constants
CURRENCY_PAIR = "ETH/USD" 
RISK_PERCENT = 0.01
MAX_DRAWDOWN = 0.1
RISK_REWARD_RATIO = 2
TRAIL_STOP_PERCENT = 0.02

# Global data cache and last fetched time
data_cache = None
last_fetched_time = None

def fetch_historical_data(pair, start, end):
    """
    Fetch historical cryptocurrency data from Alpaca API and update the data cache.
    """
    global data_cache, last_fetched_time

    try:
        # Convert pair format: remove '/' if necessary to match API expectations
        # Alpaca expects crypto symbols as "BTCUSD"
        pair_for_api = pair

        # Convert start and end to timezone-aware datetime objects (UTC)
        if isinstance(start, str):
            start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        if isinstance(end, str):
            end = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)

        # Determine the start time for fetching new data
        if data_cache is not None and not data_cache.empty:
            # Assuming the index of data_cache is datetime
            last_fetched_time = data_cache.index[-1] + timedelta(minutes=1)
            last_fetched_time = last_fetched_time.replace(tzinfo=timezone.utc)
        else:
            last_fetched_time = start

        # Ensure we do not fetch data beyond the specified end time
        last_fetched_time = min(last_fetched_time, end)

        # Determine the upper bound for fetching data: now or end (whichever is earlier)
        fetch_end = min(now, end)

        # Log the fetch attempt
        logging.info(f"Fetching data from {last_fetched_time.isoformat()} to {fetch_end.isoformat()} for {pair_for_api}")

        # Fetch new data
        bars = api.get_crypto_bars(
            pair_for_api, 
            TimeFrame.Minute, 
            start=last_fetched_time.isoformat(), 
            end=fetch_end.isoformat()
        ).df

        # Process and cache data
        if not bars.empty:
            new_data = bars[['close', 'high', 'low', 'volume']].rename(
                columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}
            )
            new_data.dropna(inplace=True)

            if data_cache is not None:
                data_cache = pd.concat([data_cache, new_data]).drop_duplicates()
            else:
                data_cache = new_data

        return data_cache

    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(data):
    data['returns'] = data['Close'].pct_change()
    data['EMA'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['short_ma'] = data['Close'].rolling(window=5).mean()
    data['long_ma'] = data['Close'].rolling(window=20).mean()
    data['MACD'] = data['EMA'] - data['Close'].ewm(span=26, adjust=False).mean()
    data['RSI'] = compute_rsi(data['Close'], 14)
    data['ATR'] = (data['High'] - data['Low']).rolling(window=14).mean()
    return data

def smooth_indicators(data, sigma=2):
    data['MACD'] = gaussian_filter1d(data['MACD'].fillna(0), sigma=sigma)
    data['RSI'] = gaussian_filter1d(data['RSI'].fillna(0), sigma=sigma)
    return data

def compute_macd_divergence(data):
    data['Price_Change'] = data['Close'].diff()
    data['MACD_Change'] = data['MACD'].diff()

    divergence_signal = None
    if data['Price_Change'].iloc[-1] < 0 and data['MACD_Change'].iloc[-1] > 0:
        divergence_signal = 'bullish'
    elif data['Price_Change'].iloc[-1] > 0 and data['MACD_Change'].iloc[-1] < 0:
        divergence_signal = 'bearish'
    return divergence_signal

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

def feature_selection(data, model):
    X = data.drop(columns=['returns']).fillna(0)  # Fill missing values
    y = (data['returns'] > 0).astype(int)  # Target: 1 for up, 0 for down

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model.fit(X_train, y_train)

    rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
    rfecv.fit(X_train, y_train)

    selected_features = X_train.columns[rfecv.support_]
    logging.info(f"Selected Features: {selected_features}")

    return selected_features

def generate_signal(data, selected_features):
    # Ensure necessary features are included
    if 'long_ma' not in selected_features:
        selected_features = list(selected_features) + ['long_ma']

    last_row = data[selected_features].iloc[-1]
    divergence_signal = compute_macd_divergence(data)

    prediction = model.predict([last_row])[0]

    # Adjust thresholds based on your strategy
    if last_row['MACD'] > 0 and last_row['RSI'] < 50 and prediction == 1:
        return 'buy'
    elif last_row['MACD'] < 0 and last_row['RSI'] > 50 and prediction == 0:
        return 'sell'
    return 'hold'

def calculate_position_size(account_balance, atr):
    risk_amount = account_balance * RISK_PERCENT
    return risk_amount / atr if atr > 0 else 0

def execute_trade(signal, pair, atr, data):
    try:
        account = api.get_account()
        cash_balance = float(account.cash)
        position_size = calculate_position_size(cash_balance, atr)
        last_row = data.iloc[-1]

        if position_size <= 0:
            logging.warning("Position size calculated as 0 or negative; skipping trade.")
            return

        if signal == 'buy':
            order = api.submit_order(
                symbol=pair.replace("/", ""),
                qty=position_size,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            logging.info(f"Executed Buy Order: {position_size} of {pair} at {last_row['Close']}")
        elif signal == 'sell':
            order = api.submit_order(
                symbol=pair.replace("/", ""),
                qty=position_size,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            logging.info(f"Executed Sell Order: {position_size} of {pair} at {last_row['Close']}")

    except Exception as e:
        logging.error(f"Error executing trade: {e}")

def monitor_drawdown(initial_balance):
    try:
        account = api.get_account()
        account_balance = float(account.equity)
        if account_balance <= initial_balance * (1 - MAX_DRAWDOWN):
            api.close_all_positions()
            raise Exception("Emergency stop triggered due to max drawdown.")
    except Exception as e:
        logging.error(f"Error monitoring drawdown: {e}")

def main():
    logging.info("Algorithm started: Monitoring trades...")

    # Define the time period for historical data
    today = datetime.now(timezone.utc)
    start_date = today - timedelta(days=365)
    start_date_str = start_date.isoformat()
    end_date_str = today.isoformat()

    try:
        initial_balance = float(api.get_account().cash)
    except Exception as e:
        logging.error(f"Error fetching account balance: {e}")
        return

    # Initial data fetch
    data = fetch_historical_data(CURRENCY_PAIR, start_date_str, end_date_str)
    if data is None or data.empty:
        logging.error("No data found during the initial fetch. Exiting.")
        return

    # Calculate indicators and smooth data
    data = calculate_indicators(data)
    data = smooth_indicators(data, sigma=2)

    # Feature selection
    selected_features = feature_selection(data, model)

    # Continuous monitoring loop
    while True:
        try:
            logging.info("Fetching latest data and checking for trade signals...")

            # Fetch new data and update our data cache
            new_data = fetch_historical_data(CURRENCY_PAIR, start_date_str, end_date_str)
            if new_data is None or new_data.empty:
                logging.warning("No new data available.")
                time.sleep(10)  # Short sleep to avoid spamming API calls
                continue

            # Update full dataset and recalc indicators
            data = pd.concat([data, new_data]).drop_duplicates()
            data = calculate_indicators(data)
            data = smooth_indicators(data, sigma=2)

            # Generate trading signal
            signal = generate_signal(data, selected_features)
            atr = data['ATR'].iloc[-1]
            logging.info(f"Signal generated: {signal} | Latest ATR: {atr}")

            # Execute trade if signal is 'buy' or 'sell'
            if signal != 'hold':
                execute_trade(signal, CURRENCY_PAIR, atr, data)

            # Monitor account drawdown
            monitor_drawdown(initial_balance)

            
            time.sleep(30)

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(10)

if __name__ == '__main__':
    main()
