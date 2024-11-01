from datetime import datetime
import pandas as pd
import requests
import datetime as dt


def make_api_call(base_url, endpoint="", method="GET", **kwargs):
    # Construct the full URL
    full_url = f'{base_url}{endpoint}'

    # Make the API call
    response = requests.request(method=method, url=full_url, **kwargs)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        return response
    else:
        # If the request was not successful, raise an exception with the error message
        raise Exception(f'API request failed with status code {response.status_code}: {response.text}')


def get_binance_historical_data(symbol, interval, start_date, end_date):
    base_url = 'https://fapi.binance.com'
    endpoint = '/fapi/v1/klines'
    method = 'GET'

    candles_data = []

    while start_date < end_date:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 1500,
            'startTime': start_date,
            'endTime': end_date
        }

        response = make_api_call(base_url, endpoint=endpoint, method=method, params=params)
        batch_data = response.json()

        if not batch_data:
            break  # Exit the loop if no data is returned

        candles_data.extend(batch_data)
        last_candle = batch_data[-1]
        start_date = last_candle[0] + 1  # Update start_date to the next candle's open time

    # Convert to DataFrame
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    dtype = {
        'open_time': 'datetime64[ms, Asia/Jerusalem]',
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'float64',
        'close_time': 'datetime64[ms, Asia/Jerusalem]',
        'quote_asset_volume': 'float64',
        'number_of_trades': 'int64',
        'taker_buy_base_asset_volume': 'float64',
        'taker_buy_quote_asset_volume': 'float64',
        'ignore': 'float64'
    }

    df = pd.DataFrame(candles_data, columns=columns)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True).dt.tz_convert('Asia/Jerusalem')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True).dt.tz_convert('Asia/Jerusalem')
    df = df.astype(dtype)

    return df


def download_data(symbol, interval):
    symbol = symbol
    interval = interval
    start_date = int(dt.datetime(year=2022 , month=1, day=1).timestamp() * 1000)
    end_date = int(dt.datetime(year=2024, month=7 , day=7).timestamp() * 1000)
    btcusdt_df = get_binance_historical_data(symbol, interval, start_date, end_date)
    btcusdt_df.to_csv(f'{symbol}_data.csv')


def get_exchange_info():
    base_url = 'https://fapi.binance.com'
    endpoint = '/fapi/v1/exchangeInfo'
    response = make_api_call(base_url, endpoint)
    return response.json()

download_data('BTCUSDT', '1d')