import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

DOWN_PERCENT = -2
UP_PERCENT = -DOWN_PERCENT
SENTIMENT_NUM = 0.4

def process_data(file_name, currency):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)

    # Convert 'open_time' column to datetime format
    df['open_time'] = pd.to_datetime(df['open_time'])

    # Plotting the 'open', 'close', 'high', and 'low' columns
    fig, ax = plt.subplots()
    ax.plot(df['open_time'], df['open'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title(f'{currency} Price')

    # Format the x-axis tick labels
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Rotate the tick labels for better readability
    plt.xticks(rotation=45)

    plt.show()

    # Printing the describe() function output for each column
    print("Open Column:")
    print(df['open'].describe())
    print()

    print("Close Column:")
    print(df['close'].describe())
    print()

    print("High Column:")
    print(df['high'].describe())
    print()

    print("Low Column:")
    print(df['low'].describe())
    print()


def prepare_prices_data(prices_df: pd.DataFrame):
    prices_df['open_time'] = pd.to_datetime(prices_df['open_time'], utc=True)
    prices_df['open_time']=prices_df['open_time'].dt.date
    prices_df.sort_values('open_time', inplace=True)

    prices_df = prices_df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    prices_df = prices_df.copy()
    prices_df['next_open'] = prices_df['open'].shift(-2)

    return prices_df

def prepare_sentiment_data(sentiment_df: pd.DataFrame):
    sentiment_df['time_published'] = pd.to_datetime(sentiment_df['time_published']).dt.date
    sentiment_df = sentiment_df[sentiment_df['time_published'] >= datetime.datetime(2022, 1, 1).date()].copy()
    sentiment_df['mean_sentiment'] = sentiment_df.groupby('time_published')['ticker_sentiment_score'].transform('mean')
    # Calculate the number of bullish, neutral, and bearish sentiments for each date
    bullish_counts = sentiment_df[sentiment_df['ticker_sentiment_score'] > SENTIMENT_NUM].groupby('time_published')['ticker_sentiment_score'].count()
    neutral_counts = sentiment_df[(sentiment_df['ticker_sentiment_score'] >= -SENTIMENT_NUM) & (sentiment_df['ticker_sentiment_score'] <= SENTIMENT_NUM)].groupby('time_published')['ticker_sentiment_score'].count()
    bearish_counts = sentiment_df[sentiment_df['ticker_sentiment_score'] < -SENTIMENT_NUM].groupby('time_published')['ticker_sentiment_score'].count()
    
    # Merge the counts back into the original DataFrame
    sentiment_df = sentiment_df.drop_duplicates(subset='time_published')
    sentiment_df = sentiment_df[['time_published', 'mean_sentiment']]
    
    sentiment_df = sentiment_df.merge(bullish_counts.rename('bullish_num'), on='time_published', how='left')
    sentiment_df = sentiment_df.merge(neutral_counts.rename('neutral_num'), on='time_published', how='left')
    sentiment_df = sentiment_df.merge(bearish_counts.rename('bearish_num'), on='time_published', how='left')
    sentiment_df.fillna(0, inplace=True)
    return sentiment_df

def merge_prices_and_sentiment_data(prices_df: pd.DataFrame, sentiment_df: pd.DataFrame):
    merged_df = pd.merge(prices_df, sentiment_df, left_on='open_time', right_on='time_published', how='left')
    merged_df.drop(columns='time_published', inplace=True)
    merged_df['mean_sentiment'] = merged_df['mean_sentiment'].interpolate(method='nearest')
    merged_df.fillna({'bearish_num':0,
                      'neutral_num':0,
                    'bullish_num':0 }, inplace=True)
    
    return merged_df

def create_indicatos(df: pd.DataFrame):
    df['price_change'] = ((df['next_open'] - df['close']) / df['close']) * 100
    df['sentiment_change'] = df['mean_sentiment'].pct_change()
    df['sentiment_change'] = df['sentiment_change'].replace([-np.inf, np.inf], 0)
    df['volume_change'] = df['volume'].pct_change()
    
    df['5_day_ma'] = df['close'].expanding(min_periods=1).mean()
    df.loc[4:, '5_day_ma'] = df['close'].rolling(window=5).mean().iloc[4:]

    df['50_day_ma'] = df['close'].expanding(min_periods=1).mean()
    df.loc[49:, '50_day_ma'] = df['close'].rolling(window=50).mean().iloc[49:]
    return df

def create_signals(df: pd.DataFrame):
    df['real_signal'] = 1
    
    df.loc[df['price_change'] > UP_PERCENT, 'real_signal'] = 2
    df.loc[df['price_change'] < DOWN_PERCENT, 'real_signal'] = 0
    return df 



def prepare_data(ticker):
    df_prices = pd.read_csv(f'data/{ticker}_PRICES.csv')
    df_sentiment = pd.read_csv(f'data/{ticker}_SENT.csv')
    df_prices = prepare_prices_data(df_prices)
    df_sentiment = prepare_sentiment_data(df_sentiment)
    df = merge_prices_and_sentiment_data(df_prices, df_sentiment)
    df = create_indicatos(df)
    df = create_signals(df)
    df['day_of_week'] = pd.to_datetime(df['open_time']).dt.dayofweek + 1

    df.dropna(inplace=True)
    return df
    
