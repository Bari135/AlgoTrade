import requests
import pandas as pd
from datetime import datetime, timedelta

# Set your API key
api_key = "KSQT3IAWYNFFTLJC"
ticker = "CRYPTO:ETH"
start_date = "20240102T1045"
end_date = "20240707T0700"
sort = "EARLIEST"
limit = 1000

# Function to get news for a specific date range
def get_news(ticker, sort, api_key, limit, date_from):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&time_from={date_from}&limit={limit}&apikey={api_key}&sort={sort}&tickers={ticker}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: ", response.status_code)
        return None

curr_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Initialize DataFrame for the filtered ticker data
filtered_ticker_data = []

while curr_date <= end_date:
    data = get_news(ticker, sort, api_key, limit, curr_date.strftime('%Y%m%dT%H%M'))
    print(f"Fetching data for {curr_date.strftime('%Y%m%dT%H%M')}")
    if data and 'feed' in data:
        # Normalize the 'feed' data into a DataFrame
        df = pd.json_normalize(data['feed'])
        
        # Convert 'time_published' column to datetime format
        df['time_published'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S')

        # Extract and filter ticker sentiment data
        for article in data['feed']:
            for ticker_data in article['ticker_sentiment']:
                if ticker_data['ticker'] == ticker:
                    filtered_ticker_data.append({
                        'title': article['title'],
                        'url': article['url'],
                        'time_published': article['time_published'],
                        'authors': article['authors'],
                        'summary': article['summary'],
                        'banner_image': article['banner_image'],
                        'source': article['source'],
                        'category_within_source': article['category_within_source'],
                        'source_domain': article['source_domain'],
                        'overall_sentiment_score': article['overall_sentiment_score'],
                        'overall_sentiment_label': article['overall_sentiment_label'],
                        'ticker': ticker_data['ticker'],
                        'relevance_score': ticker_data['relevance_score'],
                        'ticker_sentiment_score': ticker_data['ticker_sentiment_score'],
                        'ticker_sentiment_label': ticker_data['ticker_sentiment_label']
                    })
        
        # Update curr_date to the latest time_published in the current data set
        if not df.empty:
            curr_date = df['time_published'].max() + timedelta(minutes=1)  # Increment by one second to avoid overlap
            print(f"Updated curr_date to {curr_date}")
        else:
            break
    else:
        break

# Convert the filtered ticker data into a DataFrame
df_filtered_ticker = pd.DataFrame(filtered_ticker_data)

# Convert 'time_published' in the filtered ticker DataFrame to datetime format
df_filtered_ticker['time_published'] = pd.to_datetime(df_filtered_ticker['time_published'], format='%Y%m%dT%H%M%S')

df_filtered_ticker.to_csv(f'btcsent.csv', index=False)

