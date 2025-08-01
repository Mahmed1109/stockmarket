import os
import time
import pandas as pd
import numpy as np
from datetime import timedelta
from dotenv import load_dotenv
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()
api_key = os.getenv("NewsAPI_KEY")

if not api_key:
    raise ValueError("Missing NewsAPI_KEY in .env file or environment")

newsapi = NewsApiClient(api_key=api_key)
analyser = SentimentIntensityAnalyzer()
CACHE_FILE = "sentiment_cache.csv"

def fetch_headlines(ticker, date):
    query = f"{ticker} stock"
    from_date = date.strftime("%Y-%m-%d")
    to_date = (date + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        articles = newsapi.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='relevancy',
            page_size=50
        )
        return [article['title'] for article in articles['articles']]
    except Exception as e:
        print(f"Error fetching headlines for {date}: {e}")
        return []
    
def compute_daily_sentiment(headlines):
    if not headlines:
        return 0
    scores = [analyser.polarity_scores(h)['compound'] for h in headlines]
    return np.mean(scores)

def load_cache():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE, parse_dates=["date"])
    return pd.DataFrame(columns=["date", "ticker", "sentiment"])

def save_cache(cache):
    cache.to_csv(CACHE_FILE, index=False)

def attach_sentiment(df, ticker):
    cache = load_cache()
    sentiment_scores = []

    df = df.sort_index()
    recent_df = df.last("7D")

    for date in recent_df.index:
        cached = cache[(cache["date"] == date) & (cache["ticker"] == ticker)]

        if not cached.empty:
            sentiment = cached.iloc[0]["sentiment"]
        else:
            headlines = fetch_headlines(ticker, date)
            sentiment = compute_daily_sentiment(headlines)
            new_row = pd.DataFrame({"date": [date], "ticker": [ticker], "sentiment": [sentiment]})
            cache = pd.concat([cache, new_row], ignore_index=True)
            time.sleep(1.1)

        sentiment_scores.append((date, sentiment))

    save_cache(cache)

    sentiment_df = pd.DataFrame(sentiment_scores, columns=["date", "Sentiment"]).set_index("date")
    df = df.join(sentiment_df, how="left")
    return df