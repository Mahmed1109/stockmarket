import os
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
analyzer = SentimentIntensityAnalyzer()

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
    scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    return np.mean(scores)

def attach_sentiment(df, ticker):
    sentiment_scores = []
    for date in df.index:
        headlines = fetch_headlines(ticker, date)
        sentiment = compute_daily_sentiment(headlines)
        sentiment_scores.append(sentiment)
    df['sentiment'] = sentiment_scores
    return df
