from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import timedelta
import pandas as pd
import numpy as np
import os

newsapi = NewsApiClient(api_key='YOUR_NEWSAPI_KEY_HERE')  # <- Replace this
analyser = SentimentIntensityAnalyzer()

def get_sentiment_score(date, ticker):
    date_str = date.strftime('%Y-%m-%d')
    next_day = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        articles = newsapi.get_everything(
            q=ticker,
            from_param=date_str,
            to=next_day,
            language='en',
            sort_by='relevancy',
            page_size=20
        )
        scores = [analyser.polarity_scores(a['title'])['compound'] for a in articles['articles']]
        return np.mean(scores) if scores else 0
    except:
        return 0

def attach_sentiment(df, ticker):
    print("Fetching sentiment scores...")
    df['Sentiment'] = df.index.to_series().apply(lambda d: get_sentiment_score(d, ticker))
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(f"data/processed/{ticker}_with_sentiment.csv")
    return df
