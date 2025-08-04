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
    df = df.sort_index()
    df = df[df.index >= (df.index.max() - pd.Timedelta(days=7))]  # Last 7 days only

    sentiment_scores = []
    dates = []

    cache = pd.DataFrame(columns=["Date", "Sentiment"])

    for date in df.index:
        if (cache["Date"] == date).any():
            sentiment = cache.loc[cache["Date"] == date, "Sentiment"].values[0]
        else:
            headlines = fetch_headlines(ticker, date)
            sentiment = compute_daily_sentiment(headlines)
            new_row = pd.DataFrame([{"Date": date, "Sentiment": sentiment}])
            cache = pd.concat([cache, new_row], ignore_index=True)

        sentiment_scores.append(sentiment)
        dates.append(date)

    sentiment_df = pd.DataFrame({"Date": dates, "Sentiment": sentiment_scores})

    df = df.reset_index()
    sentiment_df = sentiment_df.reset_index(drop=True)

    df = pd.merge(df, sentiment_df, on="Date", how="left")
    df = df.set_index("Date")

    return df
