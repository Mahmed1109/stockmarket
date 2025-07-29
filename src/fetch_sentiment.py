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
analyzer = SentimentIntensityAnalyzer()
CACHE_FILE = "sentiment_cache.csv"