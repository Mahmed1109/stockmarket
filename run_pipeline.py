import pandas as pd
from src.fetch_stock import get_stock_data
from src.fetch_sentiment import attach_sentiment
from src.feature_engineering import add_features
from src.train_model import train_model
from src.evaluate_model import evaluate

def main():
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2025-01-01"

    
    df = get_stock_data(ticker, start_date, end_date)
    df.index = pd.to_datetime(df.index)

    
    df = attach_sentiment(df, ticker)

    df = add_features(df)

    train_model()

    evaluate()

if __name__ == "__main__":
    main()
