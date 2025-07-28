import pandas as pd
from datetime import datetime, timedelta
from src.fetch_stock import get_stock_data
from src.fetch_sentiment import attach_sentiment
from src.feature_engineer import add_features
from src.train_model import train_model
from src.evaluate_model import evaluate


def main():
    ticker = "AAPL"

    # newsapi only allows 30days, gotta pay for more
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=30)

    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    df = get_stock_data(ticker, start_date, end_date)
    df.index = pd.to_datetime(df.index)

    df = attach_sentiment(df, ticker)

    df = add_features(df)

    train_model()

    evaluate()

if __name__ == "__main__":
    main()
