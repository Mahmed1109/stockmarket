import yfinance as yf
import pandas as pd
import os

def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close', 'Volume']]
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(f"data/raw/{ticker}_stock.csv")
    return df
