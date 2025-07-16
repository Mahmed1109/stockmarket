import pandas as pd
import os

def add_features(df):
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=5).std()
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['Lag3'] = df['Close'].shift(3)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df = df.dropna()
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/final_features.csv")
    return df
