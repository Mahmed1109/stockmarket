import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate():
    df = pd.read_csv("data/processed/final_features.csv", index_col=0)
    features = ['Lag1', 'Lag2', 'Lag3', 'MA5', 'MA10', 'Volatility', 'Sentiment']
    target = 'Close'

    X = df[features]
    y = df[target]

    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    model = joblib.load("models/price_forecast.pkl")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.legend()
    plt.title("Stock Price Prediction vs Actual")
    plt.xlabel("Date Index")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()
