import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import os

def train_model():
    df = pd.read_csv("data/processed/final_features.csv", index_col=0)
    features = ['Lag1', 'Lag2', 'Lag3', 'MA5', 'MA10', 'Volatility', 'Sentiment']
    target = 'Close'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/price_forecast.pkl")
    return model, X_test, y_test
