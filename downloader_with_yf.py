#!/usr/bin/env python3

import os
import pandas as pd
import yfinance as yf

from curl_cffi import requests


def download(tickers, save_path, start_date="2006-01-01", end_date="2025-01-01"):
    if len(tickers) == 0:
        tickers = ["SPY", "EFA", "IEF", "VNQ", "DBC", "SHV"]

    os.makedirs(save_path, exist_ok=True)

    for symbol in tickers:
        try:
            session = requests.Session(impersonate="chrome")
            ticker = yf.Ticker(symbol, session=session)
            data = ticker.history(start=start_date, end=end_date)

            column_mapping = {
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Adj Close": "Adj Close",
                "Volume": "Volume",
            }

            new_data = pd.DataFrame()

            for new_col, old_col in column_mapping.items():
                if old_col in data.columns:
                    new_data[new_col] = data[old_col]

            if not new_data.empty:
                new_data.to_csv(f"{save_path}/{symbol}_data.csv")

        except Exception as e:
            pass


if __name__ == "__main__":
    download(tickers=["SPY", "EFA", "IEF", "VNQ", "DBC", "SHV"], save_path="data")
