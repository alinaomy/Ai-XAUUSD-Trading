import os
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def fetch_xauusd_data(start_date, end_date):
    """
    Fetch XAUUSD daily data using yfinance.
    Note: For hourly, premium APIs required. Using daily for training.
    """
    ticker = 'GC=F'  # Gold futures
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    if not data.empty:
        # Rename columns to standard OHLC
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data.reset_index(inplace=True)
        data.rename(columns={'Date': 'date'}, inplace=True)
        return data
    else:
        print("No data fetched")
        return None

if __name__ == "__main__":
    start_date = '2015-01-01'
    end_date = '2025-01-01'
    data = fetch_xauusd_data(start_date, end_date)
    if data is not None:
        data.to_csv('xauusd_data.csv', index=False)
        print("Data saved to xauusd_data.csv")
        print(f"Data shape: {data.shape}")
    else:
        print("Failed to fetch data")