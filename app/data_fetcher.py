import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker, start_date, end_date, ma_options=None):
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if ma_options:
        for ma in ma_options:
            days = int(ma.split('-')[0])
            data[f'MA_{days}'] = data['Close'].rolling(window=days).mean()
    
    return data
