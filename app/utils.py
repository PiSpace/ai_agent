import yfinance as yf
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(symbol, period):
    return yf.download(symbol, period=period)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_market_data():
    indices = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Dow Jones": "^DJI"
    }
    data = {}
    for name, symbol in indices.items():
        ticker = yf.Ticker(symbol)
        info = ticker.info
        data[name] = {
            "price": info.get('regularMarketPrice', 'N/A'),
            "change": info.get('regularMarketChangePercent', 0)
        }
    return data

def fetch_top_movers():
    # Use a list of major S&P 500 components instead of fetching all
    major_sp500 = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JNJ', 'JPM', 'V']
    
    data = yf.download(major_sp500, period="1d")['Close']
    returns = data.pct_change().iloc[-1].sort_values(ascending=False)
    
    top_gainers = returns.head(3)
    top_losers = returns.tail(3)
    
    return top_gainers, top_losers

def export_to_csv(data, filename):
    csv = data.to_csv().encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=filename,
        mime='text/csv'
    )

def export_to_pdf(data, filename):
    # Implement PDF export functionality
    pass
