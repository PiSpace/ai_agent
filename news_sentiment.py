# Sentiment analysis with Textblob

import yfinance as yf
import requests
from textblob import TextBlob

def fetch_news_and_analyze_sentiment(ticker):
    ticker_info = yf.Ticker(ticker)
    news = ticker_info.news
    
    analyzed_news = []
    for article in news[:10]:
        title = article.get('title', 'No title available')
        sentiment = TextBlob(title).sentiment.polarity
        sentiment_label = 'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'
        analyzed_news.append({
            'title': title,
            'publisher': article.get('publisher', 'No publisher available'),
            'source': article.get('source', 'No source available'),
            'link': article.get('link', '#'),
            'sentiment': sentiment_label
        })
    return analyzed_news