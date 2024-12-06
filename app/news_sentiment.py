import yfinance as yf
import pandas as pd
import streamlit as st
from textblob import TextBlob
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re

class SentimentAnalyzer:
    def __init__(self):
        self.news_sources = {
            'Yahoo Finance': 'https://finance.yahoo.com/quote/{}/news',
            'MarketWatch': 'https://www.marketwatch.com/investing/stock/{}',
            'Reuters': 'https://www.reuters.com/companies/{}'
        }

    def analyze_stock_sentiment(self, ticker):
        st.subheader(" News Sentiment Analysis")
        
        try:
            # Fetch news from multiple sources
            with st.spinner("Analyzing news sentiment..."):
                news_data = self.fetch_news(ticker)
                if news_data.empty:
                    st.warning("No news articles found for analysis.")
                    return
                
                sentiment_df = self.analyze_sentiment(news_data)
                self.display_sentiment_analysis(sentiment_df, ticker)
                
        except Exception as e:
            st.error(f"Error analyzing sentiment: {str(e)}")

    def fetch_news(self, ticker):
        """Fetch news from multiple sources"""
        news_data = []
        
        try:
            # Yahoo Finance news through yfinance
            stock = yf.Ticker(ticker)
            yahoo_news = stock.news
            
            if yahoo_news:  # Check if news exists
                for article in yahoo_news[:15]:  # Limit to recent articles
                    news_data.append({
                        'date': datetime.fromtimestamp(article.get('providerPublishTime', 0)),
                        'title': article.get('title', ''),
                        'summary': article.get('summary', ''),
                        'source': 'Yahoo Finance',
                        'link': article.get('link', '#')
                    })
            
            return pd.DataFrame(news_data)
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def analyze_sentiment(self, news_df):
        """Analyze sentiment of news articles"""
        def get_sentiment(text):
            if not isinstance(text, str):
                return 'Neutral'
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            
            # Classify sentiment
            if sentiment > 0.1:
                return 'Positive'
            elif sentiment < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
            
        def get_sentiment_score(text):
            if not isinstance(text, str):
                return 0.0
            return TextBlob(text).sentiment.polarity

        try:
            # Analyze both title and summary
            news_df['title_sentiment'] = news_df['title'].apply(get_sentiment)
            news_df['title_score'] = news_df['title'].apply(get_sentiment_score)
            news_df['summary_sentiment'] = news_df['summary'].apply(get_sentiment)
            news_df['summary_score'] = news_df['summary'].apply(get_sentiment_score)
            
            # Calculate combined sentiment score
            news_df['combined_score'] = (news_df['title_score'] + news_df['summary_score']) / 2
            
            return news_df
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            return news_df

    def display_sentiment_analysis(self, df, ticker):
        """Display sentiment analysis results"""
        try:
            if df.empty:
                st.warning("No sentiment data available for display.")
                return

            # Overall sentiment metrics
            avg_sentiment = df['combined_score'].mean()
            sentiment_std = df['combined_score'].std()
            recent_sentiment = df.iloc[0:5]['combined_score'].mean() if len(df) >= 5 else avg_sentiment
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                sentiment_status = "Positive 🟢" if avg_sentiment > 0.1 else "Negative 🔴" if avg_sentiment < -0.1 else "Neutral 🟡"
                st.metric("Overall Sentiment", sentiment_status, f"{avg_sentiment:.2f}")
            with col2:
                st.metric("Sentiment Volatility", f"{sentiment_std:.2f}")
            with col3:
                st.metric("Recent Sentiment", f"{recent_sentiment:.2f}")

            # Sentiment trend chart
            if len(df) > 0:  # Check if there's data to plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['combined_score'],
                    mode='lines+markers',
                    name='Sentiment Score',
                    line=dict(color='yellow'),
                    marker=dict(
                        color=df['combined_score'].apply(
                            lambda x: 'green' if x > 0.1 else 'red' if x < -0.1 else 'yellow'
                        )
                    )
                ))
                
                fig.update_layout(
                    title=f"Sentiment Trend for {ticker}",
                    xaxis_title="Date",
                    yaxis_title="Sentiment Score",
                    template="plotly_dark",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)

            # Display news with sentiment
            if len(df) > 0:
                st.subheader("Recent News Analysis")
                for _, row in df.head(10).iterrows():
                    sentiment_color = (
                        "🟢" if row['combined_score'] > 0.1 
                        else "🔴" if row['combined_score'] < -0.1 
                        else "🟡"
                    )
                    
                    with st.expander(f"{sentiment_color} {row['title']}"):
                        st.write(f"**Source:** {row['source']}")
                        st.write(f"**Date:** {row['date']}")
                        st.write(f"**Summary:** {row['summary']}")
                        st.write(f"**Sentiment Score:** {row['combined_score']:.2f}")
                        st.markdown(f"[Read More]({row['link']})")

        except Exception as e:
            st.error(f"Error displaying sentiment analysis: {str(e)}")

    def get_sentiment_summary(self, ticker):
        """Get a quick sentiment summary for the stock"""
        try:
            news_df = self.fetch_news(ticker)
            if news_df.empty:
                return "Neutral", 0
            
            sentiment_df = self.analyze_sentiment(news_df)
            avg_sentiment = sentiment_df['combined_score'].mean()
            
            if avg_sentiment > 0.1:
                return "Positive", avg_sentiment
            elif avg_sentiment < -0.1:
                return "Negative", avg_sentiment
            else:
                return "Neutral", avg_sentiment
            
        except Exception:
            return "Neutral", 0
