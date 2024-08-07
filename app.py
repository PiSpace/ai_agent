import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import numpy as np
import datetime
from dotenv import load_dotenv
import os
from io import BytesIO
from PIL import Image
import base64
from openai import OpenAI

#Custome modules
from news_sentiment import fetch_news_and_analyze_sentiment
from ml_predictor import prepare_data, train_and_predict, advanced_ml_model
from algo_trading import load_data, calculate_risk_metrics, calculate_max_drawdown



load_dotenv()

# Set up the Streamlit app
st.set_page_config(page_title="AI Investment Agent 📈🤖", layout="wide")
st.sidebar.image("Stock_Logo.png", width=250)

# Define pages
pages = ["Stock Analysis Dashboard", "AI Financial Assistant", "Stock Comparison Agent"]
selected_page = st.sidebar.selectbox("Select a Page", pages)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Get OpenAI API key from user
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY"))

def initialize_assistant(api_key):
    if "assistant" not in st.session_state:
        st.session_state.assistant = OpenAI(api_key=api_key)

initialize_assistant(openai_api_key)

# Function to interact with OpenAI GPT-4o-mini
def get_gpt4_response(api_key, user_input, image_url=None):
    client = st.session_state.assistant

    if image_url:
        messages = [
            {"role": "user", "content": "You are a financial assistant."[
                {"type": "text", "text": user_input},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}
        ]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": user_input}]}]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        #max_tokens=300, if you want to your model give short reply you can set a token limit
    )

    return response.choices[0].message.content.strip()

# Function to fetch stock data
@st.cache_data
def fetch_data(ticker, start_date, end_date, ma_options):
    data = yf.download(ticker, start=start_date, end=end_date)
    for ma in ma_options:
        window = int(ma.split('-')[0])
        data[f'{window}_day_MA'] = data['Adj Close'].rolling(window=window).mean()
    data['RSI_14'] = ta.rsi(data['Adj Close'], length=14)
    macd = ta.macd(data['Adj Close'], fast=12, slow=26, signal=9)
    bollinger = ta.bbands(data['Adj Close'], length=20, std=2)
    data = pd.concat([data, macd, bollinger], axis=1)
    data['Signal'] = np.where(data['10_day_MA'] > data['20_day_MA'], 1, -1)
    data['Position'] = data['Signal'].replace(to_replace=0, method='ffill')
    data['Market Returns'] = data['Adj Close'].pct_change()
    data['Strategy Returns'] = data['Market Returns'] * data['Position'].shift(1)
    data['Volume_MA10'] = data['Volume'].rolling(window=10).mean()
    return data

# Function to plot stock data using Plotly
def plot_stock_data(data, ticker, ma_options):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=(f'{ticker} Stock Price', 'Volume'), row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="OHLC"), row=1, col=1)
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, ma in enumerate(ma_options):
        window = int(ma.split('-')[0])
        fig.add_trace(go.Scatter(x=data.index, y=data[f'{window}_day_MA'], name=f'{window}-day MA', line=dict(color=colors[i % len(colors)])), row=1, col=1)
    
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='rgb(158,202,225)'), row=2, col=1)
    
    fig.update_layout(xaxis_rangeslider_visible=False, height=600, width=1000, showlegend=True)
    st.plotly_chart(fig)

if selected_page == "AI Financial Assistant":
    st.title("AI Financial Assistant 🤖")
    st.caption("Interact with the AI financial assistant to get advice on the stock market and financial insights.")

    user_prompt = st.text_input("Ask the Financial Assistant a question about the stock market or financial advice:", key='financial_prompt')
    
    uploaded_image = st.file_uploader("Upload an image for analysis (optional)", type=["jpg", "jpeg", "png"])

    if user_prompt and openai_api_key:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.spinner('Thinking...'):  
            if uploaded_image:
                image = Image.open(uploaded_image)
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                image_bytes = buffered.getvalue()
                advice = get_gpt4_response(openai_api_key, user_prompt, image=image_bytes)
            else:
                advice = get_gpt4_response(openai_api_key, user_prompt)
            
            st.session_state.chat_history.append({"role": "assistant", "content": advice})
            st.query_params()

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"<div style='text-align: left;'><b>User:</b> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left;'><b>PiBot 🤖:</b> {message['content']}</div>", unsafe_allow_html=True)


elif selected_page == "Stock Analysis Dashboard":
    st.title("Stock Analysis Dashboard 📊")
    st.caption("Analyze stock prices, predict future prices, and get recommendations.")
    tickers = ['AAPL', 'GOOGL', 'NVDA', 'MSFT', 'AMZN', 'TSLA', 'META']
    selected_ticker = st.sidebar.selectbox('Select Stock Ticker:', tickers)
    user_input_ticker = st.sidebar.text_input('Or enter custom Stock Ticker:', value='', key='custom_ticker')
    ticker_to_use = user_input_ticker if user_input_ticker else selected_ticker

    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2023-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime(datetime.date.today()))
    initial_investment = st.sidebar.number_input('Enter your initial investment amount:', min_value=0.0, value=10000.0, step=1000.0)
    ma_options = st.sidebar.multiselect('Select Moving Averages:', ['10-day', '20-day', '50-day'], default=['10-day', '20-day'])

    data = fetch_data(ticker_to_use, start_date, end_date, ma_options)
    plot_stock_data(data, ticker_to_use, ma_options)

    # ML Prediction and Visualization
    st.subheader('Next Day Price Prediction with ML 🤖')

    def plot_last_days(actual, predicted, last_n_days=10):
        actual_sorted = actual.sort_index()
        predicted_sorted = pd.Series(predicted, index=actual.index).sort_index()
        actual_last_n = actual_sorted.tail(last_n_days)
        predicted_last_n = predicted_sorted.tail(last_n_days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual_last_n.index, y=actual_last_n, mode='lines+markers', name='Actual', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=predicted_last_n.index, y=predicted_last_n, mode='lines+markers', name='Predicted', line=dict(color='red')))
        fig.update_layout(title='Actual vs Predicted Prices for the Last 10 Days', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)

    def display_comparison_table(actual, predicted):
        comparison_df = pd.DataFrame({'Actual': actual, 'Predicted': predicted}).sort_index()
        st.write("Comparison of Actual and Predicted Prices for the Last 10 Days:")
        st.dataframe(comparison_df.style.format("${:.2f}"))

    if st.button('Train ML Model and Predict Next Day Price'):
        df_prepared = prepare_data(data)
        predicted_price, model_score, y_test, y_pred_test, mae, mse = advanced_ml_model(df_prepared)
        st.metric(label="Predicted Next Day Price", value=f"${predicted_price:.2f}")
        st.metric(label="Model Score (R^2)", value=f"{model_score:.2f}")
        actual_last_10 = y_test.sort_index().tail(10)
        predicted_last_10 = pd.Series(y_pred_test, index=y_test.index).sort_index().tail(10)
        plot_last_days(actual_last_10, predicted_last_10)
        display_comparison_table(actual_last_10, predicted_last_10)
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame({'MAE': [mae], 'MSE': [mse], 'R² Score': [model_score]})
        st.dataframe(metrics_df.style.format("{:.4f}"))

    # Simulate ML strategy signals and returns
    def simulate_ml_strategy(data):
        np.random.seed(42)
        data['ML Signal'] = np.random.choice([-1, 1], size=len(data))
        data['ML Returns'] = data['Adj Close'].pct_change() * data['ML Signal'].shift()
        return data

    data = simulate_ml_strategy(data)

    # Calculate and display investment performance
    def display_investment_performance(data, initial_investment, strategy_column, title):
        investment = initial_investment * (1 + data[strategy_column]).cumprod()
        profit_or_loss = investment.iloc[-1] - initial_investment
        st.write(f"{title}: Final Investment Value: ${investment.iloc[-1]:,.2f} (Profit/Loss: ${profit_or_loss:,.2f})")
        return investment, profit_or_loss

    ml_investment, ml_profit_or_loss = display_investment_performance(data, initial_investment, 'ML Returns', 'Machine Learning Strategy')
    sma_investment, sma_profit_or_loss = display_investment_performance(data, initial_investment, 'Strategy Returns', 'Simple Moving Average Strategy')

    # Plot the investment over time
    st.write("Investment Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=ml_investment, mode='lines', name='ML Strategy'))
    fig.add_trace(go.Scatter(x=data.index, y=sma_investment, mode='lines', name='SMA Strategy'))
    fig.update_layout(xaxis_title='Date', yaxis_title='Investment Value ($)', yaxis_tickprefix='$', yaxis_ticksuffix='K')
    st.plotly_chart(fig)

    # Fetch and display news
    st.subheader('Latest News')
    news_data = fetch_news_and_analyze_sentiment(ticker_to_use)
    sentiment_score = sum([1 if article['sentiment'] == 'Positive' else -1 if article['sentiment'] == 'Negative' else 0 for article in news_data])
    st.write(f"Overall Sentiment Score: {sentiment_score}")

    # Display news articles in 2 columns
    st.subheader('News Articles')
    col1, col2 = st.columns(2)
    for i, article in enumerate(news_data[:6]):
        column = col1 if i % 2 == 0 else col2
        with column:
            st.markdown(f"""
            <div style="border:1px solid #ccc; padding:10px; border-radius:5px; margin-bottom:10px;">
                <h4>{article['title']}</h4>
                <p><b>Published by:</b> {article['publisher']} | <b>Source:</b> {article['source']}</p>
                <p><b>Sentiment:</b> {article['sentiment']}</p>
                <a href="{article['link']}">Read more</a>
            </div>
            """, unsafe_allow_html=True)

    # Display sentiment gauge
    st.subheader('Sentiment Gauge')
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        title={'text': "Overall Sentiment"},
        gauge={'axis': {'range': [-10, 10]},
               'bar': {'color': "darkblue"},
               'steps': [{'range': [-10, 0], 'color': "red"}, {'range': [0, 10], 'color': "green"}]}))
    st.plotly_chart(gauge_fig)



    # Buy, Sell, Hold Recommendations
    st.sidebar.subheader("Robo-Advisor Recommendations")
    recommendation = data['Signal'].iloc[-1]
    if recommendation == 1:
        st.sidebar.success("Recommendation: **Buy**")
        st.sidebar.markdown("<div style='background-color: green; padding: 10px; color: white; text-align: center;'>Buy</div>", unsafe_allow_html=True)
    elif recommendation == -1:
        st.sidebar.error("Recommendation: **Sell**")
        st.sidebar.markdown("<div style='background-color: red; padding: 10px; color: white; text-align: center;'>Sell</div>", unsafe_allow_html=True)
    else:
        st.sidebar.info("Recommendation: **Hold**")
        st.sidebar.markdown("<div style='background-color: yellow; padding: 10px; text-align: center;'>Hold</div>", unsafe_allow_html=True)

elif selected_page == "Stock Comparison Agent":
    st.title("Stock Comparison Agent 📊")
    st.caption("Compare the performance of two stocks using the AI agent.")

    # Stock Comparison Section
    tickers = ['AAPL', 'GOOGL', 'NVDA', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'DIS']
    tickers2 = ['','AAPL', 'GOOGL', 'NVDA', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'DIS']
    stock1 = st.selectbox("Select the first stock", tickers)
    stock2 = st.selectbox("Select the second stock", tickers2)

    if stock1 and stock2 and openai_api_key:
        query = f"Compare {stock1} to {stock2}. Use every tool you have."
        with st.spinner('Comparing stocks...'):
            comparison_response = get_gpt4_response(openai_api_key, query)
        st.write(comparison_response)


# Add a footer
st.markdown("---")
st.markdown("Developed by PiSpace.co. 2024")