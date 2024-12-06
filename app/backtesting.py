import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app.config import Config

class BacktestingAgent:
    def __init__(self):
        self.config = Config()

    def run(self):
        st.title("Strategy Backtesting 📈")

        # Strategy Selection and Parameters
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            stock = st.selectbox(
                "Select Stock",
                self.config.DEFAULT_STOCK_TICKERS
            )
            
        with col2:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=100000,
                step=1000
            )
            
        with col3:
            position_size = st.slider(
                "Position Size (%)",
                min_value=10,
                max_value=100,
                value=100,
                step=10
            )

        # Strategy parameters
        st.subheader("Strategy Parameters")
        strategy = st.selectbox(
            "Select Trading Strategy",
            ["Moving Average Crossover", "RSI Strategy", "Bollinger Bands"]
        )

        # Strategy-specific parameters
        if strategy == "Moving Average Crossover":
            col1, col2 = st.columns(2)
            with col1:
                short_window = st.slider("Short MA Period", 5, 50, 20)
            with col2:
                long_window = st.slider("Long MA Period", 20, 200, 50)
            params = {'short_window': short_window, 'long_window': long_window}
            
        elif strategy == "RSI Strategy":
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_period = st.slider("RSI Period", 5, 30, 14)
            with col2:
                oversold = st.slider("Oversold Level", 20, 40, 30)
            with col3:
                overbought = st.slider("Overbought Level", 60, 80, 70)
            params = {'period': rsi_period, 'oversold': oversold, 'overbought': overbought}
            
        else:  # Bollinger Bands
            col1, col2 = st.columns(2)
            with col1:
                bb_period = st.slider("BB Period", 5, 50, 20)
            with col2:
                bb_std = st.slider("Standard Deviation", 1.0, 3.0, 2.0, 0.1)
            params = {'period': bb_period, 'std_dev': bb_std}

        # Backtest period
        lookback = st.slider(
            "Backtest Period (Days)",
            min_value=30,
            max_value=365,
            value=180
        )

        if stock and st.button("Run Backtest"):
            self.run_backtest(stock, strategy, lookback, initial_capital, position_size/100, params)

    def run_backtest(self, symbol, strategy, lookback, initial_capital, position_size, params):
        try:
            with st.spinner("Running backtest..."):
                # Fetch data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback)
                df = yf.download(symbol, start=start_date, end=end_date)

                # Apply strategy
                if strategy == "Moving Average Crossover":
                    signals = self.moving_average_strategy(df, params)
                elif strategy == "RSI Strategy":
                    signals = self.rsi_strategy(df, params)
                else:
                    signals = self.bollinger_strategy(df, params)

                # Calculate portfolio performance
                portfolio = self.calculate_portfolio(df, signals, initial_capital, position_size)
                
                # Display results
                self.display_results(df, portfolio, symbol, strategy)

        except Exception as e:
            st.error(f"Error in backtesting: {str(e)}")

    def calculate_portfolio(self, df, signals, initial_capital, position_size):
        portfolio = pd.DataFrame(index=df.index)
        portfolio['Signal'] = signals
        portfolio['Price'] = df['Close']
        portfolio['Position'] = portfolio['Signal'].diff()
        
        # Calculate holdings and cash
        portfolio['Holdings'] = 0.0
        portfolio['Cash'] = initial_capital
        
        for i in range(len(portfolio)):
            if i > 0:
                portfolio.iloc[i, portfolio.columns.get_loc('Cash')] = portfolio.iloc[i-1, portfolio.columns.get_loc('Cash')]
                portfolio.iloc[i, portfolio.columns.get_loc('Holdings')] = portfolio.iloc[i-1, portfolio.columns.get_loc('Holdings')]
            
            if portfolio.iloc[i, portfolio.columns.get_loc('Position')] == 1:  # Buy
                shares = (portfolio.iloc[i, portfolio.columns.get_loc('Cash')] * position_size) // portfolio.iloc[i, portfolio.columns.get_loc('Price')]
                cost = shares * portfolio.iloc[i, portfolio.columns.get_loc('Price')]
                portfolio.iloc[i, portfolio.columns.get_loc('Holdings')] += shares
                portfolio.iloc[i, portfolio.columns.get_loc('Cash')] -= cost
                
            elif portfolio.iloc[i, portfolio.columns.get_loc('Position')] == -1:  # Sell
                shares = portfolio.iloc[i, portfolio.columns.get_loc('Holdings')]
                revenue = shares * portfolio.iloc[i, portfolio.columns.get_loc('Price')]
                portfolio.iloc[i, portfolio.columns.get_loc('Holdings')] = 0
                portfolio.iloc[i, portfolio.columns.get_loc('Cash')] += revenue
        
        portfolio['Total'] = portfolio['Cash'] + portfolio['Holdings'] * portfolio['Price']
        portfolio['Returns'] = portfolio['Total'].pct_change()
        
        return portfolio

    def display_results(self, df, portfolio, symbol, strategy):
        # Calculate metrics
        total_return = (portfolio['Total'].iloc[-1] / portfolio['Total'].iloc[0] - 1) * 100
        sharpe_ratio = np.sqrt(252) * (portfolio['Returns'].mean() / portfolio['Returns'].std())
        max_drawdown = ((portfolio['Total'] / portfolio['Total'].cummax()) - 1).min() * 100
        win_rate = len(portfolio[portfolio['Returns'] > 0]) / len(portfolio[portfolio['Returns'] != 0]) * 100
        
        # Display metrics
        st.subheader("Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{total_return:.2f}%")
        with col2:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        with col4:
            st.metric("Win Rate", f"{win_rate:.1f}%")

        # Create subplot with price and portfolio value
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           subplot_titles=(f'Price Chart - {symbol}', 'Portfolio Value'))

        # Price chart with signals
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='blue')), row=1, col=1)
        
        # Buy signals
        buy_signals = portfolio[portfolio['Position'] == 1].index
        fig.add_trace(go.Scatter(
            x=buy_signals, 
            y=df.loc[buy_signals, 'Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ), row=1, col=1)
        
        # Sell signals
        sell_signals = portfolio[portfolio['Position'] == -1].index
        fig.add_trace(go.Scatter(
            x=sell_signals, 
            y=df.loc[sell_signals, 'Close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ), row=1, col=1)
        
        # Portfolio value
        fig.add_trace(go.Scatter(
            x=portfolio.index, 
            y=portfolio['Total'],
            name='Portfolio Value',
            line=dict(color='yellow')
        ), row=2, col=1)

        fig.update_layout(
            height=800,
            template="plotly_dark",
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display trade log
        self.display_trade_log(portfolio)

    def display_trade_log(self, portfolio):
        trades = portfolio[portfolio['Position'] != 0].copy()
        trades['Type'] = trades['Position'].map({1: 'Buy', -1: 'Sell'})
        trades['Value'] = abs(trades['Position'] * trades['Price'] * trades['Holdings'])
        
        st.subheader("Trade Log")
        trade_log = pd.DataFrame({
            'Date': trades.index,
            'Type': trades['Type'],
            'Price': trades['Price'].round(2),
            'Shares': trades['Holdings'].abs(),
            'Value': trades['Value'].round(2),
            'Portfolio Value': trades['Total'].round(2)
        })
        
        st.dataframe(trade_log, use_container_width=True)
        
        # Add download button for trade log
        csv = trade_log.to_csv().encode('utf-8')
        st.download_button(
            "Download Trade Log",
            csv,
            "trade_log.csv",
            "text/csv",
            key='download-trades'
        )

    # Strategy methods remain the same but use params
    def moving_average_strategy(self, df, params):
        df['SMA_short'] = df['Close'].rolling(window=params['short_window']).mean()
        df['SMA_long'] = df['Close'].rolling(window=params['long_window']).mean()
        
        df['Signal'] = 0
        df.loc[df['SMA_short'] > df['SMA_long'], 'Signal'] = 1
        df.loc[df['SMA_short'] < df['SMA_long'], 'Signal'] = 0
        
        return df['Signal']

    def rsi_strategy(self, df, params):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=params['period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=params['period']).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Signal'] = 0
        df.loc[df['RSI'] < params['oversold'], 'Signal'] = 1
        df.loc[df['RSI'] > params['overbought'], 'Signal'] = 0
        
        return df['Signal']

    def bollinger_strategy(self, df, params):
        df['SMA'] = df['Close'].rolling(window=params['period']).mean()
        df['STD'] = df['Close'].rolling(window=params['period']).std()
        
        df['Upper'] = df['SMA'] + (df['STD'] * params['std_dev'])
        df['Lower'] = df['SMA'] - (df['STD'] * params['std_dev'])
        
        df['Signal'] = 0
        df.loc[df['Close'] < df['Lower'], 'Signal'] = 1
        df.loc[df['Close'] > df['Upper'], 'Signal'] = 0
        
        return df['Signal'] 