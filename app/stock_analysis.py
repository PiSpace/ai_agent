import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
# import pandas_ta as ta
from app.news_sentiment import SentimentAnalyzer

class StockAnalysisDashboard:
    def __init__(self):
        self.default_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
            "NVDA", "TSLA", "JPM", "V", "WMT"
        ]
        self.technical_indicators = {
            'SMA': [20, 50, 200],
            'RSI': 14,
            'MACD': {'fast': 12, 'slow': 26, 'signal': 9}
        }
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sentiment_analyzer = SentimentAnalyzer()

    def run(self):
        st.title("Stock Analysis Dashboard 📈")
        
        # Stock Selection with Dropdown and Search
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_ticker = st.selectbox(
                "Select a stock:",
                self.default_tickers,
                format_func=lambda x: f"{x} - {self._get_company_name(x)}"
            )
        with col2:
            custom_ticker = st.text_input(
                "Or enter a custom ticker:",
                help="Enter any valid stock symbol (e.g., AAPL)"
            ).upper()
            
        # Use custom ticker if provided, otherwise use selected ticker
        symbol = custom_ticker if custom_ticker else selected_ticker
        
        # Time Period Selection
        period = st.select_slider(
            "Select Time Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            value="1y"
        )

        if symbol:
            data = self._fetch_stock_data(symbol, period)
            if data is not None:
                # Display Company Info
                self._display_company_info(symbol)
                
                # Technical Analysis Tab
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 Price Analysis", 
                    "📊 Technical Indicators", 
                    "🤖 ML Predictions",
                    "📰 Sentiment Analysis"
                ])
                
                with tab1:
                    self._display_price_analysis(data)
                with tab2:
                    self._display_technical_analysis(data)
                with tab3:
                    self._display_ml_prediction(data)
                with tab4:
                    self.sentiment_analyzer.analyze_stock_sentiment(symbol)

    def _get_company_name(self, ticker):
        try:
            return yf.Ticker(ticker).info.get('longName', ticker)
        except:
            return ticker

    def _fetch_stock_data(self, symbol, period):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history( period=period, interval="1d")
            if data.empty:
                st.error(f"No data found for {symbol}")
                return None
            return data
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None

    def _display_technical_analysis(self, data):
        """Display technical analysis indicators"""
        # Calculate technical indicators
        df = data.copy()
        
        # Calculate SMAs
        for ma in self.technical_indicators['SMA']:
            df[f'SMA_{ma}'] = df['Close'].rolling(window=ma).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.technical_indicators['RSI']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.technical_indicators['RSI']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=self.technical_indicators['MACD']['fast']).mean()
        exp2 = df['Close'].ewm(span=self.technical_indicators['MACD']['slow']).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=self.technical_indicators['MACD']['signal']).mean()
        
        # Create subplots
        fig = make_subplots(
            rows=3, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price & Moving Averages', 'MACD', 'RSI')
        )

        # Plot price and MAs
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add SMAs
        colors = ['orange', 'blue', 'purple']
        for ma, color in zip(self.technical_indicators['SMA'], colors):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[f'SMA_{ma}'],
                    name=f'SMA {ma}',
                    line=dict(color=color)
                ),
                row=1, col=1
            )

        # Add MACD
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Signal'],
                name='Signal',
                line=dict(color='orange')
            ),
            row=2, col=1
        )

        # Add RSI
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='purple')
            ),
            row=3, col=1
        )

        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display Technical Analysis Summary
        st.subheader("Technical Analysis Summary")
        
        # Get latest values
        current_price = df['Close'].iloc[-1]
        sma_values = {ma: df[f'SMA_{ma}'].iloc[-1] for ma in self.technical_indicators['SMA']}
        current_rsi = df['RSI'].iloc[-1]
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['Signal'].iloc[-1]

        # Create analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Moving Averages Analysis**")
            for ma, value in sma_values.items():
                position = "ABOVE" if current_price > value else "BELOW"
                st.write(f"• Price is {position} SMA{ma} (${value:.2f})")

        with col2:
            st.write("**Momentum Indicators**")
            
            # RSI Analysis
            rsi_signal = (
                "OVERBOUGHT" if current_rsi > 70
                else "OVERSOLD" if current_rsi < 30
                else "NEUTRAL"
            )
            st.write(f"• RSI: {current_rsi:.2f} - {rsi_signal}")
            
            # MACD Analysis
            macd_signal = "BULLISH" if current_macd > current_signal else "BEARISH"
            st.write(f"• MACD: {current_macd:.2f} - {macd_signal}")

        # Overall Signal
        signals = []
        # MA Signals
        ma_bullish = sum(1 for ma_value in sma_values.values() if current_price > ma_value)
        ma_bearish = len(sma_values) - ma_bullish
        if ma_bullish > ma_bearish:
            signals.append(1)
        else:
            signals.append(-1)
        
        # RSI Signal
        if current_rsi > 70:
            signals.append(-1)
        elif current_rsi < 30:
            signals.append(1)
        else:
            signals.append(0)
        
        # MACD Signal
        if current_macd > current_signal:
            signals.append(1)
        else:
            signals.append(-1)
        
        # Calculate overall signal
        overall_signal = sum(signals) / len(signals)
        
        signal_text = (
            "🟢 STRONG BUY" if overall_signal > 0.5
            else "🟡 BUY" if overall_signal > 0
            else "🟡 SELL" if overall_signal > -0.5
            else "🔴 STRONG SELL"
        )
        
        st.subheader("Technical Signal")
        st.write(f"### {signal_text}")

    def _display_trend_analysis(self, df):
        st.subheader("Trend Analysis")
        
        # Get current trend
        current_trend = df['Trend'].iloc[-1]
        trend_duration = len(df[df['Trend'] == current_trend].iloc[-1:])
        
        # Calculate trend metrics
        trend_changes = df['Trend'].ne(df['Trend'].shift()).sum()
        uptrend_days = (df['Trend'] == 'Uptrend').sum()
        downtrend_days = (df['Trend'] == 'Downtrend').sum()
        
        # Identify trend patterns
        df['Swing_High'] = df['High'].rolling(window=5, center=True).apply(lambda x: x[2] == max(x))
        df['Swing_Low'] = df['Low'].rolling(window=5, center=True).apply(lambda x: x[2] == min(x))
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Trend", current_trend,
                     delta="↑" if current_trend == "Uptrend" else "↓" if current_trend == "Downtrend" else "→")
        with col2:
            st.metric("Trend Duration", f"{trend_duration} days")
        with col3:
            st.metric("Trend Changes", trend_changes)

        # Trend Pattern Analysis
        st.subheader("Trend Pattern Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Trend Distribution
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Bar(
                x=['Uptrend', 'Downtrend', 'Neutral'],
                y=[uptrend_days, downtrend_days, len(df) - uptrend_days - downtrend_days],
                marker_color=['green', 'red', 'gray']
            ))
            fig_dist.update_layout(title="Trend Distribution")
            st.plotly_chart(fig_dist)
            
        with col2:
            # Trend Strength
            avg_gain = df['Close'].diff().where(df['Close'].diff() > 0, 0).mean()
            avg_loss = abs(df['Close'].diff().where(df['Close'].diff() < 0, 0).mean())
            trend_strength = avg_gain / avg_loss if avg_loss != 0 else float('inf')
            
            st.metric("Trend Strength", f"{trend_strength:.2f}",
                     help="Ratio of average gains to average losses. Values > 1 indicate stronger uptrends")
            
            # Momentum indicators
            st.write("**Momentum Indicators:**")
            rsi = df['RSI'].iloc[-1]
            st.progress(rsi/100, text=f"RSI: {rsi:.1f}")

        # Trend Characteristics
        st.subheader("Trend Characteristics")
        
        # Calculate higher highs/lows for uptrend and lower highs/lows for downtrend
        window = 20
        df['Higher_High'] = df['High'].rolling(window=window).apply(lambda x: x[-1] > max(x[:-1]) if len(x) > 1 else False)
        df['Higher_Low'] = df['Low'].rolling(window=window).apply(lambda x: x[-1] > min(x[:-1]) if len(x) > 1 else False)
        df['Lower_High'] = df['High'].rolling(window=window).apply(lambda x: x[-1] < max(x[:-1]) if len(x) > 1 else False)
        df['Lower_Low'] = df['Low'].rolling(window=window).apply(lambda x: x[-1] < min(x[:-1]) if len(x) > 1 else False)

        recent_data = df.tail(window)
        higher_highs = recent_data['Higher_High'].sum()
        higher_lows = recent_data['Higher_Low'].sum()
        lower_highs = recent_data['Lower_High'].sum()
        lower_lows = recent_data['Lower_Low'].sum()

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Uptrend Characteristics:**")
            st.write(f"- Higher Highs: {higher_highs}")
            st.write(f"- Higher Lows: {higher_lows}")
            
        with col2:
            st.write("**Downtrend Characteristics:**")
            st.write(f"- Lower Highs: {lower_highs}")
            st.write(f"- Lower Lows: {lower_lows}")

        # Support and Resistance Levels
        st.subheader("Support and Resistance Levels")
        pivot_points = self._calculate_pivot_points(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Support Levels:**")
            for i, level in enumerate(pivot_points['support'], 1):
                st.write(f"S{i}: ${level:.2f}")
        
        with col2:
            st.write("**Resistance Levels:**")
            for i, level in enumerate(pivot_points['resistance'], 1):
                st.write(f"R{i}: ${level:.2f}")

    def _calculate_pivot_points(self, df):
        # Calculate classic pivot points
        pivot = (df['High'].iloc[-1] + df['Low'].iloc[-1] + df['Close'].iloc[-1]) / 3
        
        r1 = 2 * pivot - df['Low'].iloc[-1]
        r2 = pivot + (df['High'].iloc[-1] - df['Low'].iloc[-1])
        r3 = r1 + (df['High'].iloc[-1] - df['Low'].iloc[-1])
        
        s1 = 2 * pivot - df['High'].iloc[-1]
        s2 = pivot - (df['High'].iloc[-1] - df['Low'].iloc[-1])
        s3 = s1 - (df['High'].iloc[-1] - df['Low'].iloc[-1])
        
        return {
            'pivot': pivot,
            'resistance': [r1, r2, r3],
            'support': [s1, s2, s3]
        }

    def _display_ml_prediction(self, data):
        st.subheader('LSTM Price Prediction Model 🤖')
        
        col1, col2 = st.columns(2)
        with col1:
            lookback = st.slider('Lookback Period (days)', 30, 100, 60)
        with col2:
            forecast_days = st.slider('Forecast Horizon (days)', 5, 30, 7)
            
        if st.button('Generate Price Predictions'):
            with st.spinner('Training LSTM model... This may take a few minutes.'):
                predictions, metrics = self._train_lstm_model(data, lookback, forecast_days)
                
                if predictions and metrics:
                    # Display metrics
                    st.success('Model training complete!')
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Model Score (R²)", f"{metrics['r2']:.4f}")
                    with metric_cols[1]:
                        st.metric("MAE", f"${metrics['mae']:.2f}")
                    with metric_cols[2]:
                        st.metric("RMSE", f"${metrics['rmse']:.2f}")
                    with metric_cols[3]:
                        st.metric("Next Day Prediction", f"${predictions['next_day']:.2f}")
                    
                    # Plot predictions
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index[-30:],
                        y=data['Close'].iloc[-30:],
                        mode='lines',
                        name='Historical'
                    ))
                    fig.add_trace(go.Scatter(
                        x=predictions['dates'],
                        y=predictions['values'],
                        mode='lines',
                        name='Predicted',
                        line=dict(dash='dash')
                    ))
                    fig.update_layout(
                        title='Price Predictions',
                        xaxis_title='Date',
                        yaxis_title='Price ($)'
                    )
                    st.plotly_chart(fig)
                    
                    # Prediction Analysis
                    with st.expander("Detailed Prediction Analysis"):
                        st.write("### Price Trajectory Analysis")
                        trend = predictions['values'][-1] - predictions['values'][0]
                        if trend > 0:
                            st.write("🟢 The model predicts an UPWARD trend")
                        else:
                            st.write("🔴 The model predicts a DOWNWARD trend")
                        
                        st.write(f"Expected price movement: ${abs(trend):.2f}")
                        st.write(f"Predicted volatility: {metrics['volatility']:.2f}%")
                        
                        st.write("### Confidence Intervals")
                        st.write(f"95% Confidence Range: ${predictions['lower_bound']:.2f} - ${predictions['upper_bound']:.2f}")

    def _train_lstm_model(self, data, lookback, forecast_days):
        try:
            # Prepare data
            df = data['Close'].values.reshape(-1, 1)
            scaled_data = self.scaler.fit_transform(df)
            
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i, 0])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)
            
            # Reshape X to match LSTM input requirements
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build and train model
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred = self.scaler.inverse_transform(y_pred)
            y_test = self.scaler.inverse_transform([y_test]).T
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = 1 - (mse / np.var(y_test))
            
            # Future predictions
            last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
            future_predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(forecast_days):
                next_pred = model.predict(current_sequence)
                future_predictions.append(next_pred[0, 0])
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_pred[0, 0]
            
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = self.scaler.inverse_transform(future_predictions)
            
            # Prepare prediction dates
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
            
            # Calculate confidence intervals
            std_dev = np.std(y_pred - y_test)
            predictions = {
                'dates': future_dates,
                'values': future_predictions.flatten(),
                'next_day': future_predictions[0][0],
                'lower_bound': future_predictions[0][0] - 1.96 * std_dev,
                'upper_bound': future_predictions[0][0] + 1.96 * std_dev
            }
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'volatility': std_dev * 100 / np.mean(y_test)
            }
            
            return predictions, metrics
            
        except Exception as e:
            st.error(f"Error in LSTM model: {str(e)}")
            return None, None

    def _display_financial_analysis(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Financial Metrics
            st.subheader("Key Financial Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A'):.2f}")
                st.metric("EPS", f"${info.get('trailingEps', 'N/A'):.2f}")
            
            with col2:
                st.metric("Revenue Growth", f"{info.get('revenueGrowth', 'N/A')*100:.1f}%")
                st.metric("Profit Margin", f"{info.get('profitMargins', 'N/A')*100:.1f}%")
            
            with col3:
                st.metric("Debt to Equity", f"{info.get('debtToEquity', 'N/A'):.2f}")
                st.metric("Return on Equity", f"{info.get('returnOnEquity', 'N/A')*100:.1f}%")
            
            # Financial Statements with Visualizations
            if st.checkbox("Show Financial Analysis"):
                tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
                
                with tab1:
                    income_stmt = stock.income_stmt
                    st.dataframe(income_stmt)
                    
                    # Revenue and Profit Visualization
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=income_stmt.columns,
                        y=income_stmt.loc['Total Revenue'],
                        name='Revenue',
                        marker_color='lightblue'
                    ))
                    fig.add_trace(go.Bar(
                        x=income_stmt.columns,
                        y=income_stmt.loc['Net Income'],
                        name='Net Income',
                        marker_color='lightgreen'
                    ))
                    fig.update_layout(title="Revenue vs Net Income", barmode='group')
                    st.plotly_chart(fig)
                
                with tab2:
                    balance_sheet = stock.balance_sheet
                    st.dataframe(balance_sheet)
                    
                    # Assets vs Liabilities
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=balance_sheet.columns,
                        y=balance_sheet.loc['Total Assets'],
                        name='Total Assets',
                        marker_color='blue'
                    ))
                    fig.add_trace(go.Bar(
                        x=balance_sheet.columns,
                        y=balance_sheet.loc['Total Liabilities Net Minority Interest'],
                        name='Total Liabilities',
                        marker_color='red'
                    ))
                    fig.update_layout(title="Assets vs Liabilities", barmode='group')
                    st.plotly_chart(fig)
                
                with tab3:
                    cashflow = stock.cashflow
                    st.dataframe(cashflow)
                    
                    # Cash Flow Components
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=cashflow.columns,
                        y=cashflow.loc['Operating Cash Flow'],
                        name='Operating Cash Flow',
                        marker_color='green'
                    ))
                    fig.add_trace(go.Bar(
                        x=cashflow.columns,
                        y=cashflow.loc['Free Cash Flow'],
                        name='Free Cash Flow',
                        marker_color='purple'
                    ))
                    fig.update_layout(title="Operating vs Free Cash Flow", barmode='group')
                    st.plotly_chart(fig)
                    
        except Exception as e:
            st.error(f"Error fetching financial data: {str(e)}")

    def _display_company_info(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Company Overview
            st.subheader("Company Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}",
                    f"{info.get('regularMarketChangePercent', 0):.2f}%"
                )
            
            with col2:
                market_cap = info.get('marketCap', 0)
                if market_cap > 1e9:
                    market_cap_str = f"${market_cap/1e9:.2f}B"
                else:
                    market_cap_str = f"${market_cap/1e6:.2f}M"
                st.metric("Market Cap", market_cap_str)
            
            with col3:
                st.metric(
                    "52W Range",
                    f"${info.get('fiftyTwoWeekLow', 0):.2f} - ${info.get('fiftyTwoWeekHigh', 0):.2f}"
                )
            
            # Company Details
            with st.expander("Company Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Sector:**", info.get('sector', 'N/A'))
                    st.write("**Industry:**", info.get('industry', 'N/A'))
                    st.write("**Country:**", info.get('country', 'N/A'))
                with col2:
                    st.write("**Website:**", info.get('website', 'N/A'))
                    st.write("**Employees:**", f"{info.get('fullTimeEmployees', 0):,}")
                    st.write("**Founded:**", info.get('founded', 'N/A'))
                
                st.write("**Business Summary:**")
                st.write(info.get('longBusinessSummary', 'No summary available'))
                
        except Exception as e:
            st.error(f"Error fetching company info: {str(e)}")

    def _display_price_analysis(self, data):
        """Display price analysis with candlestick chart and volume"""
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           subplot_titles=('Price', 'Volume'),
                           row_heights=[0.7, 0.3])

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # Add volume bar chart
        colors = ['red' if row['Open'] > row['Close'] else 'green' 
                 for index, row in data.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )

        # Price Statistics
        price_stats = pd.DataFrame({
            'Current': data['Close'].iloc[-1],
            'Open': data['Open'].iloc[-1],
            'High': data['High'].iloc[-1],
            'Low': data['Low'].iloc[-1],
            'Change %': ((data['Close'].iloc[-1] - data['Open'].iloc[-1]) / data['Open'].iloc[-1] * 100),
            'Volume': data['Volume'].iloc[-1]
        }, index=['Value']).T

        # Update layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=False,
            title_text="Price Analysis",
            template="plotly_dark"
        )

        # Show the plot
        st.plotly_chart(fig, use_container_width=True)

        # Display price statistics
        st.subheader("Price Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Price",
                f"${price_stats.loc['Current', 'Value']:.2f}",
                f"{price_stats.loc['Change %', 'Value']:.2f}%"
            )
        
        with col2:
            st.metric(
                "Day Range",
                f"${price_stats.loc['Low', 'Value']:.2f} - ${price_stats.loc['High', 'Value']:.2f}"
            )
        
        with col3:
            st.metric(
                "Volume",
                f"{price_stats.loc['Volume', 'Value']:,.0f}"
            )

        # Additional Analysis
        with st.expander("Detailed Price Analysis"):
            # Calculate price metrics
            daily_returns = data['Close'].pct_change()
            volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
            avg_volume = data['Volume'].mean()
            volume_change = ((data['Volume'].iloc[-1] - avg_volume) / avg_volume) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Price Metrics:**")
                st.write(f"• Volatility (Annual): {volatility:.2f}%")
                st.write(f"• Average Daily Range: ${(data['High'] - data['Low']).mean():.2f}")
                st.write(f"• 52-Week High: ${data['High'].max():.2f}")
                st.write(f"• 52-Week Low: ${data['Low'].min():.2f}")
            
            with col2:
                st.write("**Volume Analysis:**")
                st.write(f"• Average Volume: {avg_volume:,.0f}")
                st.write(f"• Volume Change: {volume_change:.2f}%")
                st.write(f"• Volume/Price Correlation: {data['Volume'].corr(data['Close']):.2f}")

