import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from app.config import Config

class StockComparisonAgent:
    def __init__(self):
        self.config = Config()

    def run(self):
        st.title("Stock Comparison Agent 📊")
        
        # Stock Selection
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_stocks = st.multiselect(
                "Select stocks to compare",
                self.config.DEFAULT_STOCK_TICKERS,
                default=self.config.DEFAULT_STOCK_TICKERS[:2]
            )
        
        with col2:
            # Time Period Selection
            period = st.selectbox(
                "Select Time Period",
                options=list(self.config.TIME_PERIODS.keys()),
                index=list(self.config.TIME_PERIODS.keys()).index('1Y')
            )
        
        if selected_stocks:
            self.compare_stocks(selected_stocks, self.config.TIME_PERIODS[period])
    
    def compare_stocks(self, stocks, period):
        try:
            # Fetch data
            data = {}
            with st.spinner("Fetching stock data..."):
                for stock in stocks:
                    ticker = yf.Ticker(stock)
                    data[stock] = ticker.history(period=period)
            
            # Create comparison chart
            fig = go.Figure()
            for stock in stocks:
                normalized_prices = (data[stock]['Close'] / data[stock]['Close'].iloc[0]) * 100
                fig.add_trace(go.Scatter(
                    x=data[stock].index,
                    y=normalized_prices,
                    name=stock,
                    mode='lines'
                ))
            
            fig.update_layout(
                title="Stock Price Comparison (Normalized)",
                xaxis_title="Date",
                yaxis_title="Normalized Price (%)",
                height=self.config.CHART_HEIGHT,
                width=self.config.CHART_WIDTH,
                template="plotly_dark",  # Using a valid plotly template
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display key metrics comparison
            self.display_metrics_comparison(stocks, data)
            
        except Exception as e:
            st.error(f"Error comparing stocks: {str(e)}")
    
    def display_metrics_comparison(self, stocks, data):
        try:
            metrics = pd.DataFrame()
            
            for stock in stocks:
                ticker = yf.Ticker(stock)
                info = ticker.info
                
                # Format market cap to billions
                market_cap = info.get('marketCap', 0)
                market_cap_b = f"${market_cap / 1e9:.2f}B" if market_cap else 'N/A'
                
                metrics.loc[stock, 'Market Cap'] = market_cap_b
                metrics.loc[stock, 'P/E Ratio'] = f"{info.get('trailingPE', 'N/A'):.2f}"
                metrics.loc[stock, 'Revenue Growth'] = f"{info.get('revenueGrowth', 0) * 100:.1f}%" if info.get('revenueGrowth') else 'N/A'
                metrics.loc[stock, 'Profit Margin'] = f"{info.get('profitMargins', 0) * 100:.1f}%" if info.get('profitMargins') else 'N/A'
                
                # Calculate additional metrics
                stock_data = data[stock]
                ytd_return = ((stock_data['Close'][-1] / stock_data['Close'][0]) - 1) * 100
                volatility = stock_data['Close'].pct_change().std() * (252 ** 0.5) * 100
                
                metrics.loc[stock, 'YTD Return'] = f"{ytd_return:.1f}%"
                metrics.loc[stock, 'Volatility (Annual)'] = f"{volatility:.1f}%"
            
            # Display metrics
            st.subheader("Key Metrics Comparison")
            st.dataframe(metrics, use_container_width=True)
            
            # Add download button for metrics
            csv = metrics.to_csv().encode('utf-8')
            st.download_button(
                "Download Metrics CSV",
                csv,
                "stock_metrics.csv",
                "text/csv",
                key='download-metrics'
            )
            
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
