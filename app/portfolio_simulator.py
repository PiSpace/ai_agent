import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

class PortfolioSimulator:
    def __init__(self):
        self.default_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]
        self.risk_free_rate = 0.04  # 4% risk-free rate

    def run(self):
        st.title("Portfolio Simulator 💼")
        st.write("""
        Simulate and analyze different portfolio combinations to optimize your investment strategy. 
        This tool helps you understand potential returns, risks, and portfolio performance metrics.
        """)

        # Portfolio Setup Section
        st.header("1. Portfolio Setup")
        col1, col2 = st.columns(2)
        
        with col1:
            investment_amount = st.number_input(
                "Initial Investment Amount ($)", 
                min_value=1000, 
                value=10000, 
                step=1000,
                help="Enter the amount you want to invest"
            )
            
            selected_stocks = st.multiselect(
                "Select Stocks for Your Portfolio",
                self.default_stocks + ["FB", "TSLA", "JPM", "V", "WMT"],
                default=self.default_stocks[:3],
                help="Choose stocks to include in your portfolio"
            )

        with col2:
            start_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=365),
                help="Historical data start date"
            )
            end_date = st.date_input(
                "End Date",
                datetime.now(),
                help="Historical data end date"
            )

        if selected_stocks:
            # Fetch historical data
            portfolio_data = self._fetch_stock_data(selected_stocks, start_date, end_date)
            
            # Portfolio Allocation Section
            st.header("2. Portfolio Allocation")
            allocations = self._get_allocations(selected_stocks)
            
            # Display current stock prices and values
            self._display_current_holdings(selected_stocks, allocations, investment_amount)

            # Portfolio Analysis Section
            st.header("3. Portfolio Analysis")
            
            # Calculate and display portfolio metrics
            returns, volatility, sharpe_ratio = self._calculate_portfolio_metrics(
                portfolio_data, allocations
            )
            
            # Display metrics in columns
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Expected Annual Return", f"{returns*100:.2f}%")
            with metric_col2:
                st.metric("Annual Volatility", f"{volatility*100:.2f}%")
            with metric_col3:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

            # Portfolio Performance Visualization
            st.subheader("Portfolio Performance Simulation")
            self._plot_portfolio_performance(portfolio_data, allocations, investment_amount)

            # Risk Analysis
            st.subheader("Risk Analysis")
            self._plot_risk_analysis(portfolio_data, allocations)

            # Optimization Suggestions
            st.header("4. Optimization Suggestions")
            self._show_optimization_suggestions(portfolio_data, selected_stocks)

    def _fetch_stock_data(self, stocks, start_date, end_date):
        data = pd.DataFrame()
        with st.spinner('Fetching stock data...'):
            for stock in stocks:
                try:
                    stock_data = yf.download(stock, start=start_date, end=end_date)['Adj Close']
                    data[stock] = stock_data
                except Exception as e:
                    st.error(f"Error fetching data for {stock}: {str(e)}")
        return data

    def _get_allocations(self, stocks):
        st.write("Adjust the allocation percentages (total must equal 100%)")
        cols = st.columns(len(stocks))
        allocations = {}
        total = 0
        
        for i, stock in enumerate(stocks):
            with cols[i]:
                value = st.slider(f"{stock}", 0, 100, int(100/len(stocks)))
                allocations[stock] = value/100
                total += value
        
        if total != 100:
            st.warning(f"Total allocation is {total}%. Please adjust to equal 100%.")
        
        return allocations

    def _display_current_holdings(self, stocks, allocations, investment_amount):
        st.subheader("Current Portfolio Holdings")
        
        holdings_data = []
        for stock in stocks:
            current_price = yf.Ticker(stock).history(period='1d')['Close'].iloc[-1]
            allocation = allocations[stock]
            value = investment_amount * allocation
            shares = value / current_price
            
            holdings_data.append({
                "Stock": stock,
                "Shares": f"{shares:.2f}",
                "Price": f"${current_price:.2f}",
                "Value": f"${value:.2f}",
                "Allocation": f"{allocation*100:.1f}%"
            })
        
        st.table(pd.DataFrame(holdings_data))

    def _calculate_portfolio_metrics(self, data, allocations):
        # Calculate daily returns
        returns = data.pct_change()
        
        # Calculate portfolio returns
        portfolio_returns = returns.dot(list(allocations.values()))
        
        # Calculate annualized metrics
        annual_returns = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_returns - self.risk_free_rate) / annual_volatility
        
        return annual_returns, annual_volatility, sharpe_ratio

    def _plot_portfolio_performance(self, data, allocations, investment_amount):
        # Calculate daily portfolio value
        portfolio_value = (1 + data.pct_change().dot(list(allocations.values()))).cumprod() * investment_amount
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#4CAF50')
        ))
        
        fig.update_layout(
            title='Simulated Portfolio Value Over Time',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)

    def _plot_risk_analysis(self, data, allocations):
        returns = data.pct_change()
        portfolio_returns = returns.dot(list(allocations.values()))
        
        # Create risk visualization
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=portfolio_returns,
            nbinsx=50,
            name='Daily Returns',
            marker_color='#4CAF50'
        ))
        
        fig.update_layout(
            title='Distribution of Daily Returns',
            xaxis_title='Daily Returns',
            yaxis_title='Frequency',
            showlegend=False
        )
        
        st.plotly_chart(fig)

    def _show_optimization_suggestions(self, data, stocks):
        # Calculate correlation matrix
        correlation_matrix = data.pct_change().corr()
        
        # Identify highly correlated pairs
        high_correlation_pairs = []
        for i in range(len(stocks)):
            for j in range(i+1, len(stocks)):
                if abs(correlation_matrix.iloc[i,j]) > 0.7:
                    high_correlation_pairs.append((stocks[i], stocks[j]))
        
        # Display suggestions
        st.write("Based on the analysis, here are some suggestions to optimize your portfolio:")
        
        if high_correlation_pairs:
            st.warning("Consider diversifying these highly correlated pairs:")
            for pair in high_correlation_pairs:
                st.write(f"• {pair[0]} and {pair[1]}")
        
        # Add general recommendations
        st.info("""
        💡 Optimization Tips:
        - Consider adding assets from different sectors for better diversification
        - Review and rebalance your portfolio periodically
        - Monitor and adjust based on your risk tolerance
        - Consider adding bonds or other asset classes for better risk management
        """)