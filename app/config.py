import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        # API Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not self.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
            
        # Stock Configuration
        self.DEFAULT_STOCK_TICKERS = ['AAPL', 'GOOGL', 'NVDA', 'MSFT', 'AMZN', 'TSLA', 'META']
        self.DEFAULT_START_DATE = '2023-01-01'
        self.DEFAULT_END_DATE = None  # Will use current date by default
        
        # Investment Configuration
        self.DEFAULT_INVESTMENT_AMOUNT = 10000.0
        self.DEFAULT_RISK_FREE_RATE = 0.035  # 3.5% risk-free rate
        
        # Technical Analysis Configuration
        self.DEFAULT_MA_PERIODS = [20, 50, 200]  # Moving average periods
        self.DEFAULT_RSI_PERIOD = 14
        self.DEFAULT_BOLLINGER_PERIOD = 20
        
        # Chart Configuration
        self.CHART_THEME = 'plotly_dark'
        self.CHART_HEIGHT = 600
        self.CHART_WIDTH = None
        
        # Time Periods for Analysis
        self.TIME_PERIODS = {
            '1D': '1d',
            '5D': '5d',
            '1M': '1mo',
            '3M': '3mo',
            '6M': '6mo',
            '1Y': '1y',
            '2Y': '2y',
            '5Y': '5y',
            'MAX': 'max'
        }
        
        # Backtesting Configuration
        self.BACKTEST_STRATEGIES = {
            'MA_Crossover': {
                'short_window': 20,
                'long_window': 50
            },
            'RSI': {
                'period': 14,
                'overbought': 70,
                'oversold': 30
            },
            'Bollinger': {
                'period': 20,
                'std_dev': 2
            }
        }
