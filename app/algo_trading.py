import pandas as pd
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def calculate_risk_metrics(df):
    returns = df['Pnl_Percentage'] / 100
    risk_free_rate = 0.05 / 252
    excess_returns = returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
    sortino_ratio = excess_returns.mean() / returns[returns < 0].std() * np.sqrt(252)
    return sharpe_ratio, sortino_ratio

def calculate_max_drawdown(data):
    cumulative_profit = data['Profit'].cumsum()
    peak = cumulative_profit.cummax()
    drawdown = peak - cumulative_profit
    max_drawdown = drawdown.max()
    max_drawdown_idx = drawdown.idxmax()
    peak_value = peak.loc[max_drawdown_idx]
    max_drawdown_percentage = (max_drawdown / peak_value) * 100 if peak_value != 0 else 0
    drawdown_start = (peak != cumulative_profit).idxmax()
    drawdown_end = max_drawdown_idx
    max_drawdown_duration = data.loc[drawdown_end, 'Date'] - data.loc[drawdown_start, 'Date']
    return max_drawdown, max_drawdown_percentage, max_drawdown_duration
