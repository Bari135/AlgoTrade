import pandas as pd
import numpy as np

def calc_total_return(portfolio_values):
    return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1.0

def calc_annualized_return(portfolio_values):
    yearly_trading_days = 252
    portfolio_trading_days = portfolio_values.shape[0]
    portfolio_trading_years = portfolio_trading_days / yearly_trading_days 
    return (portfolio_values.iloc[-1] / portfolio_values.iloc[0])**(1/portfolio_trading_years) - 1.0

def calc_annualized_sharpe(portfolio_values: pd.Series, rf: float=0.0):
    yearly_trading_days = 252
    annualized_return = calc_annualized_return(portfolio_values)
    annualized_std = portfolio_values.pct_change().std() * np.sqrt(yearly_trading_days)
    if annualized_std is None or annualized_std == 0:
        return 0
    sharpe = (annualized_return - rf) / annualized_std
    return sharpe

def calc_downside_deviation(portfolio_values):
    porfolio_returns = portfolio_values.pct_change().dropna()
    return porfolio_returns[porfolio_returns < 0].std()

def calc_sortino(portfolio_values, rf=0.0):
    yearly_trading_days = 252
    down_deviation = calc_downside_deviation(portfolio_values) * np.sqrt(yearly_trading_days)
    annualized_return = calc_annualized_return(portfolio_values)
    if down_deviation is None or down_deviation == 0:
        return 0
    sortino = (annualized_return - rf) / down_deviation
    return sortino

def calc_max_drawdown(portfolio_values):
    cumulative_max = portfolio_values.cummax()
    drawdown = (cumulative_max - portfolio_values) / cumulative_max
    return drawdown.max()

def calc_calmar(portfolio_values):
    max_drawdown = calc_max_drawdown(portfolio_values)
    annualized_return = calc_annualized_return(portfolio_values)
    return annualized_return / max_drawdown
   

def evaluate_strategy(b_df, strat_name):
    total_return = calc_total_return(b_df['portfolio_value'])
    annualized_return = calc_annualized_return(b_df['portfolio_value'])
    annualized_sharpe = calc_annualized_sharpe(b_df['portfolio_value'])
    sortino_ratio = calc_sortino(b_df['portfolio_value'])
    max_drawdown = calc_max_drawdown(b_df['portfolio_value'])
    calmar_ratio = calc_calmar(b_df['portfolio_value'])
    
    print(f"Results for {strat_name}:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Sharpe Ratio: {annualized_sharpe:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")

def plot_strategy(b_df, strat_name):
    b_df['portfolio_value'].plot(figsize=(12, 8), title=f"{strat_name} Portfolio Value")