import pandas as pd
import numpy as np
import datetime as dt


from models import ActionType, PositionType, Position, StrategySignal
from strategies import BaseStrategy, OurStrategy, XGBoostStrategy
from evaluation import evaluate_strategy, plot_strategy, calc_total_return
from procces_data import prepare_data
from xgModel import XGBoostModel


SLIPPAGE=5.0
COMMISSION=0
    
def calc_realistic_price(row: pd.Series ,action_type: ActionType, slippage_factor=np.inf):
    slippage_rate = ((row['close'] - row['open']) / row['open']) / slippage_factor
    slippage_price = row['open'] + row['open'] * slippage_rate
    
    if action_type == ActionType.BUY:
        return max(slippage_price, row['open'])
    else:
        return min(slippage_price, row['open'])
    
def initialize_data(data:pd.DataFrame):
    data['qty'] = 0.0
    data['balance'] = 0.0
    data['position'] = PositionType.EMPTY_POSITION

def update_QtyAndBalance(data:pd.DataFrame, rowInd, qty, bal):
    data.loc[rowInd, 'qty'] = qty
    data.loc[rowInd, 'balance'] = bal

def carryPrevQtyBalAndPortfolio(data:pd.DataFrame, rowInd, start_balance):
    data['portfolio_value'] = data['close'] * data['qty'] + data['balance']
    return data.loc[rowInd - 1, 'qty'] if rowInd > 0 else 0, data.loc[rowInd - 1, 'balance'] if rowInd > 0 else start_balance


def enter_position(data: pd.DataFrame, index: int, row: pd.Series, curr_qty: float, curr_balance: float, position_type: PositionType) -> Position:
        if position_type == PositionType.LONG:
            buy_price = calc_realistic_price(row, ActionType.BUY, slippage_factor=SLIPPAGE)
            qty_to_buy = strategy.calc_qty(buy_price, curr_balance, ActionType.BUY)
            position = Position(qty_to_buy, buy_price, position_type)
            data.loc[index, 'qty'] = curr_qty + qty_to_buy
            data.loc[index, 'balance'] = curr_balance - qty_to_buy * buy_price - COMMISSION
            
        
        elif position_type == PositionType.SHORT:
            sell_price = calc_realistic_price(row, ActionType.SELL, slippage_factor=SLIPPAGE)
            qty_to_sell = strategy.calc_qty(sell_price, curr_balance, ActionType.SELL)
            position = Position(qty_to_sell, sell_price, position_type)
            data.loc[index, 'qty'] = curr_qty - qty_to_sell
            data.loc[index, 'balance'] = curr_balance + qty_to_sell * sell_price - COMMISSION
        
        return position
    
def close_position(data: pd.DataFrame, index: int, row: pd.Series, curr_qty: float, curr_balance: float, position: Position):
    if position.type == PositionType.LONG:
        sell_price = calc_realistic_price(row, ActionType.SELL, slippage_factor=SLIPPAGE)
        data.loc[index, 'qty'] = curr_qty - position.qty
        data.loc[index, 'balance'] = curr_balance + position.qty * sell_price - COMMISSION

    elif position.type == PositionType.SHORT:
        buy_price = calc_realistic_price(row, ActionType.BUY, slippage_factor=SLIPPAGE)
        data.loc[index, 'qty'] = curr_qty + position.qty
        data.loc[index, 'balance'] = curr_balance - position.qty * buy_price - COMMISSION
    
    data.loc[index, 'position'] = StrategySignal.CLOSE_LONG if position.type == PositionType.LONG else StrategySignal.CLOSE_SHORT
    position_updated=True
    return None, position_updated

def update_PositionStatus(data:pd.DataFrame, index, position:Position):
    if data.loc[index-1, 'position'] == StrategySignal.ENTER_LONG and position:
                data.loc[index, 'position'] = PositionType.LONG
    elif data.loc[index-1, 'position'] == StrategySignal.ENTER_SHORT and position:
        data.loc[index, 'position'] = PositionType.SHORT
    elif ((data.loc[index-1, 'position'] == PositionType.LONG) or
            (data.loc[index-1, 'position'] == PositionType.SHORT)) and position:
        data.loc[index, 'position'] = data.loc[index-1, 'position']
    else:
        if ((data.loc[index, 'position'] != StrategySignal.CLOSE_LONG) and 
            (data.loc[index, 'position'] != StrategySignal.CLOSE_SHORT)):
            data.loc[index, 'position'] = PositionType.EMPTY_POSITION

def enter_newPosition(data:pd.DataFrame, signal:ActionType, index, row, curr_qty, curr_balance):
    position_type = PositionType.LONG if signal == ActionType.BUY else PositionType.SHORT
    if signal==ActionType.BUY:
        data.loc[index, 'position'] = StrategySignal.ENTER_LONG
    elif signal==ActionType.SELL:
        data.loc[index, 'position'] = StrategySignal.ENTER_SHORT
    
    position = enter_position(data, index, row, curr_qty, curr_balance, position_type)
    position_updated=True
    return position_updated, position

def backtest(data, strategy, starting_balance, slippage_factor=5.0, commission=0.0):

    # initialize df 
    initialize_data(data)
    strategy.calc_signal(data)

    position = None
    position_updated=False
    data.reset_index(inplace=True)
    for index, row in data.iterrows():
        position_updated=False
        curr_qty, curr_balance=carryPrevQtyBalAndPortfolio(data, index, starting_balance)
        
        signal = row['strategy_signal']

        # If theres an open postion well check wether it's got a sl or tp.
        if position:
            sl_tp_res = strategy.check_sl_tp(data.iloc[index - 1], position)
            if sl_tp_res is not None:
                sl_tp_qty, sl_tp_price, sl_tp_action = sl_tp_res
                position, position_updated = close_position(data, index, row, curr_qty, curr_balance, position)


            
            # Close existing position on opposite signal using close_position method
            elif ((position.type == PositionType.LONG and signal == ActionType.SELL) or
                (position.type == PositionType.SHORT and signal == ActionType.BUY)):
                
                position, position_updated = close_position(data, index, row, curr_qty, curr_balance, position)
                
            update_PositionStatus(data, index, position)
            
            
        # Enter new position if there's no current position
        elif not position and signal in [ActionType.BUY, ActionType.SELL]:
            position_updated, position= enter_newPosition(data, signal, index, row, curr_qty, curr_balance)
        
        # If no position change occurred, carry forward the qty and balance
        if not position_updated:
            update_QtyAndBalance(data, index, curr_qty, curr_balance)

        if index==len(data)-1:
            position = close_position(data, index, row, curr_qty, curr_balance, position) if position else None

    # Calculate portfolio value
    data['portfolio_value'] = data['close'] * data['qty'] + data['balance']
    return data



if __name__ == '__main__':
    start_balance = 100000
    model= XGBoostModel(symbol='BTC')
    # model.train_model() # if not trained yet
    model.evaluate_model()
    strategy = XGBoostStrategy(sl_rate=0.15, tp_rate=0.18, model=model)
    btc_df = prepare_data('BTC')

    btc_df['open_time'] = pd.to_datetime(btc_df['open_time'])
    df = btc_df[btc_df['open_time'] >= pd.to_datetime('2024-01-01')].copy()
    
    b_df = backtest(df.copy(deep=True), strategy, start_balance)
    evaluate_strategy(b_df, 'XGboost Strategy')

    b_df.to_csv('backtesting_results.csv')
    
