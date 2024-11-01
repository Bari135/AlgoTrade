import pandas as pd
from models import ActionType, Position, PositionType, StrategySignal
from abc import ABC, abstractmethod
from typing import Tuple
from indicators import UTBot, STC, UPPER_STC, LOWER_STC
from xgModel import XGBoostModel

class BaseStrategy(ABC):
    def __init__(self, sl_rate: float=None, tp_rate: float=None) -> None:
        super().__init__()
        self.sl_rate = sl_rate
        self.tp_rate = tp_rate
    
    @abstractmethod
    def calc_signal(self, data: pd.DataFrame):
        pass

    def calc_qty(self, real_price: float, balance: float, action: ActionType, **kwargs) -> float:
        # if action == ActionType.BUY:
        #     qty = balance / real_price
        
        # elif action == ActionType.SELL:
        #     qty =  balance / real_price

        qty= balance // real_price
        return qty    
    
    def check_sl_tp(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        sl_res = self.is_stop_loss(row, position)
        if sl_res is not None:
            return sl_res
        
        tp_res = self.is_take_profit(row, position)
        if tp_res is not None:
            return tp_res
    
    def is_stop_loss(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        """
        Checks if the price has hit the stop-loss level.
        
        Returns:
            Tuple[float, float, ActionType] or None: If stop-loss is triggered, returns a tuple containing quantity and stop-loss price and action type, otherwise returns None.
        """
        if self.sl_rate is not None:
            long_stop_loss_price = position.price * (1 - self.sl_rate)
            if position.type == PositionType.LONG and row['low'] <= long_stop_loss_price:
                return position.qty, long_stop_loss_price, ActionType.SELL
            
            short_stop_loss_price = position.price * (1 + self.sl_rate)
            if position.type == PositionType.SHORT and row['high'] >= short_stop_loss_price:
                return position.qty, short_stop_loss_price, ActionType.BUY
    
    def is_take_profit(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        """
        Checks if the price has hit the take-profit level.

        Returns:
            Tuple[float, float, ActionType] or None: If take-profit is triggered, returns a tuple containing quantity and take-profit price and action type, otherwise returns None.
        """
        if self.tp_rate is not None:
            long_take_profit_price = position.price * (1 + self.tp_rate)
            if position.type == PositionType.LONG and row['high'] >= long_take_profit_price:
                return position.qty, long_take_profit_price, ActionType.SELL
            
            short_take_profit_price = position.price * (1 - self.tp_rate)
            if position.type == PositionType.SHORT and row['low'] <= short_take_profit_price:
                return position.qty, short_take_profit_price, ActionType.BUY



class XGBoostStrategy(BaseStrategy):
    def __init__(self, sl_rate: float = None, tp_rate: float = None, model = None) -> pd.Series:
        super().__init__(sl_rate, tp_rate)
        self.model = model

    def calc_signal(self, data: pd.DataFrame):
        predictions = self.model.predict(data)
        data['strategy_signal'] = [StrategySignal.DO_NOTHING if pred == 1 else ActionType.BUY if pred == 0 else ActionType.SELL for pred in predictions]

        


