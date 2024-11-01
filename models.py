from enum import Enum

class StrategySignal(Enum):
    ENTER_LONG = 2
    ENTER_SHORT = 1
    DO_NOTHING = 0
    CLOSE_SHORT = -1
    CLOSE_LONG = -2

class PositionType(Enum):
    LONG = 1
    SHORT = -1
    EMPTY_POSITION = 0

class Position():
    def __init__(self, qty: float, price: float, type: PositionType):
        self.qty = qty
        self.price = price
        self.type = type

class ActionType(Enum):
    BUY = 1
    SELL = -1
    
