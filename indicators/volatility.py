"""
Volatility Indicators: Bollinger Bands, ATR, Historical Volatility
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BOLLINGER_PERIOD, BOLLINGER_STD, ATR_PERIOD

class VolatilityCalculator:
    def __init__(self):
        self.bb_period = BOLLINGER_PERIOD
        self.bb_std = BOLLINGER_STD
        self.atr_period = ATR_PERIOD
    
    def calculate_bollinger_bands(self, prices: pd.Series) -> tuple:
        sma = prices.rolling(window=self.bb_period).mean()
        std = prices.rolling(window=self.bb_period).std()
        upper = sma + (std * self.bb_std)
        lower = sma - (std * self.bb_std)
        return upper, sma, lower
    
    def calculate_bb_position(self, prices: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
        """Position within Bollinger Bands (0 = lower, 1 = upper)"""
        return (prices - lower) / (upper - lower + 1e-10)
    
    def calculate_bb_width(self, upper: pd.Series, lower: pd.Series, middle: pd.Series) -> pd.Series:
        """Bollinger Band Width"""
        return (upper - lower) / middle
    
    def detect_bb_squeeze(self, bb_width: pd.Series, lookback: int = 20) -> pd.Series:
        """Detect Bollinger Band squeeze (low volatility)"""
        min_width = bb_width.rolling(window=lookback).min()
        return (bb_width <= min_width * 1.1).astype(int)
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_period).mean()
    
    def calculate_atr_percent(self, df: pd.DataFrame) -> pd.Series:
        """ATR as percentage of price"""
        atr = self.calculate_atr(df)
        return atr / df['close'] * 100
    
    def calculate_historical_volatility(self, prices: pd.Series, period: int = 20) -> pd.Series:
        log_returns = np.log(prices / prices.shift(1))
        return log_returns.rolling(window=period).std() * np.sqrt(252)
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_position'] = self.calculate_bb_position(df['close'], upper, lower)
        df['bb_width'] = self.calculate_bb_width(upper, lower, middle)
        df['bb_squeeze'] = self.detect_bb_squeeze(df['bb_width'])
        
        # ATR
        df['atr'] = self.calculate_atr(df)
        df['atr_percent'] = self.calculate_atr_percent(df)
        
        # Historical Volatility
        df['hist_volatility'] = self.calculate_historical_volatility(df['close'])
        
        # Volatility regime (high/low based on percentile)
        vol_pct = df['atr_percent'].rolling(window=50).rank(pct=True)
        df['volatility_regime'] = np.where(vol_pct > 0.7, 1, np.where(vol_pct < 0.3, -1, 0))
        
        return df
