"""
Moving Average Indicators: SMA, EMA, WMA and crossover detection
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SMA_PERIODS, EMA_PERIODS

class MovingAverageCalculator:
    def __init__(self):
        self.sma_periods = SMA_PERIODS
        self.ema_periods = EMA_PERIODS
    
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        return prices.rolling(window=period).mean()
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_wma(self, prices: pd.Series, period: int) -> pd.Series:
        weights = np.arange(1, period + 1)
        return prices.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    def detect_golden_cross(self, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        """Golden Cross: Fast MA crosses above Slow MA"""
        return ((fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))).astype(int)
    
    def detect_death_cross(self, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        """Death Cross: Fast MA crosses below Slow MA"""
        return ((fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))).astype(int)
    
    def get_price_vs_ma(self, prices: pd.Series, ma: pd.Series) -> pd.Series:
        """Returns position of price relative to MA (normalized)"""
        return (prices - ma) / ma
    
    def add_ma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df['close']
        
        # SMAs
        for period in self.sma_periods:
            df[f'sma_{period}'] = self.calculate_sma(close, period)
            df[f'close_vs_sma_{period}'] = self.get_price_vs_ma(close, df[f'sma_{period}'])
        
        # EMAs
        for period in self.ema_periods:
            df[f'ema_{period}'] = self.calculate_ema(close, period)
            df[f'close_vs_ema_{period}'] = self.get_price_vs_ma(close, df[f'ema_{period}'])
        
        # Crossovers (9 vs 21 EMA)
        if 9 in self.ema_periods and 21 in self.ema_periods:
            df['ema_golden_cross'] = self.detect_golden_cross(df['ema_9'], df['ema_21'])
            df['ema_death_cross'] = self.detect_death_cross(df['ema_9'], df['ema_21'])
            df['ema_cross_signal'] = df['ema_golden_cross'] - df['ema_death_cross']
        
        # SMA 50 vs 200 (Golden/Death Cross)
        if 50 in self.sma_periods and 200 in self.sma_periods:
            df['sma_golden_cross'] = self.detect_golden_cross(df['sma_50'], df['sma_200'])
            df['sma_death_cross'] = self.detect_death_cross(df['sma_50'], df['sma_200'])
        
        # Trend direction based on EMAs
        if 9 in self.ema_periods and 21 in self.ema_periods:
            df['ma_trend'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
        
        return df
