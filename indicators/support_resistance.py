"""
Support and Resistance Level Detection
"""
import pandas as pd
import numpy as np
from typing import List, Tuple

class SupportResistanceCalculator:
    def __init__(self, lookback: int = 20, threshold: float = 0.002):
        self.lookback = lookback
        self.threshold = threshold  # Price clustering threshold
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> dict:
        """Calculate standard pivot points from previous day's data"""
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {'pivot': pivot, 'r1': r1, 'r2': r2, 'r3': r3, 's1': s1, 's2': s2, 's3': s3}
    
    def calculate_fibonacci_pivots(self, df: pd.DataFrame) -> dict:
        """Calculate Fibonacci pivot points"""
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        diff = high - low
        
        return {
            'pivot': pivot,
            'r1': pivot + 0.382 * diff,
            'r2': pivot + 0.618 * diff,
            'r3': pivot + diff,
            's1': pivot - 0.382 * diff,
            's2': pivot - 0.618 * diff,
            's3': pivot - diff
        }
    
    def find_swing_levels(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Find swing high and swing low levels"""
        highs = df['high'].values
        lows = df['low'].values
        
        resistances = []
        supports = []
        
        for i in range(self.lookback, len(df) - self.lookback):
            if highs[i] == max(highs[i-self.lookback:i+self.lookback+1]):
                resistances.append(highs[i])
            if lows[i] == min(lows[i-self.lookback:i+self.lookback+1]):
                supports.append(lows[i])
        
        return supports, resistances
    
    def cluster_levels(self, levels: List[float]) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
        levels = sorted(levels)
        clustered = [levels[0]]
        
        for level in levels[1:]:
            if (level - clustered[-1]) / clustered[-1] > self.threshold:
                clustered.append(level)
            else:
                clustered[-1] = (clustered[-1] + level) / 2
        
        return clustered
    
    def get_nearest_sr(self, price: float, supports: List[float], resistances: List[float]) -> Tuple[float, float]:
        """Get nearest support and resistance to current price"""
        support = max([s for s in supports if s < price], default=None)
        resistance = min([r for r in resistances if r > price], default=None)
        return support, resistance
    
    def add_sr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        supports, resistances = self.find_swing_levels(df)
        supports = self.cluster_levels(supports)
        resistances = self.cluster_levels(resistances)
        
        # Distance to nearest S/R
        df['nearest_support'] = np.nan
        df['nearest_resistance'] = np.nan
        df['dist_to_support'] = np.nan
        df['dist_to_resistance'] = np.nan
        
        for i in range(len(df)):
            price = df['close'].iloc[i]
            support, resistance = self.get_nearest_sr(price, supports, resistances)
            
            if support:
                df.iloc[i, df.columns.get_loc('nearest_support')] = support
                df.iloc[i, df.columns.get_loc('dist_to_support')] = (price - support) / price
            if resistance:
                df.iloc[i, df.columns.get_loc('nearest_resistance')] = resistance
                df.iloc[i, df.columns.get_loc('dist_to_resistance')] = (resistance - price) / price
        
        # Pivot points for latest data
        pivots = self.calculate_pivot_points(df)
        for key, value in pivots.items():
            df[f'pivot_{key}'] = value
        
        return df
