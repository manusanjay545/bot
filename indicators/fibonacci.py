"""
Fibonacci Retracement and Extension Calculator
Automatically detects swing highs/lows and calculates Fibonacci levels
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIBONACCI_RETRACEMENT, FIBONACCI_EXTENSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FibonacciLevels:
    """Container for Fibonacci levels"""
    swing_high: float
    swing_low: float
    trend: str  # 'up' or 'down'
    retracement_levels: Dict[float, float]  # ratio -> price level
    extension_levels: Dict[float, float]
    
    def get_nearest_level(self, price: float) -> Tuple[float, float]:
        """Get the nearest Fibonacci level to a given price"""
        all_levels = {**self.retracement_levels, **self.extension_levels}
        nearest_ratio = min(all_levels.keys(), key=lambda k: abs(all_levels[k] - price))
        return nearest_ratio, all_levels[nearest_ratio]


class FibonacciCalculator:
    """
    Calculates Fibonacci retracement and extension levels
    based on automatic swing high/low detection
    """
    
    def __init__(
        self,
        retracement_levels: List[float] = None,
        extension_levels: List[float] = None,
        swing_lookback: int = 10
    ):
        """
        Initialize the Fibonacci calculator
        
        Args:
            retracement_levels: List of retracement ratios
            extension_levels: List of extension ratios
            swing_lookback: Number of candles to look back for swing detection
        """
        self.retracement_levels = retracement_levels or FIBONACCI_RETRACEMENT
        self.extension_levels = extension_levels or FIBONACCI_EXTENSION
        self.swing_lookback = swing_lookback
        
    def detect_swing_points(
        self,
        df: pd.DataFrame,
        lookback: int = None
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Detect swing high and swing low points
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of candles for swing detection
            
        Returns:
            Tuple of (swing_highs, swing_lows) as list of (index, price) tuples
        """
        if lookback is None:
            lookback = self.swing_lookback
            
        highs = df['high'].values
        lows = df['low'].values
        
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(df) - lookback):
            # Check for swing high
            if highs[i] == max(highs[i-lookback:i+lookback+1]):
                swing_highs.append((i, highs[i]))
                
            # Check for swing low
            if lows[i] == min(lows[i-lookback:i+lookback+1]):
                swing_lows.append((i, lows[i]))
                
        return swing_highs, swing_lows
    
    def get_recent_swing_points(
        self,
        df: pd.DataFrame,
        n_points: int = 2
    ) -> Tuple[Optional[Tuple[int, float]], Optional[Tuple[int, float]]]:
        """
        Get the most recent swing high and swing low
        
        Args:
            df: DataFrame with OHLCV data
            n_points: Number of recent points to consider
            
        Returns:
            Tuple of (most_recent_swing_high, most_recent_swing_low)
        """
        swing_highs, swing_lows = self.detect_swing_points(df)
        
        recent_high = swing_highs[-1] if swing_highs else None
        recent_low = swing_lows[-1] if swing_lows else None
        
        return recent_high, recent_low
    
    def calculate_retracement_levels(
        self,
        swing_high: float,
        swing_low: float,
        trend: str = 'up'
    ) -> Dict[float, float]:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            swing_high: The swing high price
            swing_low: The swing low price
            trend: 'up' or 'down' to determine direction
            
        Returns:
            Dictionary mapping ratio to price level
        """
        price_range = swing_high - swing_low
        levels = {}
        
        for ratio in self.retracement_levels:
            if trend == 'up':
                # In uptrend, retracements are from high going down
                levels[ratio] = swing_high - (price_range * ratio)
            else:
                # In downtrend, retracements are from low going up
                levels[ratio] = swing_low + (price_range * ratio)
                
        return levels
    
    def calculate_extension_levels(
        self,
        swing_high: float,
        swing_low: float,
        trend: str = 'up'
    ) -> Dict[float, float]:
        """
        Calculate Fibonacci extension levels
        
        Args:
            swing_high: The swing high price
            swing_low: The swing low price
            trend: 'up' or 'down' to determine direction
            
        Returns:
            Dictionary mapping ratio to price level
        """
        price_range = swing_high - swing_low
        levels = {}
        
        for ratio in self.extension_levels:
            if trend == 'up':
                # Extensions above the swing high
                levels[ratio] = swing_low + (price_range * ratio)
            else:
                # Extensions below the swing low
                levels[ratio] = swing_high - (price_range * ratio)
                
        return levels
    
    def calculate_fibonacci_levels(
        self,
        df: pd.DataFrame,
        use_recent: bool = True
    ) -> Optional[FibonacciLevels]:
        """
        Calculate complete Fibonacci levels for the current price action
        
        Args:
            df: DataFrame with OHLCV data
            use_recent: Whether to use the most recent swing points
            
        Returns:
            FibonacciLevels object or None if insufficient data
        """
        recent_high, recent_low = self.get_recent_swing_points(df)
        
        if recent_high is None or recent_low is None:
            return None
            
        high_idx, swing_high = recent_high
        low_idx, swing_low = recent_low
        
        # Determine trend based on which swing point is more recent
        if high_idx > low_idx:
            trend = 'up'
        else:
            trend = 'down'
            
        retracement = self.calculate_retracement_levels(swing_high, swing_low, trend)
        extension = self.calculate_extension_levels(swing_high, swing_low, trend)
        
        return FibonacciLevels(
            swing_high=swing_high,
            swing_low=swing_low,
            trend=trend,
            retracement_levels=retracement,
            extension_levels=extension
        )
    
    def add_fibonacci_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Fibonacci-based features to the DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional Fibonacci features
        """
        df = df.copy()
        
        # Calculate for each point using a rolling window
        fib_features = {
            'fib_trend': [],
            'distance_to_fib_0': [],
            'distance_to_fib_236': [],
            'distance_to_fib_382': [],
            'distance_to_fib_5': [],
            'distance_to_fib_618': [],
            'distance_to_fib_786': [],
            'distance_to_fib_1': [],
            'nearest_fib_level': [],
            'nearest_fib_distance': []
        }
        
        min_window = self.swing_lookback * 3
        
        for i in range(len(df)):
            if i < min_window:
                for key in fib_features:
                    fib_features[key].append(np.nan)
                continue
                
            window_df = df.iloc[:i+1].tail(100)  # Use last 100 candles for calculation
            fib_levels = self.calculate_fibonacci_levels(window_df)
            
            if fib_levels is None:
                for key in fib_features:
                    fib_features[key].append(np.nan)
                continue
                
            current_price = df['close'].iloc[i]
            price_range = fib_levels.swing_high - fib_levels.swing_low
            
            if price_range == 0:
                for key in fib_features:
                    fib_features[key].append(np.nan)
                continue
            
            fib_features['fib_trend'].append(1 if fib_levels.trend == 'up' else -1)
            
            # Distance to each retracement level (normalized)
            fib_features['distance_to_fib_0'].append(
                (current_price - fib_levels.retracement_levels[0]) / price_range
            )
            fib_features['distance_to_fib_236'].append(
                (current_price - fib_levels.retracement_levels[0.236]) / price_range
            )
            fib_features['distance_to_fib_382'].append(
                (current_price - fib_levels.retracement_levels[0.382]) / price_range
            )
            fib_features['distance_to_fib_5'].append(
                (current_price - fib_levels.retracement_levels[0.5]) / price_range
            )
            fib_features['distance_to_fib_618'].append(
                (current_price - fib_levels.retracement_levels[0.618]) / price_range
            )
            fib_features['distance_to_fib_786'].append(
                (current_price - fib_levels.retracement_levels[0.786]) / price_range
            )
            fib_features['distance_to_fib_1'].append(
                (current_price - fib_levels.retracement_levels[1.0]) / price_range
            )
            
            # Nearest Fibonacci level
            nearest_ratio, nearest_price = fib_levels.get_nearest_level(current_price)
            fib_features['nearest_fib_level'].append(nearest_ratio)
            fib_features['nearest_fib_distance'].append(
                (current_price - nearest_price) / price_range
            )
        
        for key, values in fib_features.items():
            df[key] = values
            
        return df
    
    def get_support_resistance_from_fib(
        self,
        fib_levels: FibonacciLevels,
        current_price: float
    ) -> Tuple[float, float]:
        """
        Get nearby support and resistance levels from Fibonacci
        
        Args:
            fib_levels: Calculated Fibonacci levels
            current_price: Current market price
            
        Returns:
            Tuple of (support, resistance) prices
        """
        all_levels = sorted(list(fib_levels.retracement_levels.values()) + 
                          list(fib_levels.extension_levels.values()))
        
        support = None
        resistance = None
        
        for level in all_levels:
            if level < current_price:
                support = level
            elif level > current_price and resistance is None:
                resistance = level
                break
                
        return support, resistance


if __name__ == "__main__":
    # Test the Fibonacci calculator
    from data.fetcher import DataFetcher
    
    fetcher = DataFetcher()
    df = fetcher.fetch_historical_data('NIFTY', days=30)
    
    fib_calc = FibonacciCalculator()
    
    # Calculate Fibonacci levels
    fib_levels = fib_calc.calculate_fibonacci_levels(df)
    
    if fib_levels:
        print(f"Swing High: {fib_levels.swing_high}")
        print(f"Swing Low: {fib_levels.swing_low}")
        print(f"Trend: {fib_levels.trend}")
        print("\nRetracement Levels:")
        for ratio, price in fib_levels.retracement_levels.items():
            print(f"  {ratio*100:.1f}%: {price:.2f}")
        print("\nExtension Levels:")
        for ratio, price in fib_levels.extension_levels.items():
            print(f"  {ratio*100:.1f}%: {price:.2f}")
            
        # Get support/resistance
        current_price = df['close'].iloc[-1]
        support, resistance = fib_calc.get_support_resistance_from_fib(fib_levels, current_price)
        print(f"\nCurrent Price: {current_price:.2f}")
        print(f"Support: {support:.2f}")
        print(f"Resistance: {resistance:.2f}")
