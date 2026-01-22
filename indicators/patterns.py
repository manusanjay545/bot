"""
Candlestick Pattern Recognition
Identifies bullish, bearish, and neutral candlestick patterns
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternType(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


@dataclass
class PatternSignal:
    """Container for pattern detection result"""
    name: str
    type: PatternType
    strength: float  # 0.0 to 1.0
    description: str


class PatternRecognizer:
    """
    Recognizes candlestick patterns in price data
    """
    
    def __init__(self, body_threshold: float = 0.1, shadow_threshold: float = 0.5):
        """
        Initialize the pattern recognizer
        
        Args:
            body_threshold: Minimum body size as ratio of range for significant body
            shadow_threshold: Threshold for shadow comparisons
        """
        self.body_threshold = body_threshold
        self.shadow_threshold = shadow_threshold
        
    def _get_candle_metrics(self, row: pd.Series) -> Dict:
        """Calculate metrics for a single candle"""
        open_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']
        
        body = abs(close - open_price)
        range_hl = high - low
        
        if range_hl == 0:
            range_hl = 0.0001  # Avoid division by zero
            
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        return {
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'body': body,
            'range': range_hl,
            'body_ratio': body / range_hl,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'upper_shadow_ratio': upper_shadow / range_hl,
            'lower_shadow_ratio': lower_shadow / range_hl,
            'is_bullish': close > open_price,
            'is_bearish': close < open_price,
            'is_doji': body / range_hl < 0.1
        }
    
    # ==================== Single Candle Patterns ====================
    
    def is_doji(self, row: pd.Series) -> Optional[PatternSignal]:
        """Detect Doji pattern"""
        metrics = self._get_candle_metrics(row)
        
        if metrics['is_doji']:
            return PatternSignal(
                name="Doji",
                type=PatternType.NEUTRAL,
                strength=0.6,
                description="Indecision candle - potential reversal"
            )
        return None
    
    def is_hammer(self, row: pd.Series) -> Optional[PatternSignal]:
        """Detect Hammer pattern (bullish reversal)"""
        metrics = self._get_candle_metrics(row)
        
        # Hammer: Small body at top, long lower shadow, little/no upper shadow
        if (metrics['lower_shadow_ratio'] > 0.6 and
            metrics['upper_shadow_ratio'] < 0.1 and
            metrics['body_ratio'] < 0.3):
            return PatternSignal(
                name="Hammer",
                type=PatternType.BULLISH,
                strength=0.7,
                description="Bullish reversal pattern"
            )
        return None
    
    def is_inverted_hammer(self, row: pd.Series) -> Optional[PatternSignal]:
        """Detect Inverted Hammer pattern (bullish reversal)"""
        metrics = self._get_candle_metrics(row)
        
        if (metrics['upper_shadow_ratio'] > 0.6 and
            metrics['lower_shadow_ratio'] < 0.1 and
            metrics['body_ratio'] < 0.3):
            return PatternSignal(
                name="Inverted Hammer",
                type=PatternType.BULLISH,
                strength=0.6,
                description="Potential bullish reversal"
            )
        return None
    
    def is_shooting_star(self, row: pd.Series) -> Optional[PatternSignal]:
        """Detect Shooting Star pattern (bearish reversal)"""
        metrics = self._get_candle_metrics(row)
        
        if (metrics['upper_shadow_ratio'] > 0.6 and
            metrics['lower_shadow_ratio'] < 0.1 and
            metrics['body_ratio'] < 0.3):
            return PatternSignal(
                name="Shooting Star",
                type=PatternType.BEARISH,
                strength=0.7,
                description="Bearish reversal pattern"
            )
        return None
    
    def is_hanging_man(self, row: pd.Series) -> Optional[PatternSignal]:
        """Detect Hanging Man pattern (bearish reversal)"""
        metrics = self._get_candle_metrics(row)
        
        if (metrics['lower_shadow_ratio'] > 0.6 and
            metrics['upper_shadow_ratio'] < 0.1 and
            metrics['body_ratio'] < 0.3):
            return PatternSignal(
                name="Hanging Man",
                type=PatternType.BEARISH,
                strength=0.6,
                description="Potential bearish reversal"
            )
        return None
    
    def is_marubozu(self, row: pd.Series) -> Optional[PatternSignal]:
        """Detect Marubozu pattern (strong trend continuation)"""
        metrics = self._get_candle_metrics(row)
        
        if (metrics['body_ratio'] > 0.9 and
            metrics['upper_shadow_ratio'] < 0.05 and
            metrics['lower_shadow_ratio'] < 0.05):
            pattern_type = PatternType.BULLISH if metrics['is_bullish'] else PatternType.BEARISH
            return PatternSignal(
                name="Marubozu",
                type=pattern_type,
                strength=0.8,
                description="Strong momentum candle"
            )
        return None
    
    def is_spinning_top(self, row: pd.Series) -> Optional[PatternSignal]:
        """Detect Spinning Top pattern"""
        metrics = self._get_candle_metrics(row)
        
        if (metrics['body_ratio'] < 0.3 and
            metrics['upper_shadow_ratio'] > 0.2 and
            metrics['lower_shadow_ratio'] > 0.2):
            return PatternSignal(
                name="Spinning Top",
                type=PatternType.NEUTRAL,
                strength=0.5,
                description="Indecision - possible reversal"
            )
        return None
    
    # ==================== Multi-Candle Patterns ====================
    
    def is_bullish_engulfing(self, df: pd.DataFrame, idx: int) -> Optional[PatternSignal]:
        """Detect Bullish Engulfing pattern"""
        if idx < 1:
            return None
            
        prev = self._get_candle_metrics(df.iloc[idx - 1])
        curr = self._get_candle_metrics(df.iloc[idx])
        
        if (prev['is_bearish'] and curr['is_bullish'] and
            curr['close'] > prev['open'] and
            curr['open'] < prev['close'] and
            curr['body'] > prev['body']):
            return PatternSignal(
                name="Bullish Engulfing",
                type=PatternType.BULLISH,
                strength=0.8,
                description="Strong bullish reversal"
            )
        return None
    
    def is_bearish_engulfing(self, df: pd.DataFrame, idx: int) -> Optional[PatternSignal]:
        """Detect Bearish Engulfing pattern"""
        if idx < 1:
            return None
            
        prev = self._get_candle_metrics(df.iloc[idx - 1])
        curr = self._get_candle_metrics(df.iloc[idx])
        
        if (prev['is_bullish'] and curr['is_bearish'] and
            curr['close'] < prev['open'] and
            curr['open'] > prev['close'] and
            curr['body'] > prev['body']):
            return PatternSignal(
                name="Bearish Engulfing",
                type=PatternType.BEARISH,
                strength=0.8,
                description="Strong bearish reversal"
            )
        return None
    
    def is_piercing_line(self, df: pd.DataFrame, idx: int) -> Optional[PatternSignal]:
        """Detect Piercing Line pattern (bullish)"""
        if idx < 1:
            return None
            
        prev = self._get_candle_metrics(df.iloc[idx - 1])
        curr = self._get_candle_metrics(df.iloc[idx])
        
        mid_prev = (prev['open'] + prev['close']) / 2
        
        if (prev['is_bearish'] and curr['is_bullish'] and
            curr['open'] < prev['low'] and
            curr['close'] > mid_prev and
            curr['close'] < prev['open']):
            return PatternSignal(
                name="Piercing Line",
                type=PatternType.BULLISH,
                strength=0.7,
                description="Bullish reversal pattern"
            )
        return None
    
    def is_dark_cloud_cover(self, df: pd.DataFrame, idx: int) -> Optional[PatternSignal]:
        """Detect Dark Cloud Cover pattern (bearish)"""
        if idx < 1:
            return None
            
        prev = self._get_candle_metrics(df.iloc[idx - 1])
        curr = self._get_candle_metrics(df.iloc[idx])
        
        mid_prev = (prev['open'] + prev['close']) / 2
        
        if (prev['is_bullish'] and curr['is_bearish'] and
            curr['open'] > prev['high'] and
            curr['close'] < mid_prev and
            curr['close'] > prev['open']):
            return PatternSignal(
                name="Dark Cloud Cover",
                type=PatternType.BEARISH,
                strength=0.7,
                description="Bearish reversal pattern"
            )
        return None
    
    def is_morning_star(self, df: pd.DataFrame, idx: int) -> Optional[PatternSignal]:
        """Detect Morning Star pattern (bullish)"""
        if idx < 2:
            return None
            
        first = self._get_candle_metrics(df.iloc[idx - 2])
        second = self._get_candle_metrics(df.iloc[idx - 1])
        third = self._get_candle_metrics(df.iloc[idx])
        
        if (first['is_bearish'] and first['body_ratio'] > 0.5 and
            second['body_ratio'] < 0.3 and
            third['is_bullish'] and third['body_ratio'] > 0.5 and
            second['close'] < first['close'] and
            third['close'] > (first['open'] + first['close']) / 2):
            return PatternSignal(
                name="Morning Star",
                type=PatternType.BULLISH,
                strength=0.85,
                description="Strong bullish reversal"
            )
        return None
    
    def is_evening_star(self, df: pd.DataFrame, idx: int) -> Optional[PatternSignal]:
        """Detect Evening Star pattern (bearish)"""
        if idx < 2:
            return None
            
        first = self._get_candle_metrics(df.iloc[idx - 2])
        second = self._get_candle_metrics(df.iloc[idx - 1])
        third = self._get_candle_metrics(df.iloc[idx])
        
        if (first['is_bullish'] and first['body_ratio'] > 0.5 and
            second['body_ratio'] < 0.3 and
            third['is_bearish'] and third['body_ratio'] > 0.5 and
            second['close'] > first['close'] and
            third['close'] < (first['open'] + first['close']) / 2):
            return PatternSignal(
                name="Evening Star",
                type=PatternType.BEARISH,
                strength=0.85,
                description="Strong bearish reversal"
            )
        return None
    
    def is_three_white_soldiers(self, df: pd.DataFrame, idx: int) -> Optional[PatternSignal]:
        """Detect Three White Soldiers pattern (bullish)"""
        if idx < 2:
            return None
            
        candles = [self._get_candle_metrics(df.iloc[idx - i]) for i in range(2, -1, -1)]
        
        if all(c['is_bullish'] and c['body_ratio'] > 0.5 for c in candles):
            if (candles[1]['close'] > candles[0]['close'] and
                candles[2]['close'] > candles[1]['close'] and
                candles[1]['open'] > candles[0]['open'] and
                candles[2]['open'] > candles[1]['open']):
                return PatternSignal(
                    name="Three White Soldiers",
                    type=PatternType.BULLISH,
                    strength=0.9,
                    description="Very strong bullish continuation"
                )
        return None
    
    def is_three_black_crows(self, df: pd.DataFrame, idx: int) -> Optional[PatternSignal]:
        """Detect Three Black Crows pattern (bearish)"""
        if idx < 2:
            return None
            
        candles = [self._get_candle_metrics(df.iloc[idx - i]) for i in range(2, -1, -1)]
        
        if all(c['is_bearish'] and c['body_ratio'] > 0.5 for c in candles):
            if (candles[1]['close'] < candles[0]['close'] and
                candles[2]['close'] < candles[1]['close'] and
                candles[1]['open'] < candles[0]['open'] and
                candles[2]['open'] < candles[1]['open']):
                return PatternSignal(
                    name="Three Black Crows",
                    type=PatternType.BEARISH,
                    strength=0.9,
                    description="Very strong bearish continuation"
                )
        return None
    
    def detect_all_patterns(self, df: pd.DataFrame, idx: int = -1) -> List[PatternSignal]:
        """
        Detect all patterns at a given index
        
        Args:
            df: DataFrame with OHLCV data
            idx: Index to check (default: last candle)
            
        Returns:
            List of detected patterns
        """
        if idx < 0:
            idx = len(df) + idx
            
        patterns = []
        
        # Single candle patterns
        row = df.iloc[idx]
        single_patterns = [
            self.is_doji(row),
            self.is_hammer(row),
            self.is_inverted_hammer(row),
            self.is_shooting_star(row),
            self.is_hanging_man(row),
            self.is_marubozu(row),
            self.is_spinning_top(row)
        ]
        
        # Multi-candle patterns
        multi_patterns = [
            self.is_bullish_engulfing(df, idx),
            self.is_bearish_engulfing(df, idx),
            self.is_piercing_line(df, idx),
            self.is_dark_cloud_cover(df, idx),
            self.is_morning_star(df, idx),
            self.is_evening_star(df, idx),
            self.is_three_white_soldiers(df, idx),
            self.is_three_black_crows(df, idx)
        ]
        
        all_patterns = single_patterns + multi_patterns
        patterns = [p for p in all_patterns if p is not None]
        
        return patterns
    
    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pattern-based features to the DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with pattern features
        """
        df = df.copy()
        
        # Initialize pattern columns
        df['pattern_bullish_signal'] = 0.0
        df['pattern_bearish_signal'] = 0.0
        df['pattern_neutral_signal'] = 0.0
        df['pattern_strength'] = 0.0
        df['pattern_count'] = 0
        
        for i in range(len(df)):
            patterns = self.detect_all_patterns(df, i)
            
            if patterns:
                bullish_strength = sum(p.strength for p in patterns if p.type == PatternType.BULLISH)
                bearish_strength = sum(p.strength for p in patterns if p.type == PatternType.BEARISH)
                neutral_strength = sum(p.strength for p in patterns if p.type == PatternType.NEUTRAL)
                
                df.iloc[i, df.columns.get_loc('pattern_bullish_signal')] = bullish_strength
                df.iloc[i, df.columns.get_loc('pattern_bearish_signal')] = bearish_strength
                df.iloc[i, df.columns.get_loc('pattern_neutral_signal')] = neutral_strength
                df.iloc[i, df.columns.get_loc('pattern_strength')] = max(bullish_strength, bearish_strength)
                df.iloc[i, df.columns.get_loc('pattern_count')] = len(patterns)
        
        # Net pattern signal
        df['pattern_net_signal'] = df['pattern_bullish_signal'] - df['pattern_bearish_signal']
        
        return df


if __name__ == "__main__":
    # Test the pattern recognizer
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fetcher import DataFetcher
    
    fetcher = DataFetcher()
    df = fetcher.fetch_historical_data('NIFTY', days=30)
    
    recognizer = PatternRecognizer()
    
    # Detect patterns on last candle
    patterns = recognizer.detect_all_patterns(df)
    
    print("Detected Patterns on last candle:")
    for pattern in patterns:
        print(f"  - {pattern.name} ({pattern.type.name}): {pattern.strength:.2f}")
        print(f"    {pattern.description}")
    
    # Add pattern features
    df_with_patterns = recognizer.add_pattern_features(df)
    print(f"\nPattern features added. Shape: {df_with_patterns.shape}")
    print(df_with_patterns[['pattern_bullish_signal', 'pattern_bearish_signal', 'pattern_net_signal']].tail(10))
