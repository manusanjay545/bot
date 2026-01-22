"""
Oscillator Indicators: RSI, MACD, Stochastic
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, STOCH_K, STOCH_D

@dataclass
class MACDResult:
    macd_line: pd.Series
    signal_line: pd.Series
    histogram: pd.Series

@dataclass
class StochasticResult:
    k_line: pd.Series
    d_line: pd.Series

class OscillatorCalculator:
    def __init__(self):
        self.rsi_period = RSI_PERIOD
        self.rsi_overbought = RSI_OVERBOUGHT
        self.rsi_oversold = RSI_OVERSOLD
        self.macd_fast = MACD_FAST
        self.macd_slow = MACD_SLOW
        self.macd_signal = MACD_SIGNAL
        self.stoch_k = STOCH_K
        self.stoch_d = STOCH_D
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        if period is None: period = self.rsi_period
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)
        avg_gains = gains.ewm(span=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, adjust=False).mean()
        rs = avg_gains / (avg_losses + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def get_rsi_signals(self, rsi: pd.Series) -> pd.Series:
        signals = pd.Series(index=rsi.index, data=0)
        signals[rsi < self.rsi_oversold] = 1
        signals[rsi > self.rsi_overbought] = -1
        return signals
    
    def calculate_macd(self, prices: pd.Series) -> MACDResult:
        ema_fast = prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return MACDResult(macd_line, signal_line, histogram)
    
    def get_macd_signals(self, macd_result: MACDResult) -> pd.Series:
        macd, signal = macd_result.macd_line, macd_result.signal_line
        signals = pd.Series(index=macd.index, data=0)
        signals[(macd > signal) & (macd.shift(1) <= signal.shift(1))] = 1
        signals[(macd < signal) & (macd.shift(1) >= signal.shift(1))] = -1
        return signals
    
    def calculate_stochastic(self, df: pd.DataFrame) -> StochasticResult:
        lowest_low = df['low'].rolling(window=self.stoch_k).min()
        highest_high = df['high'].rolling(window=self.stoch_k).max()
        k_line = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
        d_line = k_line.rolling(window=self.stoch_d).mean()
        return StochasticResult(k_line, d_line)
    
    def get_stochastic_signals(self, stoch: StochasticResult) -> pd.Series:
        k, d = stoch.k_line, stoch.d_line
        signals = pd.Series(index=k.index, data=0)
        signals[(k > d) & (k.shift(1) <= d.shift(1)) & (k < 20)] = 1
        signals[(k < d) & (k.shift(1) >= d.shift(1)) & (k > 80)] = -1
        return signals
    
    def add_oscillator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['rsi_signal'] = self.get_rsi_signals(df['rsi'])
        macd_result = self.calculate_macd(df['close'])
        df['macd'] = macd_result.macd_line
        df['macd_signal_line'] = macd_result.signal_line
        df['macd_histogram'] = macd_result.histogram
        df['macd_crossover'] = self.get_macd_signals(macd_result)
        stoch = self.calculate_stochastic(df)
        df['stoch_k'] = stoch.k_line
        df['stoch_d'] = stoch.d_line
        df['stoch_signal'] = self.get_stochastic_signals(stoch)
        df['oscillator_net_signal'] = (
            (df['rsi_signal'] == 1).astype(int) + (df['macd_crossover'] == 1).astype(int) + (df['stoch_signal'] == 1).astype(int) -
            (df['rsi_signal'] == -1).astype(int) - (df['macd_crossover'] == -1).astype(int) - (df['stoch_signal'] == -1).astype(int)
        )
        return df
