"""
Trading Signal Generator
Combines RL agent predictions with technical analysis for actionable signals
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SIGNAL_CONFIG, ACTIONS
from agents.dqn_agent import DQNAgent
from indicators.fibonacci import FibonacciCalculator
from indicators.oscillators import OscillatorCalculator
from indicators.patterns import PatternRecognizer
from indicators.volatility import VolatilityCalculator


@dataclass
class TradingSignal:
    """Structured trading signal"""
    timestamp: str
    instrument: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    reasoning: List[str]
    q_values: Dict[str, float]
    
    def to_dict(self):
        return asdict(self)
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)


class SignalGenerator:
    """
    Generates trading signals by combining RL agent with technical analysis
    """
    
    def __init__(
        self,
        agent: DQNAgent,
        min_confidence: float = None,
        risk_reward_ratio: float = None,
        atr_multiplier_sl: float = None,
        atr_multiplier_tp: float = None
    ):
        self.agent = agent
        self.min_confidence = min_confidence or SIGNAL_CONFIG['min_confidence']
        self.min_rr = risk_reward_ratio or SIGNAL_CONFIG['risk_reward_ratio']
        self.atr_mult_sl = atr_multiplier_sl or SIGNAL_CONFIG['atr_multiplier_sl']
        self.atr_mult_tp = atr_multiplier_tp or SIGNAL_CONFIG['atr_multiplier_tp']
        
        # Indicator calculators
        self.fib_calc = FibonacciCalculator()
        self.osc_calc = OscillatorCalculator()
        self.pattern_rec = PatternRecognizer()
        self.vol_calc = VolatilityCalculator()
    
    def generate_signal(
        self,
        state: np.ndarray,
        df: pd.DataFrame,
        instrument: str = 'NIFTY'
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal from current state and data
        
        Args:
            state: Current state observation
            df: Recent OHLCV data with indicators
            instrument: Instrument name
            
        Returns:
            TradingSignal or None if no actionable signal
        """
        # Get agent's Q-values and action
        q_values = self.agent.get_q_values(state)
        action_idx = np.argmax(q_values)
        action = ACTIONS[action_idx]
        
        # Calculate confidence from Q-values
        q_softmax = self._softmax(q_values)
        confidence = q_softmax[action_idx]
        
        # Skip if HOLD or low confidence
        if action == 'HOLD' or confidence < self.min_confidence:
            return None
        
        # Get current price and ATR
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.01
        
        # Calculate stop loss and targets
        if action in ['BUY', 'CLOSE']:
            stop_loss = current_price - (atr * self.atr_mult_sl)
            target_1 = current_price + (atr * self.atr_mult_tp)
            target_2 = current_price + (atr * self.atr_mult_tp * 1.5)
            trade_action = 'BUY'
        else:  # SELL
            stop_loss = current_price + (atr * self.atr_mult_sl)
            target_1 = current_price - (atr * self.atr_mult_tp)
            target_2 = current_price - (atr * self.atr_mult_tp * 1.5)
            trade_action = 'SELL'
        
        # Calculate risk-reward ratio
        risk = abs(current_price - stop_loss)
        reward = abs(target_1 - current_price)
        risk_reward = reward / risk if risk > 0 else 0
        
        # Skip if poor risk-reward
        if risk_reward < self.min_rr:
            return None
        
        # Build reasoning from indicators
        reasoning = self._build_reasoning(df, trade_action)
        
        # Q-values dict
        q_dict = {ACTIONS[i]: float(q_values[i]) for i in range(len(q_values))}
        
        return TradingSignal(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            instrument=instrument,
            action=trade_action,
            confidence=float(confidence),
            entry_price=float(current_price),
            stop_loss=float(stop_loss),
            target_1=float(target_1),
            target_2=float(target_2),
            risk_reward=float(risk_reward),
            reasoning=reasoning,
            q_values=q_dict
        )
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def _build_reasoning(self, df: pd.DataFrame, action: str) -> List[str]:
        """Build reasoning from technical indicators"""
        reasons = []
        
        # RSI
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if action == 'BUY' and rsi < 35:
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif action == 'SELL' and rsi > 65:
                reasons.append(f"RSI overbought ({rsi:.1f})")
        
        # MACD
        if 'macd_crossover' in df.columns:
            cross = df['macd_crossover'].iloc[-1]
            if cross == 1:
                reasons.append("MACD bullish crossover")
            elif cross == -1:
                reasons.append("MACD bearish crossover")
        
        # Moving average trend
        if 'ma_trend' in df.columns:
            trend = df['ma_trend'].iloc[-1]
            if action == 'BUY' and trend == 1:
                reasons.append("Price above EMA trend")
            elif action == 'SELL' and trend == -1:
                reasons.append("Price below EMA trend")
        
        # Bollinger Bands
        if 'bb_position' in df.columns:
            bb_pos = df['bb_position'].iloc[-1]
            if action == 'BUY' and bb_pos < 0.2:
                reasons.append("Price near lower Bollinger Band")
            elif action == 'SELL' and bb_pos > 0.8:
                reasons.append("Price near upper Bollinger Band")
        
        # Pattern signals
        if 'pattern_net_signal' in df.columns:
            pattern = df['pattern_net_signal'].iloc[-1]
            if action == 'BUY' and pattern > 0:
                reasons.append(f"Bullish candlestick pattern (strength: {pattern:.1f})")
            elif action == 'SELL' and pattern < 0:
                reasons.append(f"Bearish candlestick pattern (strength: {abs(pattern):.1f})")
        
        # Fibonacci
        if 'nearest_fib_level' in df.columns:
            fib_level = df['nearest_fib_level'].iloc[-1]
            fib_dist = df['nearest_fib_distance'].iloc[-1] if 'nearest_fib_distance' in df.columns else 0
            if abs(fib_dist) < 0.02:
                reasons.append(f"Price at Fibonacci {fib_level*100:.1f}% level")
        
        if not reasons:
            reasons.append("RL agent high confidence signal")
        
        return reasons
    
    def generate_multiple_signals(
        self,
        states: List[np.ndarray],
        dfs: Dict[str, pd.DataFrame]
    ) -> List[TradingSignal]:
        """Generate signals for multiple instruments"""
        signals = []
        
        for instrument, df in dfs.items():
            if instrument in ['NIFTY', 'BANKNIFTY']:
                # Create state from latest data
                state = states.get(instrument) if isinstance(states, dict) else states
                if state is not None:
                    signal = self.generate_signal(state, df, instrument)
                    if signal:
                        signals.append(signal)
        
        return signals


def format_signal_for_display(signal: TradingSignal) -> str:
    """Format signal for console/dashboard display"""
    emoji = "🟢" if signal.action == "BUY" else "🔴"
    
    output = f"""
╔══════════════════════════════════════════════════════════════╗
║  {emoji} TRADING SIGNAL - {signal.instrument}
╠══════════════════════════════════════════════════════════════╣
║  Time: {signal.timestamp}
║  Action: {signal.action}
║  Confidence: {signal.confidence*100:.1f}%
╠══════════════════════════════════════════════════════════════╣
║  Entry: ₹{signal.entry_price:.2f}
║  Stop Loss: ₹{signal.stop_loss:.2f}
║  Target 1: ₹{signal.target_1:.2f}
║  Target 2: ₹{signal.target_2:.2f}
║  Risk/Reward: {signal.risk_reward:.2f}
╠══════════════════════════════════════════════════════════════╣
║  Reasoning:
"""
    for reason in signal.reasoning:
        output += f"║    • {reason}\n"
    
    output += "╚══════════════════════════════════════════════════════════════╝"
    return output
