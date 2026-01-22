"""
Configuration settings for the RL Trading Bot
"""
import os
from datetime import datetime, timedelta

# =============================================================================
# Trading Configuration
# =============================================================================
INSTRUMENTS = {
    'NIFTY': '^NSEI',           # Yahoo Finance symbol for Nifty 50
    'BANKNIFTY': '^NSEBANK'      # Yahoo Finance symbol for Bank Nifty
}

TIMEFRAME = '15m'               # 15-minute candles
LOOKBACK_DAYS = 55              # Days of historical data to fetch (Yahoo 15m limit is 60 days)

# Trading hours (IST)
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

# =============================================================================
# Technical Analysis Parameters
# =============================================================================
# Fibonacci levels
FIBONACCI_RETRACEMENT = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIBONACCI_EXTENSION = [1.272, 1.618, 2.618]

# Moving Averages
SMA_PERIODS = [9, 20, 50, 200]
EMA_PERIODS = [9, 21, 55]

# Oscillators
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

STOCH_K = 14
STOCH_D = 3

# Volatility
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14

# =============================================================================
# RL Environment Configuration
# =============================================================================
INITIAL_BALANCE = 100000        # Starting capital in INR
TRANSACTION_COST = 0.0003       # 0.03% per trade (brokerage + taxes)
SLIPPAGE = 0.0001              # 0.01% slippage

# State space configuration
STATE_WINDOW = 20               # Number of candles to look back
MAX_POSITION = 1                # Maximum position size (1 = 1 lot)

# Action space
ACTIONS = {
    0: 'HOLD',
    1: 'BUY',
    2: 'SELL',
    3: 'CLOSE'
}

# =============================================================================
# RL Agent Configuration
# =============================================================================
# DQN Parameters
DQN_CONFIG = {
    'learning_rate': 0.0001,
    'gamma': 0.99,              # Discount factor
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'batch_size': 64,
    'memory_size': 100000,
    'target_update': 10,        # Update target network every N episodes
    'hidden_layers': [256, 128, 64]
}

# PPO Parameters
PPO_CONFIG = {
    'learning_rate': 0.0003,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'n_epochs': 10,
    'batch_size': 64
}

# =============================================================================
# Training Configuration
# =============================================================================
TRAINING_CONFIG = {
    'episodes': 1000,
    'max_steps_per_episode': 1000,
    'eval_frequency': 50,       # Evaluate every N episodes
    'save_frequency': 100,      # Save model every N episodes
    'early_stopping_patience': 50
}

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data_cache')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# =============================================================================
# Signal Generation
# =============================================================================
SIGNAL_CONFIG = {
    'min_confidence': 0.6,      # Minimum confidence to generate signal
    'risk_reward_ratio': 2.0,   # Minimum risk-reward ratio
    'max_risk_percent': 2.0,    # Maximum risk per trade (% of capital)
    'atr_multiplier_sl': 1.5,   # ATR multiplier for stop loss
    'atr_multiplier_tp': 3.0    # ATR multiplier for take profit
}
