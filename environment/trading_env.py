"""
Gym-compatible Trading Environment for Reinforcement Learning
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from enum import IntEnum
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INITIAL_BALANCE, TRANSACTION_COST, SLIPPAGE, STATE_WINDOW, MAX_POSITION, ACTIONS

class Action(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3

class TradingEnvironment(gym.Env):
    """
    OpenAI Gym compatible trading environment for RL agents.
    State: Technical indicator features + position info
    Action: HOLD, BUY, SELL, CLOSE
    Reward: P&L based with risk adjustments
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list = None,
        initial_balance: float = INITIAL_BALANCE,
        transaction_cost: float = TRANSACTION_COST,
        slippage: float = SLIPPAGE,
        window_size: int = STATE_WINDOW,
        max_position: int = MAX_POSITION,
        reward_scaling: float = 1e-4
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.window_size = window_size
        self.max_position = max_position
        self.reward_scaling = reward_scaling
        
        # Feature columns (exclude non-numeric)
        if feature_columns is None:
            self.feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.feature_columns = feature_columns
        
        self.n_features = len(self.feature_columns)
        
        # State: window of features + position info (2 extra: position, unrealized_pnl)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size * self.n_features + 2,),
            dtype=np.float32
        )
        
        # Actions: HOLD, BUY, SELL, CLOSE
        self.action_space = spaces.Discrete(4)
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Ensure we have enough data
        if len(self.df) <= self.window_size + 1:
            raise ValueError(f"Insufficient data: {len(self.df)} rows, need at least {self.window_size + 2}")
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # 0=flat, 1=long, -1=short
        self.position_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.trade_history = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        current_price = self._get_current_price()
        reward = 0.0
        
        # Execute action
        if action == Action.BUY and self.position <= 0:
            reward = self._execute_trade(1, current_price)
        elif action == Action.SELL and self.position >= 0:
            reward = self._execute_trade(-1, current_price)
        elif action == Action.CLOSE and self.position != 0:
            reward = self._close_position(current_price)
        elif action == Action.HOLD:
            # Small penalty for holding with no position in trending market
            pass
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = self.balance <= 0
        
        # Add unrealized P&L to reward
        if self.position != 0:
            unrealized = self._calculate_unrealized_pnl(current_price)
            reward += unrealized * self.reward_scaling * 0.1  # Small contribution
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _execute_trade(self, direction: int, price: float) -> float:
        """Execute a trade (buy=1, sell=-1)"""
        reward = 0.0
        
        # Close existing position first if opposite direction
        if self.position != 0 and self.position != direction:
            reward = self._close_position(price)
        
        # Apply slippage
        execution_price = price * (1 + self.slippage * direction)
        
        # Open new position
        self.position = direction
        self.position_price = execution_price
        
        # Transaction cost
        cost = self.balance * self.transaction_cost
        self.balance -= cost
        reward -= cost * self.reward_scaling
        
        self.total_trades += 1
        self.trade_history.append({
            'step': self.current_step,
            'action': 'BUY' if direction == 1 else 'SELL',
            'price': execution_price,
            'balance': self.balance
        })
        
        return reward
    
    def _close_position(self, current_price: float) -> float:
        """Close current position"""
        if self.position == 0:
            return 0.0
        
        # Apply slippage (opposite direction)
        execution_price = current_price * (1 - self.slippage * self.position)
        
        # Calculate P&L
        if self.position == 1:  # Long
            pnl = (execution_price - self.position_price) / self.position_price
        else:  # Short
            pnl = (self.position_price - execution_price) / self.position_price
        
        pnl_amount = self.balance * pnl
        
        # Transaction cost
        cost = self.balance * self.transaction_cost
        pnl_amount -= cost
        
        self.balance += pnl_amount
        self.total_pnl += pnl_amount
        
        if pnl_amount > 0:
            self.winning_trades += 1
        
        self.trade_history.append({
            'step': self.current_step,
            'action': 'CLOSE',
            'price': execution_price,
            'pnl': pnl_amount,
            'balance': self.balance
        })
        
        # Reset position
        self.position = 0
        self.position_price = 0.0
        
        return pnl_amount * self.reward_scaling
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if self.position == 0:
            return 0.0
        
        if self.position == 1:
            return (current_price - self.position_price) / self.position_price * self.balance
        else:
            return (self.position_price - current_price) / self.position_price * self.balance
    
    def _get_current_price(self) -> float:
        return self.df['close'].iloc[self.current_step]
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        # Get window of features
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        window_data = self.df[self.feature_columns].iloc[start_idx:end_idx].values.flatten()
        
        # Normalize price-related features
        window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Add position info
        current_price = self._get_current_price()
        unrealized_pnl = self._calculate_unrealized_pnl(current_price) / self.initial_balance
        
        position_info = np.array([self.position, unrealized_pnl])
        
        observation = np.concatenate([window_data, position_info]).astype(np.float32)
        return observation
    
    def _get_info(self) -> dict:
        """Get info dict"""
        win_rate = self.winning_trades / max(self.total_trades, 1)
        return {
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'current_step': self.current_step
        }
    
    def render(self, mode='human'):
        info = self._get_info()
        print(f"Step: {info['current_step']} | Balance: ₹{info['balance']:.2f} | "
              f"Position: {info['position']} | Trades: {info['total_trades']} | "
              f"Win Rate: {info['win_rate']:.2%}")


def create_trading_env(df: pd.DataFrame, **kwargs) -> TradingEnvironment:
    """Factory function to create trading environment"""
    return TradingEnvironment(df, **kwargs)
