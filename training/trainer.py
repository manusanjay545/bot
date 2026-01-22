"""
Training pipeline for RL trading agent
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINING_CONFIG, MODELS_DIR, LOGS_DIR
from agents.dqn_agent import DQNAgent
from environment.trading_env import TradingEnvironment


class Trainer:
    """Training pipeline for RL trading agent"""
    
    def __init__(
        self,
        env: TradingEnvironment,
        agent: DQNAgent,
        episodes: int = None,
        max_steps: int = None,
        eval_frequency: int = None,
        save_frequency: int = None,
        early_stopping_patience: int = None
    ):
        self.env = env
        self.agent = agent
        
        self.episodes = episodes or TRAINING_CONFIG['episodes']
        self.max_steps = max_steps or TRAINING_CONFIG['max_steps_per_episode']
        self.eval_frequency = eval_frequency or TRAINING_CONFIG['eval_frequency']
        self.save_frequency = save_frequency or TRAINING_CONFIG['save_frequency']
        self.patience = early_stopping_patience or TRAINING_CONFIG['early_stopping_patience']
        
        self.training_history = []
        self.best_reward = float('-inf')
        self.patience_counter = 0
    
    def train(self, verbose: bool = True) -> Dict:
        """Run full training loop"""
        start_time = time.time()
        
        for episode in range(1, self.episodes + 1):
            episode_reward, episode_loss, info = self._run_episode()
            
            self.training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'loss': episode_loss,
                'epsilon': self.agent.epsilon,
                'balance': info['balance'],
                'total_trades': info['total_trades'],
                'win_rate': info['win_rate']
            })
            
            self.agent.end_episode()
            
            if verbose and episode % 10 == 0:
                self._print_progress(episode, episode_reward, episode_loss, info)
            
            # Evaluation
            if episode % self.eval_frequency == 0:
                eval_reward = self._evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self.patience_counter = 0
                    self.agent.save(os.path.join(MODELS_DIR, 'best_model.pt'))
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at episode {episode}")
                    break
            
            # Save checkpoint
            if episode % self.save_frequency == 0:
                self.agent.save()
        
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/60:.1f} minutes")
        
        return self._get_training_summary()
    
    def _run_episode(self) -> Tuple[float, float, Dict]:
        """Run a single training episode"""
        state, info = self.env.reset()
        episode_reward = 0.0
        losses = []
        
        for step in range(self.max_steps):
            action = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            self.agent.store_transition(state, action, reward, next_state, terminated or truncated)
            
            loss = self.agent.train_step_update()
            if loss is not None:
                losses.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        avg_loss = np.mean(losses) if losses else 0.0
        return episode_reward, avg_loss, info
    
    def _evaluate(self, n_episodes: int = 5) -> float:
        """Evaluate agent without exploration"""
        total_reward = 0.0
        
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            
            while True:
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
        
        return total_reward / n_episodes
    
    def _print_progress(self, episode, reward, loss, info):
        print(f"Episode {episode}/{self.episodes} | "
              f"Reward: {reward:.4f} | Loss: {loss:.6f} | "
              f"ε: {self.agent.epsilon:.4f} | "
              f"Balance: ₹{info['balance']:.0f} | "
              f"Trades: {info['total_trades']} | "
              f"Win: {info['win_rate']:.1%}")
    
    def _get_training_summary(self) -> Dict:
        """Generate training summary"""
        df = pd.DataFrame(self.training_history)
        return {
            'total_episodes': len(df),
            'final_epsilon': self.agent.epsilon,
            'best_reward': self.best_reward,
            'avg_reward_last_100': df['reward'].tail(100).mean(),
            'avg_win_rate_last_100': df['win_rate'].tail(100).mean(),
            'training_history': df
        }
    
    def save_training_log(self):
        """Save training history to file"""
        df = pd.DataFrame(self.training_history)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(LOGS_DIR, f'training_log_{timestamp}.csv')
        df.to_csv(filepath, index=False)
        print(f"Training log saved to {filepath}")
