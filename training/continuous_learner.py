"""
Continuous Learning System
Enables the bot to keep learning from new market data
"""
import os
import sys
import schedule
import time
from datetime import datetime
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor
from agents.dqn_agent import DQNAgent
from environment.trading_env import TradingEnvironment
from .trainer import Trainer


class ContinuousLearner:
    """
    Manages continuous learning:
    1. Daily updates with new market data
    2. Periodic retraining
    3. Model versioning
    """
    
    def __init__(
        self,
        agent: DQNAgent,
        fetcher: DataFetcher,
        preprocessor: DataPreprocessor,
        feature_columns: list,
        instrument: str = 'NIFTY',
        daily_update_episodes: int = 50,
        weekly_retrain_episodes: int = 200
    ):
        self.agent = agent
        self.fetcher = fetcher
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns
        self.instrument = instrument
        self.daily_episodes = daily_update_episodes
        self.weekly_episodes = weekly_retrain_episodes
        
        self.last_daily_update = None
        self.last_weekly_retrain = None
        self.update_history = []
    
    def daily_update(self):
        """Perform daily incremental learning with latest data"""
        print(f"\n[{datetime.now()}] Starting daily update...")
        
        try:
            # Fetch latest data
            df = self.fetcher.fetch_historical_data(self.instrument, days=7, use_cache=False)
            
            if df.empty:
                print("No new data available. Skipping update.")
                return
            
            # Preprocess
            df_processed = self.preprocessor.prepare_for_training(df)
            
            # Create environment with latest data
            env = TradingEnvironment(df_processed, feature_columns=self.feature_columns)
            
            # Short training session
            trainer = Trainer(env, self.agent, episodes=self.daily_episodes)
            summary = trainer.train(verbose=False)
            
            # Save updated model
            timestamp = datetime.now().strftime("%Y%m%d")
            self.agent.save(os.path.join(MODELS_DIR, f'daily_{timestamp}.pt'))
            
            self.last_daily_update = datetime.now()
            self.update_history.append({
                'type': 'daily',
                'timestamp': self.last_daily_update,
                'episodes': self.daily_episodes,
                'avg_reward': summary['avg_reward_last_100']
            })
            
            print(f"Daily update completed. Avg reward: {summary['avg_reward_last_100']:.4f}")
            
        except Exception as e:
            print(f"Daily update failed: {e}")
    
    def weekly_retrain(self):
        """Perform weekly full retraining"""
        print(f"\n[{datetime.now()}] Starting weekly retrain...")
        
        try:
            # Fetch more historical data
            df = self.fetcher.fetch_historical_data(self.instrument, days=60, use_cache=False)
            
            if df.empty:
                print("No data available for retraining.")
                return
            
            # Preprocess
            df_processed = self.preprocessor.prepare_for_training(df)
            
            # Create environment
            env = TradingEnvironment(df_processed, feature_columns=self.feature_columns)
            
            # Longer training session
            trainer = Trainer(env, self.agent, episodes=self.weekly_episodes)
            summary = trainer.train(verbose=True)
            
            # Save retrained model
            timestamp = datetime.now().strftime("%Y%m%d")
            self.agent.save(os.path.join(MODELS_DIR, f'weekly_{timestamp}.pt'))
            
            self.last_weekly_retrain = datetime.now()
            self.update_history.append({
                'type': 'weekly',
                'timestamp': self.last_weekly_retrain,
                'episodes': self.weekly_episodes,
                'avg_reward': summary['avg_reward_last_100'],
                'win_rate': summary['avg_win_rate_last_100']
            })
            
            print(f"Weekly retrain completed.")
            
        except Exception as e:
            print(f"Weekly retrain failed: {e}")
    
    def start_scheduled_updates(self):
        """Start scheduled learning updates"""
        # Daily update at 4 PM IST (after market close)
        schedule.every().day.at("16:00").do(self.daily_update)
        
        # Weekly retrain on Saturday
        schedule.every().saturday.at("10:00").do(self.weekly_retrain)
        
        print("Scheduled updates started:")
        print("  - Daily updates at 4:00 PM")
        print("  - Weekly retraining on Saturday 10:00 AM")
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def manual_update(self, episodes: int = 100):
        """Trigger manual learning update"""
        print(f"Starting manual update with {episodes} episodes...")
        
        df = self.fetcher.fetch_historical_data(self.instrument, days=30, use_cache=False)
        df_processed = self.preprocessor.prepare_for_training(df)
        env = TradingEnvironment(df_processed, feature_columns=self.feature_columns)
        
        trainer = Trainer(env, self.agent, episodes=episodes)
        summary = trainer.train(verbose=True)
        
        self.agent.save(os.path.join(MODELS_DIR, 'manual_update.pt'))
        
        return summary
    
    def get_update_history(self):
        """Get learning update history"""
        return self.update_history
