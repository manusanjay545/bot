"""
RL Trading Bot - Main Entry Point
Trains the agent and generates signals for Nifty/Banknifty on 15-min timeframe
"""
import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MODELS_DIR, TRAINING_CONFIG
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor
from indicators.fibonacci import FibonacciCalculator
from indicators.oscillators import OscillatorCalculator
from indicators.patterns import PatternRecognizer
from indicators.moving_averages import MovingAverageCalculator
from indicators.volatility import VolatilityCalculator
from indicators.support_resistance import SupportResistanceCalculator
from environment.trading_env import TradingEnvironment
from agents.dqn_agent import DQNAgent
from training.trainer import Trainer
from training.continuous_learner import ContinuousLearner
from signals.signal_generator import SignalGenerator, format_signal_for_display


def add_all_indicators(df):
    """Add all technical indicators to the dataframe"""
    df = OscillatorCalculator().add_oscillator_features(df)
    df = MovingAverageCalculator().add_ma_features(df)
    df = VolatilityCalculator().add_volatility_features(df)
    df = PatternRecognizer().add_pattern_features(df)
    df = FibonacciCalculator().add_fibonacci_features(df)
    df = SupportResistanceCalculator().add_sr_features(df)
    return df.dropna()


def train_agent(instrument='NIFTY', episodes=500, days=55):
    """Train a new agent from scratch"""
    print(f"\n{'='*60}")
    print(f"  TRAINING RL AGENT FOR {instrument}")
    print(f"  Episodes: {episodes} | Historical Days: {days}")
    print(f"{'='*60}\n")
    
    # Fetch and prepare data
    print("📊 Fetching market data...")
    fetcher = DataFetcher()
    df = fetcher.fetch_historical_data(instrument, days=days)
    
    if df.empty:
        print("❌ Failed to fetch data. Exiting.")
        return None
    
    print(f"✅ Fetched {len(df)} candles")
    
    # Add indicators
    print("📈 Adding technical indicators...")
    df = add_all_indicators(df)
    print(f"✅ Features: {len(df.columns)} columns")
    
    # Get feature columns
    feature_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Create environment
    print("🎮 Creating trading environment...")
    env = TradingEnvironment(df, feature_columns=feature_columns)
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"🤖 Creating DQN Agent (state: {state_dim}, actions: {action_dim})...")
    agent = DQNAgent(state_dim, action_dim)
    
    # Train
    print("\n🚀 Starting training...\n")
    trainer = Trainer(env, agent, episodes=episodes)
    summary = trainer.train(verbose=True)
    
    # Save model and log
    agent.save(os.path.join(MODELS_DIR, f'{instrument.lower()}_agent_final.pt'))
    trainer.save_training_log()
    
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"  Best Reward: {summary['best_reward']:.4f}")
    print(f"  Final Win Rate: {summary['avg_win_rate_last_100']:.1%}")
    print(f"{'='*60}\n")
    
    return agent, feature_columns


def generate_live_signals(instrument='NIFTY'):
    """Generate trading signals using trained agent"""
    print(f"\n📡 Generating signals for {instrument}...")
    
    # Fetch latest data
    fetcher = DataFetcher()
    df = fetcher.fetch_historical_data(instrument, days=30, use_cache=False)
    
    if df.empty:
        print("❌ Failed to fetch data")
        return
    
    # Add indicators
    df = add_all_indicators(df)
    feature_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Create environment to get state
    env = TradingEnvironment(df, feature_columns=feature_columns)
    state, _ = env.reset()
    
    # Walk to the latest step
    while env.current_step < len(df) - 1:
        _, _, terminated, truncated, _ = env.step(0)  # HOLD
        if terminated or truncated:
            break
    
    state = env._get_observation()
    
    # Load trained agent
    model_path = os.path.join(MODELS_DIR, f'{instrument.lower()}_agent_final.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, 'best_model.pt')
    
    if not os.path.exists(model_path):
        print("❌ No trained model found. Please train first.")
        return
    
    state_dim = len(state)
    agent = DQNAgent(state_dim, 4)
    agent.load(model_path)
    
    # Generate signal
    signal_gen = SignalGenerator(agent)
    signal = signal_gen.generate_signal(state, df, instrument)
    
    if signal:
        print(format_signal_for_display(signal))
    else:
        print(f"⏸️  No actionable signal for {instrument} at this time (HOLD)")


def run_dashboard():
    """Launch the Streamlit dashboard"""
    import subprocess
    dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard', 'app.py')
    print("🚀 Launching dashboard...")
    subprocess.run(['streamlit', 'run', dashboard_path])


def main():
    parser = argparse.ArgumentParser(description='RL Trading Bot for Nifty & Banknifty')
    parser.add_argument('command', choices=['train', 'signal', 'dashboard', 'continuous'],
                       help='Command to run')
    parser.add_argument('--instrument', '-i', default='NIFTY', choices=['NIFTY', 'BANKNIFTY'],
                       help='Trading instrument')
    parser.add_argument('--episodes', '-e', type=int, default=500,
                       help='Number of training episodes')
    parser.add_argument('--days', '-d', type=int, default=55,
                       help='Days of historical data (max 55 for 15m timeframe)')
    
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║          RL TRADING BOT - NIFTY & BANKNIFTY               ║
    ║                  15-Minute Timeframe                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    if args.command == 'train':
        train_agent(args.instrument, args.episodes, args.days)
    
    elif args.command == 'signal':
        generate_live_signals(args.instrument)
    
    elif args.command == 'dashboard':
        run_dashboard()
    
    elif args.command == 'continuous':
        print("Starting continuous learning mode...")
        # First train if no model exists
        model_path = os.path.join(MODELS_DIR, f'{args.instrument.lower()}_agent_final.pt')
        if not os.path.exists(model_path):
            agent, feature_columns = train_agent(args.instrument, args.episodes, args.days)
        else:
            # Load existing agent
            fetcher = DataFetcher()
            df = fetcher.fetch_historical_data(args.instrument, days=30)
            df = add_all_indicators(df)
            feature_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            env = TradingEnvironment(df, feature_columns=feature_columns)
            agent = DQNAgent(env.observation_space.shape[0], 4)
            agent.load(model_path)
        
        # Start continuous learning
        learner = ContinuousLearner(
            agent=agent,
            fetcher=DataFetcher(),
            preprocessor=DataPreprocessor(),
            feature_columns=feature_columns,
            instrument=args.instrument
        )
        learner.start_scheduled_updates()


if __name__ == "__main__":
    main()
