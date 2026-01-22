# 🤖 RL Trading Bot - Nifty & Banknifty

A **Reinforcement Learning-powered Trading Bot** that generates 15-minute timeframe signals for Nifty and Banknifty indices. Features a modern dark-themed web dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- 🧠 **Deep Q-Network (DQN)** agent with Dueling architecture
- 📊 **47+ Technical Indicators** including Fibonacci, RSI, MACD, Bollinger Bands
- 🕯️ **15+ Candlestick Patterns** recognition
- 🔄 **Continuous Learning** - daily updates and weekly retraining
- 🌐 **Modern Web Dashboard** with dark theme
- 📈 **Interactive Charts** powered by Lightweight Charts

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/manusanjay545/bot.git
cd bot

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
# Train on Nifty (200+ episodes recommended)
python main.py train --instrument NIFTY --episodes 500

# Train on Banknifty
python main.py train --instrument BANKNIFTY --episodes 500
```

### Generate Signals

```bash
python main.py signal --instrument NIFTY
```

### Launch Web Dashboard

```bash
# Option 1: Flask Dashboard (recommended)
python dashboard/server.py

# Option 2: Streamlit Dashboard
python main.py dashboard
```

Then open: **http://localhost:5000**

## 📁 Project Structure

```
bot/
├── main.py                    # CLI entry point
├── config.py                  # Configuration settings
├── requirements.txt           # Dependencies
├── data/
│   ├── fetcher.py            # Market data from Yahoo Finance
│   └── preprocessor.py       # Data normalization
├── indicators/
│   ├── fibonacci.py          # Fibonacci levels
│   ├── patterns.py           # Candlestick patterns
│   ├── oscillators.py        # RSI, MACD, Stochastic
│   ├── moving_averages.py    # SMA, EMA, crossovers
│   ├── volatility.py         # Bollinger Bands, ATR
│   └── support_resistance.py # Pivot points
├── environment/
│   └── trading_env.py        # Gym-compatible environment
├── agents/
│   ├── dqn_agent.py          # Dueling DQN agent
│   └── replay_buffer.py      # Experience replay
├── training/
│   ├── trainer.py            # Training loop
│   └── continuous_learner.py # Scheduled retraining
├── signals/
│   └── signal_generator.py   # Signal generation
├── dashboard/
│   ├── server.py             # Flask API server
│   ├── templates/            # HTML templates
│   └── static/               # CSS & JavaScript
└── models/                   # Saved model weights
```

## 🎯 Trading Signal Output

```
╔══════════════════════════════════════════════════════════════╗
║  🟢 TRADING SIGNAL - NIFTY
╠══════════════════════════════════════════════════════════════╣
║  Action: BUY
║  Confidence: 72.5%
║  Entry: ₹23,150.00
║  Stop Loss: ₹23,050.00
║  Target 1: ₹23,350.00
║  Risk/Reward: 2.5
╠══════════════════════════════════════════════════════════════╣
║  Reasoning:
║    • RSI oversold (28.5)
║    • MACD bullish crossover
║    • Price at Fibonacci 61.8% level
╚══════════════════════════════════════════════════════════════╝
```

## 📊 Technical Indicators

| Category | Indicators |
|----------|------------|
| **Fibonacci** | Retracement (23.6%, 38.2%, 50%, 61.8%, 78.6%), Extensions |
| **Patterns** | Hammer, Engulfing, Morning/Evening Star, Doji, Three Soldiers |
| **Oscillators** | RSI (14), MACD (12/26/9), Stochastic (14/3) |
| **Moving Averages** | SMA (9, 20, 50, 200), EMA (9, 21, 55) |
| **Volatility** | Bollinger Bands, ATR, Historical Volatility |

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Trading
TIMEFRAME = '15m'              # Candle timeframe
INITIAL_BALANCE = 100000       # Starting capital (INR)
TRANSACTION_COST = 0.0003      # 0.03% per trade

# DQN Agent
DQN_CONFIG = {
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'epsilon_decay': 0.995,
    'batch_size': 64,
}

# Signal Generation
SIGNAL_CONFIG = {
    'min_confidence': 0.6,
    'risk_reward_ratio': 2.0,
}
```

## ⚠️ Disclaimer

This bot is for **educational purposes only**. Trading in financial markets involves substantial risk of loss. Always:

- Paper trade first
- Use proper risk management
- Never invest more than you can afford to lose
- Past performance does not guarantee future results

## 📝 License

MIT License - feel free to use and modify.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.
