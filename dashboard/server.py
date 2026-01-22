"""
Flask Web Server for RL Trading Bot Dashboard
"""
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
import sys
import json
from datetime import datetime
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import DataFetcher
from indicators.oscillators import OscillatorCalculator
from indicators.patterns import PatternRecognizer
from indicators.moving_averages import MovingAverageCalculator
from indicators.volatility import VolatilityCalculator
from indicators.fibonacci import FibonacciCalculator
from environment.trading_env import TradingEnvironment
from agents.dqn_agent import DQNAgent
from signals.signal_generator import SignalGenerator
from config import MODELS_DIR

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global components
fetcher = DataFetcher()

def add_indicators(df):
    """Add all technical indicators"""
    df = OscillatorCalculator().add_oscillator_features(df)
    df = MovingAverageCalculator().add_ma_features(df)
    df = VolatilityCalculator().add_volatility_features(df)
    df = PatternRecognizer().add_pattern_features(df)
    df = FibonacciCalculator().add_fibonacci_features(df)
    return df.dropna()

def get_market_data(instrument='NIFTY', days=30):
    """Fetch and prepare market data"""
    df = fetcher.fetch_historical_data(instrument, days=days, use_cache=False)
    if df.empty:
        return None
    df = add_indicators(df)
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/market-data/<instrument>')
def api_market_data(instrument):
    """Get OHLCV data with indicators"""
    days = request.args.get('days', 30, type=int)
    df = get_market_data(instrument.upper(), days)
    
    if df is None or df.empty:
        return jsonify({'error': 'Failed to fetch data'}), 500
    
    # Convert to JSON-friendly format
    data = {
        'timestamps': df.index.strftime('%Y-%m-%d %H:%M').tolist(),
        'open': df['open'].round(2).tolist(),
        'high': df['high'].round(2).tolist(),
        'low': df['low'].round(2).tolist(),
        'close': df['close'].round(2).tolist(),
        'volume': df['volume'].tolist() if 'volume' in df.columns else [],
        'rsi': df['rsi'].round(2).tolist() if 'rsi' in df.columns else [],
        'macd': df['macd'].round(4).tolist() if 'macd' in df.columns else [],
        'macd_signal': df['macd_signal_line'].round(4).tolist() if 'macd_signal_line' in df.columns else [],
        'bb_upper': df['bb_upper'].round(2).tolist() if 'bb_upper' in df.columns else [],
        'bb_lower': df['bb_lower'].round(2).tolist() if 'bb_lower' in df.columns else [],
        'ema_9': df['ema_9'].round(2).tolist() if 'ema_9' in df.columns else [],
        'ema_21': df['ema_21'].round(2).tolist() if 'ema_21' in df.columns else [],
    }
    
    return jsonify(data)

@app.route('/api/signal/<instrument>')
def api_signal(instrument):
    """Generate trading signal"""
    try:
        df = get_market_data(instrument.upper(), days=30)
        if df is None or len(df) < 25:
            return jsonify({'error': 'Insufficient data', 'signal': None})
        
        feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        env = TradingEnvironment(df, feature_columns=feature_cols)
        
        # Walk to latest state
        state, _ = env.reset()
        while env.current_step < len(df) - 1:
            _, _, done, trunc, _ = env.step(0)
            if done or trunc:
                break
        state = env._get_observation()
        
        # Load agent
        model_path = os.path.join(MODELS_DIR, f'{instrument.lower()}_agent_final.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, 'best_model.pt')
        
        if not os.path.exists(model_path):
            return jsonify({'error': 'No trained model found', 'signal': None})
        
        agent = DQNAgent(len(state), 4)
        agent.load(model_path)
        
        # Generate signal
        signal_gen = SignalGenerator(agent)
        signal = signal_gen.generate_signal(state, df, instrument.upper())
        
        if signal:
            return jsonify({
                'signal': {
                    'action': signal.action,
                    'confidence': round(signal.confidence * 100, 1),
                    'entry': round(signal.entry_price, 2),
                    'stopLoss': round(signal.stop_loss, 2),
                    'target1': round(signal.target_1, 2),
                    'target2': round(signal.target_2, 2),
                    'riskReward': round(signal.risk_reward, 2),
                    'reasoning': signal.reasoning,
                    'timestamp': signal.timestamp
                }
            })
        else:
            return jsonify({
                'signal': {
                    'action': 'HOLD',
                    'confidence': 0,
                    'reasoning': ['No high-confidence signal at this time'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            })
    except Exception as e:
        return jsonify({'error': str(e), 'signal': None}), 500

@app.route('/api/indicators/<instrument>')
def api_indicators(instrument):
    """Get current indicator values"""
    df = get_market_data(instrument.upper(), days=7)
    if df is None or df.empty:
        return jsonify({'error': 'Failed to fetch data'}), 500
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    current_price = float(latest['close'])
    change = current_price - float(prev['close'])
    change_pct = (change / float(prev['close'])) * 100
    
    indicators = {
        'price': round(current_price, 2),
        'change': round(change, 2),
        'changePct': round(change_pct, 2),
        'rsi': round(float(latest['rsi']), 1) if 'rsi' in df.columns else None,
        'macd': round(float(latest['macd']), 4) if 'macd' in df.columns else None,
        'atr': round(float(latest['atr']), 2) if 'atr' in df.columns else None,
        'trend': 'Bullish' if latest.get('ma_trend', 0) == 1 else 'Bearish',
        'bbPosition': round(float(latest['bb_position']) * 100, 1) if 'bb_position' in df.columns else None,
        'volatility': 'High' if latest.get('volatility_regime', 0) == 1 else 'Low' if latest.get('volatility_regime', 0) == -1 else 'Normal'
    }
    
    return jsonify(indicators)

@app.route('/api/patterns/<instrument>')
def api_patterns(instrument):
    """Get detected candlestick patterns"""
    df = get_market_data(instrument.upper(), days=7)
    if df is None or df.empty:
        return jsonify({'patterns': []})
    
    recognizer = PatternRecognizer()
    patterns = recognizer.detect_all_patterns(df)
    
    result = []
    for p in patterns:
        result.append({
            'name': p.name,
            'type': p.type.name,
            'strength': round(p.strength * 100, 0),
            'description': p.description
        })
    
    return jsonify({'patterns': result})

@app.route('/api/status')
def api_status():
    """API health check"""
    models = []
    if os.path.exists(MODELS_DIR):
        models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')]
    
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'models': models
    })

if __name__ == '__main__':
    print("🚀 Starting RL Trading Bot Server...")
    print("📊 Dashboard: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
