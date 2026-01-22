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
    """Add technical indicators (simplified for dashboard)"""
    try:
        df = df.copy()
        df = OscillatorCalculator().add_oscillator_features(df)
        df = MovingAverageCalculator().add_ma_features(df)
        df = VolatilityCalculator().add_volatility_features(df)
        df = PatternRecognizer().add_pattern_features(df)
        # Use fillna to keep data, skip rows only at the beginning
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='bfill').fillna(0)
        return df
    except Exception as e:
        print(f"Error adding indicators: {e}")
        return df

def get_market_data(instrument='NIFTY', days=30):
    """Fetch and prepare market data"""
    try:
        df = fetcher.fetch_historical_data(instrument, days=days, use_cache=False)
        if df is None or df.empty:
            return None
        df = add_indicators(df)
        if df.empty:
            # Return raw data if indicators fail
            df = fetcher.fetch_historical_data(instrument, days=days, use_cache=False)
        return df
    except Exception as e:
        print(f"Error getting market data: {e}")
        return None

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
    """Generate trading signal based on technical indicators"""
    try:
        df = get_market_data(instrument.upper(), days=30)
        if df is None or len(df) < 10:
            return jsonify({'error': 'Insufficient data', 'signal': None})
        
        latest = df.iloc[-1]
        current_price = float(latest['close'])
        
        # Rule-based signal generation
        reasoning = []
        bullish_score = 0
        bearish_score = 0
        
        # RSI Analysis
        if 'rsi' in df.columns:
            rsi = float(latest['rsi'])
            if rsi < 30:
                bullish_score += 2
                reasoning.append(f'RSI oversold ({rsi:.1f})')
            elif rsi < 40:
                bullish_score += 1
                reasoning.append(f'RSI approaching oversold ({rsi:.1f})')
            elif rsi > 70:
                bearish_score += 2
                reasoning.append(f'RSI overbought ({rsi:.1f})')
            elif rsi > 60:
                bearish_score += 1
                reasoning.append(f'RSI approaching overbought ({rsi:.1f})')
        
        # MACD Analysis
        if 'macd_crossover' in df.columns:
            macd_cross = int(latest['macd_crossover'])
            if macd_cross == 1:
                bullish_score += 2
                reasoning.append('MACD bullish crossover')
            elif macd_cross == -1:
                bearish_score += 2
                reasoning.append('MACD bearish crossover')
        
        # Moving Average Trend
        if 'ma_trend' in df.columns:
            trend = int(latest['ma_trend'])
            if trend == 1:
                bullish_score += 1
                reasoning.append('Price above EMA trend')
            else:
                bearish_score += 1
                reasoning.append('Price below EMA trend')
        
        # Bollinger Band Position
        if 'bb_position' in df.columns:
            bb_pos = float(latest['bb_position'])
            if bb_pos < 0.2:
                bullish_score += 1
                reasoning.append('Price near lower Bollinger Band')
            elif bb_pos > 0.8:
                bearish_score += 1
                reasoning.append('Price near upper Bollinger Band')
        
        # Stochastic
        if 'stoch_signal' in df.columns:
            stoch = int(latest['stoch_signal'])
            if stoch == 1:
                bullish_score += 1
                reasoning.append('Stochastic bullish')
            elif stoch == -1:
                bearish_score += 1
                reasoning.append('Stochastic bearish')
        
        # Pattern signals
        if 'pattern_net_signal' in df.columns:
            pattern = float(latest['pattern_net_signal'])
            if pattern > 0.5:
                bullish_score += 1
                reasoning.append(f'Bullish candlestick pattern')
            elif pattern < -0.5:
                bearish_score += 1
                reasoning.append('Bearish candlestick pattern')
        
        # Calculate ATR for stops/targets
        atr = float(latest['atr']) if 'atr' in df.columns else current_price * 0.01
        
        # Determine signal
        total_score = bullish_score - bearish_score
        
        # Calculate levels based on direction even if confidence is low
        if total_score >= 0:
            # Bullish bias
            stop_loss = current_price - (atr * 1.5)
            target_1 = current_price + (atr * 2)
            target_2 = current_price + (atr * 3)
            risk_reward = (target_1 - current_price) / (current_price - stop_loss)
        else:
            # Bearish bias
            stop_loss = current_price + (atr * 1.5)
            target_1 = current_price - (atr * 2)
            target_2 = current_price - (atr * 3)
            risk_reward = (current_price - target_1) / (stop_loss - current_price)

        if total_score >= 3:
            action = 'BUY'
            confidence = min(90, 50 + total_score * 10)
        elif total_score <= -3:
            action = 'SELL'
            confidence = min(90, 50 + abs(total_score) * 10)
        else:
            action = 'HOLD'
            confidence = 0
            if not reasoning:
                reasoning = ['No clear directional signal - wait for better setup']
        
        return jsonify({
            'signal': {
                'action': action,
                'confidence': confidence,
                'entry': round(current_price, 2),
                'stopLoss': round(stop_loss, 2),
                'target1': round(target_1, 2),
                'target2': round(target_2, 2),
                'riskReward': round(risk_reward, 2),
                'reasoning': reasoning,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        })
    except Exception as e:
        print(f"Signal error: {e}")
        return jsonify({
            'signal': {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': [f'Analysis pending: {str(e)[:50]}'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        })

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
