"""
Streamlit Dashboard for RL Trading Bot
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from signals.signal_generator import SignalGenerator, format_signal_for_display
from config import MODELS_DIR

# Page config
st.set_page_config(
    page_title="RL Trading Bot - Nifty & Banknifty",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4aa, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .signal-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #3d3d5c;
    }
    .buy-signal { border-left: 4px solid #00d4aa; }
    .sell-signal { border-left: 4px solid #ff6b6b; }
    .metric-box {
        background: #2d2d44;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_components():
    """Load and cache bot components"""
    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()
    return fetcher, preprocessor


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_market_data(instrument: str, days: int = 30):
    """Fetch and cache market data"""
    fetcher, _ = load_components()
    return fetcher.fetch_historical_data(instrument, days=days, use_cache=False)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to dataframe"""
    df = df.copy()
    
    # Add indicators
    df = OscillatorCalculator().add_oscillator_features(df)
    df = MovingAverageCalculator().add_ma_features(df)
    df = VolatilityCalculator().add_volatility_features(df)
    df = PatternRecognizer().add_pattern_features(df)
    df = FibonacciCalculator().add_fibonacci_features(df)
    
    return df.dropna()


def create_candlestick_chart(df: pd.DataFrame, title: str = "Price Chart"):
    """Create interactive candlestick chart with indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(title, 'RSI', 'MACD')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    if 'bb_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                                  line=dict(color='rgba(255,255,255,0.3)', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                                  line=dict(color='rgba(255,255,255,0.3)', dash='dot'),
                                  fill='tonexty', fillcolor='rgba(255,255,255,0.05)'), row=1, col=1)
    
    # EMAs
    if 'ema_9' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ema_9'], name='EMA 9',
                                  line=dict(color='#00d4aa', width=1)), row=1, col=1)
    if 'ema_21' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ema_21'], name='EMA 21',
                                  line=dict(color='#7c3aed', width=1)), row=1, col=1)
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                                  line=dict(color='#fbbf24')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'macd' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD',
                                  line=dict(color='#00d4aa')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal_line'], name='Signal',
                                  line=dict(color='#ff6b6b')), row=3, col=1)
        colors = ['#00d4aa' if v >= 0 else '#ff6b6b' for v in df['macd_histogram']]
        fig.add_trace(go.Bar(x=df.index, y=df['macd_histogram'], name='Histogram',
                              marker_color=colors), row=3, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        height=700,
        showlegend=False,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def main():
    st.markdown('<h1 class="main-header">🤖 RL Trading Bot - Nifty & Banknifty</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    instrument = st.sidebar.selectbox("Select Instrument", ["NIFTY", "BANKNIFTY"])
    days = st.sidebar.slider("Historical Days", 7, 60, 30)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Actions")
    refresh_data = st.sidebar.button("🔄 Refresh Data", use_container_width=True)
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Fetch data
    with st.spinner(f"Loading {instrument} data..."):
        df = fetch_market_data(instrument, days)
        if not df.empty:
            df = add_all_indicators(df)
    
    if df.empty:
        st.error("Unable to fetch market data. Please check your connection.")
        return
    
    # Key metrics
    current_price = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100
    
    with col1:
        st.metric("Current Price", f"₹{current_price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    
    with col2:
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        st.metric("RSI", f"{rsi:.1f}", rsi_status)
    
    with col3:
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0
        st.metric("ATR", f"₹{atr:.2f}", "Volatility")
    
    with col4:
        trend = "Bullish 📈" if df['ma_trend'].iloc[-1] == 1 else "Bearish 📉" if 'ma_trend' in df.columns else "N/A"
        st.metric("Trend", trend)
    
    # Chart
    st.plotly_chart(create_candlestick_chart(df, f"{instrument} - 15min Chart"), use_container_width=True)
    
    # Technical Analysis Summary
    st.header("📊 Technical Analysis Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Oscillators")
        osc_data = {
            'Indicator': ['RSI', 'MACD', 'Stochastic'],
            'Value': [
                f"{df['rsi'].iloc[-1]:.1f}" if 'rsi' in df.columns else 'N/A',
                f"{df['macd'].iloc[-1]:.2f}" if 'macd' in df.columns else 'N/A',
                f"{df['stoch_k'].iloc[-1]:.1f}" if 'stoch_k' in df.columns else 'N/A'
            ],
            'Signal': [
                'Oversold' if df['rsi'].iloc[-1] < 30 else 'Overbought' if df['rsi'].iloc[-1] > 70 else 'Neutral',
                'Bullish' if df['macd_crossover'].iloc[-1] == 1 else 'Bearish' if df['macd_crossover'].iloc[-1] == -1 else 'Neutral',
                'Bullish' if df['stoch_signal'].iloc[-1] == 1 else 'Bearish' if df['stoch_signal'].iloc[-1] == -1 else 'Neutral'
            ] if all(c in df.columns for c in ['rsi', 'macd_crossover', 'stoch_signal']) else ['N/A']*3
        }
        st.dataframe(pd.DataFrame(osc_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("Moving Averages")
        ma_data = {
            'MA': ['EMA 9', 'EMA 21', 'SMA 50'],
            'Value': [
                f"₹{df['ema_9'].iloc[-1]:,.2f}" if 'ema_9' in df.columns else 'N/A',
                f"₹{df['ema_21'].iloc[-1]:,.2f}" if 'ema_21' in df.columns else 'N/A',
                f"₹{df['sma_50'].iloc[-1]:,.2f}" if 'sma_50' in df.columns else 'N/A'
            ],
            'Position': [
                'Above' if current_price > df['ema_9'].iloc[-1] else 'Below' if 'ema_9' in df.columns else 'N/A',
                'Above' if current_price > df['ema_21'].iloc[-1] else 'Below' if 'ema_21' in df.columns else 'N/A',
                'Above' if current_price > df['sma_50'].iloc[-1] else 'Below' if 'sma_50' in df.columns else 'N/A'
            ]
        }
        st.dataframe(pd.DataFrame(ma_data), hide_index=True, use_container_width=True)
    
    # Pattern detection
    st.header("🕯️ Candlestick Patterns")
    patterns = PatternRecognizer().detect_all_patterns(df)
    if patterns:
        for p in patterns:
            emoji = "🟢" if p.type.name == "BULLISH" else "🔴" if p.type.name == "BEARISH" else "⚪"
            st.info(f"{emoji} **{p.name}** - {p.description} (Strength: {p.strength:.0%})")
    else:
        st.info("No significant candlestick patterns detected")
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data source: Yahoo Finance")


if __name__ == "__main__":
    main()
