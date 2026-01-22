"""
Market data fetcher using Yahoo Finance
Supports both historical and real-time data for Nifty and Banknifty
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INSTRUMENTS, TIMEFRAME, LOOKBACK_DAYS, DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches market data for Nifty and Banknifty from Yahoo Finance
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the data fetcher
        
        Args:
            cache_enabled: Whether to cache data locally
        """
        self.instruments = INSTRUMENTS
        self.timeframe = TIMEFRAME
        self.cache_enabled = cache_enabled
        self.cache_dir = DATA_DIR
        
    def fetch_historical_data(
        self,
        instrument: str = 'NIFTY',
        days: int = LOOKBACK_DAYS,
        interval: str = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data
        
        Args:
            instrument: 'NIFTY' or 'BANKNIFTY'
            days: Number of days of historical data
            interval: Candle interval (default: from config)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        if interval is None:
            interval = self.timeframe
            
        symbol = self.instruments.get(instrument.upper())
        if not symbol:
            raise ValueError(f"Unknown instrument: {instrument}")
        
        # Check cache first
        cache_file = self._get_cache_path(instrument, interval)
        if use_cache and self.cache_enabled and os.path.exists(cache_file):
            cached_data = self._load_from_cache(cache_file)
            if cached_data is not None:
                # Check if cache is recent enough
                if self._is_cache_valid(cached_data):
                    logger.info(f"Loaded {instrument} data from cache")
                    return cached_data
        
        # Fetch from Yahoo Finance
        logger.info(f"Fetching {instrument} data from Yahoo Finance...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True
            )
            
            if df.empty:
                logger.warning(f"No data received for {instrument}")
                return pd.DataFrame()
            
            # Clean and standardize column names
            df = self._clean_data(df)
            
            # Cache the data
            if self.cache_enabled:
                self._save_to_cache(df, cache_file)
                
            logger.info(f"Fetched {len(df)} candles for {instrument}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {instrument}: {e}")
            # Try to return cached data as fallback
            if os.path.exists(cache_file):
                return self._load_from_cache(cache_file)
            return pd.DataFrame()
    
    def fetch_multiple_instruments(
        self,
        instruments: List[str] = None,
        days: int = LOOKBACK_DAYS
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple instruments
        
        Args:
            instruments: List of instrument names
            days: Number of days of historical data
            
        Returns:
            Dictionary mapping instrument names to DataFrames
        """
        if instruments is None:
            instruments = list(self.instruments.keys())
            
        data = {}
        for instrument in instruments:
            data[instrument] = self.fetch_historical_data(instrument, days)
            
        return data
    
    def get_latest_candle(
        self,
        instrument: str = 'NIFTY'
    ) -> Optional[pd.Series]:
        """
        Get the most recent candle
        
        Args:
            instrument: Instrument name
            
        Returns:
            Series with latest OHLCV data
        """
        df = self.fetch_historical_data(instrument, days=7, use_cache=False)
        if not df.empty:
            return df.iloc[-1]
        return None
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the data
        """
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
                
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Sort by index (datetime)
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    def _get_cache_path(self, instrument: str, interval: str) -> str:
        """Get the cache file path for an instrument"""
        filename = f"{instrument.lower()}_{interval.replace(' ', '_')}.parquet"
        return os.path.join(self.cache_dir, filename)
    
    def _save_to_cache(self, df: pd.DataFrame, filepath: str):
        """Save data to cache"""
        try:
            df.to_parquet(filepath)
            logger.debug(f"Saved data to cache: {filepath}")
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _load_from_cache(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return None
    
    def _is_cache_valid(self, df: pd.DataFrame, max_age_hours: int = 1) -> bool:
        """Check if cached data is still valid"""
        if df.empty:
            return False
            
        last_timestamp = df.index[-1]
        if isinstance(last_timestamp, pd.Timestamp):
            last_timestamp = last_timestamp.to_pydatetime()
            
        # Make both timestamps timezone-naive for comparison
        if last_timestamp.tzinfo is not None:
            last_timestamp = last_timestamp.replace(tzinfo=None)
            
        age = datetime.now() - last_timestamp
        return age.total_seconds() < max_age_hours * 3600


def fetch_nifty_data(days: int = 60) -> pd.DataFrame:
    """Convenience function to fetch Nifty data"""
    fetcher = DataFetcher()
    return fetcher.fetch_historical_data('NIFTY', days)


def fetch_banknifty_data(days: int = 60) -> pd.DataFrame:
    """Convenience function to fetch Banknifty data"""
    fetcher = DataFetcher()
    return fetcher.fetch_historical_data('BANKNIFTY', days)


if __name__ == "__main__":
    # Test the fetcher
    fetcher = DataFetcher()
    
    print("Fetching Nifty data...")
    nifty_data = fetcher.fetch_historical_data('NIFTY', days=30)
    print(f"Nifty data shape: {nifty_data.shape}")
    print(nifty_data.tail())
    
    print("\nFetching Banknifty data...")
    banknifty_data = fetcher.fetch_historical_data('BANKNIFTY', days=30)
    print(f"Banknifty data shape: {banknifty_data.shape}")
    print(banknifty_data.tail())
