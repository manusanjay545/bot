"""
Data preprocessor for preparing market data for the RL agent
Handles normalization, feature engineering, and data windowing
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses market data for the RL trading environment
    """
    
    def __init__(
        self,
        normalization: str = 'minmax',
        window_size: int = 20
    ):
        """
        Initialize the preprocessor
        
        Args:
            normalization: 'minmax', 'standard', or 'none'
            window_size: Number of candles to include in each state
        """
        self.normalization = normalization
        self.window_size = window_size
        
        if normalization == 'minmax':
            self.scaler = MinMaxScaler()
        elif normalization == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = None
            
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the scaler on the training data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            self
        """
        if self.scaler is not None:
            # Fit on price and volume columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.scaler.fit(df[numeric_cols])
            self.is_fitted = True
            
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted scaler
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        df_transformed = df.copy()
        
        if self.scaler is not None and self.is_fitted:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_transformed[numeric_cols] = self.scaler.transform(df[numeric_cols])
            
        return df_transformed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step
        """
        self.fit(df)
        return self.transform(df)
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add return-based features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional return features
        """
        df = df.copy()
        
        # Simple returns
        df['returns'] = df['close'].pct_change()
        
        # Log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Cumulative returns over different periods
        for period in [5, 10, 20]:
            df[f'cum_returns_{period}'] = df['close'].pct_change(period)
            
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional price features
        """
        df = df.copy()
        
        # High-Low range
        df['hl_range'] = df['high'] - df['low']
        df['hl_range_pct'] = df['hl_range'] / df['close']
        
        # Close position within range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Gap (open vs previous close)
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1)
        
        # Body size (absolute and relative)
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_size_pct'] = df['body_size'] / df['close']
        
        # Upper and lower shadows
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Candle direction
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional volume features
        """
        df = df.copy()
        
        # Volume change
        df['volume_change'] = df['volume'].pct_change()
        
        # Volume moving average
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # Relative volume
        df['relative_volume'] = df['volume'] / df['volume_sma_20']
        
        # Volume-price relationship (On-Balance Volume style)
        df['volume_direction'] = df['volume'] * np.sign(df['close'] - df['close'].shift(1))
        
        return df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling
        
        Args:
            df: DataFrame with features
            feature_cols: Columns to include in sequences
            
        Returns:
            Tuple of (X, indices) where X is the sequence array
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
        data = df[feature_cols].values
        
        sequences = []
        indices = []
        
        for i in range(self.window_size, len(data)):
            sequences.append(data[i - self.window_size:i])
            indices.append(df.index[i])
            
        return np.array(sequences), np.array(indices)
    
    def prepare_for_training(
        self,
        df: pd.DataFrame,
        add_features: bool = True
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for training
        
        Args:
            df: Raw OHLCV DataFrame
            add_features: Whether to add engineered features
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        if add_features:
            df = self.add_returns(df)
            df = self.add_price_features(df)
            df = self.add_volume_features(df)
            
        # Drop rows with NaN values
        df = df.dropna()
        
        # Normalize if enabled
        if self.scaler is not None:
            df = self.fit_transform(df)
            
        logger.info(f"Preprocessed data shape: {df.shape}")
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of numeric feature column names"""
        return df.select_dtypes(include=[np.number]).columns.tolist()


if __name__ == "__main__":
    # Test the preprocessor
    from fetcher import DataFetcher
    
    fetcher = DataFetcher()
    df = fetcher.fetch_historical_data('NIFTY', days=30)
    
    preprocessor = DataPreprocessor(normalization='minmax', window_size=20)
    df_processed = preprocessor.prepare_for_training(df)
    
    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {df_processed.shape}")
    print(f"\nFeatures: {preprocessor.get_feature_names(df_processed)}")
    print(f"\nSample data:")
    print(df_processed.tail())
