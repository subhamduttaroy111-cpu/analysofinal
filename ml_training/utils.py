"""
Utility functions for ML trading pipeline.
Includes technical indicators, SMC detection, feature extraction, and more.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta

# Setup logger
logger = logging.getLogger(__name__)


# ==================== DATA VALIDATION ====================

def validate_data(df: pd.DataFrame, timeframe: str) -> bool:
    """
    Validate data quality for a given timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: Timeframe identifier ('intraday', 'swing', 'longterm')
        
    Returns:
        True if data is valid, False otherwise
    """
    if df is None or df.empty:
        logger.warning(f"Empty DataFrame for {timeframe}")
        return False
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        logger.warning(f"Missing required columns for {timeframe}")
        return False
    
    # Check for NaN values
    if df[required_columns].isnull().any().any():
        logger.warning(f"NaN values found in {timeframe} data")
        return False
    
    # Check for zero or negative prices
    if (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
        logger.warning(f"Invalid price values in {timeframe} data")
        return False
    
    # Check high >= low
    if (df['High'] < df['Low']).any():
        logger.warning(f"High < Low found in {timeframe} data")
        return False
    
    return True


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data quality checks.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'nan_count': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'date_range': (df.index[0], df.index[-1]) if len(df) > 0 else (None, None),
        'price_outliers': 0,
        'volume_outliers': 0
    }
    
    # Check for price outliers (using IQR method)
    if 'Close' in df.columns:
        Q1 = df['Close'].quantile(0.25)
        Q3 = df['Close'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df['Close'] < (Q1 - 3 * IQR)) | (df['Close'] > (Q3 + 3 * IQR))).sum()
        quality_report['price_outliers'] = int(outliers)
    
    # Check for volume outliers
    if 'Volume' in df.columns:
        Q1 = df['Volume'].quantile(0.25)
        Q3 = df['Volume'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df['Volume'] < (Q1 - 3 * IQR)) | (df['Volume'] > (Q3 + 3 * IQR))).sum()
        quality_report['volume_outliers'] = int(outliers)
    
    return quality_report


# ==================== TECHNICAL INDICATORS ====================

def calculate_technical_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate technical indicators based on configuration.
    
    Args:
        df: DataFrame with OHLCV data
        config: Configuration dictionary with indicator parameters
        
    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()
    indicators = config['indicators']
    
    # RSI
    df['rsi'] = calculate_rsi(df['Close'], indicators['rsi_period'])
    
    # MACD
    macd, signal, histogram = calculate_macd(
        df['Close'],
        indicators['macd_fast'],
        indicators['macd_slow'],
        indicators['macd_signal']
    )
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_histogram'] = histogram
    
    # EMAs
    for period in indicators['ema_periods']:
        df[f'ema_{period}'] = calculate_ema(df['Close'], period)
    
    # ATR
    df['atr'] = calculate_atr(df, indicators['atr_period'])
    
    # Volume ratio
    df['volume_ma'] = df['Volume'].rolling(window=indicators['volume_ma']).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(
        df['Close'],
        indicators['bb_period'],
        indicators['bb_std']
    )
    
    # Price position (where price is relative to high/low range)
    df['price_position'] = (df['Close'] - df['Low'].rolling(20).min()) / \
                           (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
    
    return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal line, and Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


# ==================== SMC DETECTION ====================

def detect_order_blocks(df: pd.DataFrame, lookback_period: int = 20) -> pd.DataFrame:
    """
    Detect order blocks (last bullish/bearish candle before reversal).
    
    Args:
        df: DataFrame with OHLCV data
        lookback_period: Number of candles to look back
        
    Returns:
        DataFrame with order block information
    """
    df = df.copy()
    
    df['bullish_ob'] = 0.0
    df['bearish_ob'] = 0.0
    df['ob_distance'] = 0.0
    
    for i in range(lookback_period, len(df)):
        # Get recent data
        recent = df.iloc[max(0, i-lookback_period):i]
        
        # Find bullish order block (last green candle before price drops then rallies)
        bullish_candles = recent[recent['Close'] > recent['Open']]
        if len(bullish_candles) > 0:
            last_bullish = bullish_candles.iloc[-1]
            df.loc[df.index[i], 'bullish_ob'] = last_bullish['Low']
        
        # Find bearish order block (last red candle before price rallies then drops)
        bearish_candles = recent[recent['Close'] < recent['Open']]
        if len(bearish_candles) > 0:
            last_bearish = bearish_candles.iloc[-1]
            df.loc[df.index[i], 'bearish_ob'] = last_bearish['High']
        
        # Calculate distance to nearest order block
        current_price = df.loc[df.index[i], 'Close']
        bullish_ob = df.loc[df.index[i], 'bullish_ob']
        bearish_ob = df.loc[df.index[i], 'bearish_ob']
        
        distances = []
        if bullish_ob > 0:
            distances.append(abs(current_price - bullish_ob) / current_price)
        if bearish_ob > 0:
            distances.append(abs(current_price - bearish_ob) / current_price)
        
        if distances:
            df.loc[df.index[i], 'ob_distance'] = min(distances)
    
    return df


def find_liquidity_zones(df: pd.DataFrame, sensitivity: float = 0.001) -> pd.DataFrame:
    """
    Find liquidity zones (equal highs/lows, swing points).
    
    Args:
        df: DataFrame with OHLCV data
        sensitivity: Threshold for considering prices "equal"
        
    Returns:
        DataFrame with liquidity zone information
    """
    df = df.copy()
    
    df['liquidity_high'] = 0.0
    df['liquidity_low'] = 0.0
    df['liquidity_grab'] = 0
    
    # Find equal highs
    for i in range(2, len(df)-2):
        curr_high = df.loc[df.index[i], 'High']
        
        # Check for equal highs in nearby candles
        nearby_highs = df.loc[df.index[max(0, i-5):min(len(df), i+6)], 'High']
        equal_highs = nearby_highs[(nearby_highs / curr_high - 1).abs() < sensitivity]
        
        if len(equal_highs) >= 2:
            df.loc[df.index[i], 'liquidity_high'] = curr_high
    
    # Find equal lows
    for i in range(2, len(df)-2):
        curr_low = df.loc[df.index[i], 'Low']
        
        # Check for equal lows in nearby candles
        nearby_lows = df.loc[df.index[max(0, i-5):min(len(df), i+6)], 'Low']
        equal_lows = nearby_lows[(nearby_lows / curr_low - 1).abs() < sensitivity]
        
        if len(equal_lows) >= 2:
            df.loc[df.index[i], 'liquidity_low'] = curr_low
    
    # Detect liquidity grabs (price spikes through equal highs/lows then reverses)
    for i in range(1, len(df)):
        prev_idx = df.index[i-1]
        curr_idx = df.index[i]
        
        # Bullish liquidity grab (spike down through equal lows then up)
        if df.loc[curr_idx, 'liquidity_low'] > 0:
            if df.loc[curr_idx, 'Low'] <= df.loc[curr_idx, 'liquidity_low'] and \
               df.loc[curr_idx, 'Close'] > df.loc[prev_idx, 'Close']:
                df.loc[curr_idx, 'liquidity_grab'] = 1
        
        # Bearish liquidity grab (spike up through equal highs then down)
        if df.loc[curr_idx, 'liquidity_high'] > 0:
            if df.loc[curr_idx, 'High'] >= df.loc[curr_idx, 'liquidity_high'] and \
               df.loc[curr_idx, 'Close'] < df.loc[prev_idx, 'Close']:
                df.loc[curr_idx, 'liquidity_grab'] = -1
    
    return df


def identify_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify Fair Value Gaps (FVG) - gaps between candles.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with FVG information
    """
    df = df.copy()
    
    df['fvg_present'] = 0
    df['fvg_size'] = 0.0
    
    for i in range(2, len(df)):
        prev2_idx = df.index[i-2]
        prev1_idx = df.index[i-1]
        curr_idx = df.index[i]
        
        # Bullish FVG: gap between candle[i-2].high and candle[i].low
        if df.loc[curr_idx, 'Low'] > df.loc[prev2_idx, 'High']:
            gap_size = df.loc[curr_idx, 'Low'] - df.loc[prev2_idx, 'High']
            df.loc[curr_idx, 'fvg_present'] = 1
            df.loc[curr_idx, 'fvg_size'] = gap_size / df.loc[curr_idx, 'Close']
        
        # Bearish FVG: gap between candle[i].high and candle[i-2].low
        elif df.loc[curr_idx, 'High'] < df.loc[prev2_idx, 'Low']:
            gap_size = df.loc[prev2_idx, 'Low'] - df.loc[curr_idx, 'High']
            df.loc[curr_idx, 'fvg_present'] = -1
            df.loc[curr_idx, 'fvg_size'] = gap_size / df.loc[curr_idx, 'Close']
    
    return df


# ==================== SIGNAL OUTCOME CALCULATION ====================

def calculate_signal_outcome(
    df: pd.DataFrame,
    entry_idx: int,
    target_pct: float,
    sl_pct: float,
    max_bars: int
) -> int:
    """
    Calculate if a signal hit target or stop loss within holding period.
    
    Args:
        df: DataFrame with price data
        entry_idx: Index of entry point
        target_pct: Target profit percentage
        sl_pct: Stop loss percentage
        max_bars: Maximum holding period in candles
        
    Returns:
        1 if target hit first, 0 if SL hit or holding period expired
    """
    if entry_idx >= len(df) - 1:
        return 0
    
    entry_price = df.loc[df.index[entry_idx], 'Close']
    target_price = entry_price * (1 + target_pct / 100)
    sl_price = entry_price * (1 - sl_pct / 100)
    
    # Check future candles within holding period
    end_idx = min(entry_idx + max_bars, len(df))
    
    for i in range(entry_idx + 1, end_idx):
        high = df.loc[df.index[i], 'High']
        low = df.loc[df.index[i], 'Low']
        
        # Check if target hit
        if high >= target_price:
            return 1
        
        # Check if SL hit
        if low <= sl_price:
            return 0
    
    # Holding period expired without hitting target
    return 0


# ==================== FEATURE EXTRACTION ====================

def extract_intraday_features(df: pd.DataFrame, idx: int) -> Dict[str, float]:
    """
    Extract intraday-specific features.
    
    Args:
        df: DataFrame with technical indicators
        idx: Current index
        
    Returns:
        Dictionary of features
    """
    if idx >= len(df):
        return {}
    
    row = df.iloc[idx]
    
    # Time of day feature (0-1, where 0 = market open, 1 = market close)
    time_of_day = 0.5  # Default middle of day
    if hasattr(row.name, 'hour'):
        # NSE hours: 9:15 AM to 3:30 PM (375 minutes)
        hour = row.name.hour
        minute = row.name.minute
        minutes_since_open = (hour - 9) * 60 + (minute - 15)
        time_of_day = max(0, min(1, minutes_since_open / 375))
    
    # Calculate volatility
    if idx >= 20:
        recent_closes = df['Close'].iloc[idx-20:idx]
        volatility = recent_closes.std() / recent_closes.mean()
    else:
        volatility = 0.0
    
    # Trend strength
    if idx >= 9:
        trend_strength = (df['ema_9'].iloc[idx] - df['ema_20'].iloc[idx]) / df['ema_20'].iloc[idx]
    else:
        trend_strength = 0.0
    
    # Support/resistance proximity
    if idx >= 20:
        recent_high = df['High'].iloc[idx-20:idx].max()
        recent_low = df['Low'].iloc[idx-20:idx].min()
        range_size = recent_high - recent_low
        if range_size > 0:
            support_resistance_proximity = min(
                abs(row['Close'] - recent_high) / range_size,
                abs(row['Close'] - recent_low) / range_size
            )
        else:
            support_resistance_proximity = 0.5
    else:
        support_resistance_proximity = 0.5
    
    # Risk-reward ratio
    if 'atr' in df.columns and idx > 0:
        atr = df['atr'].iloc[idx]
        if atr > 0:
            risk_reward_ratio = (1.5 / 0.75)  # Based on config target/SL
        else:
            risk_reward_ratio = 2.0
    else:
        risk_reward_ratio = 2.0
    
    features = {
        'rsi': float(row.get('rsi', 50)),
        'macd_signal': float(row.get('macd_signal', 0)),
        'macd_histogram': float(row.get('macd_histogram', 0)),
        'ema_9': float(row.get('ema_9', row['Close'])),
        'ema_20': float(row.get('ema_20', row['Close'])),
        'ema_50': float(row.get('ema_50', row['Close'])),
        'volume_ratio': float(row.get('volume_ratio', 1.0)),
        'atr': float(row.get('atr', 0)),
        'price_position': float(row.get('price_position', 0.5)),
        'order_block_distance': float(row.get('ob_distance', 0)),
        'liquidity_grab': float(row.get('liquidity_grab', 0)),
        'fvg_present': float(row.get('fvg_present', 0)),
        'time_of_day': float(time_of_day),
        'volatility': float(volatility),
        'trend_strength': float(trend_strength),
        'support_resistance_proximity': float(support_resistance_proximity),
        'risk_reward_ratio': float(risk_reward_ratio)
    }
    
    return features


def extract_swing_features(df_1h: pd.DataFrame, df_1d: pd.DataFrame, idx: int) -> Dict[str, float]:
    """
    Extract swing trading features from hourly and daily data.
    
    Args:
        df_1h: Hourly DataFrame with technical indicators
        df_1d: Daily DataFrame for context
        idx: Current index in hourly data
        
    Returns:
        Dictionary of features
    """
    if idx >= len(df_1h):
        return {}
    
    row = df_1h.iloc[idx]
    
    # Get daily context
    current_date = row.name.date() if hasattr(row.name, 'date') else row.name
    daily_row = None
    if df_1d is not None and len(df_1d) > 0:
        try:
            daily_row = df_1d.loc[df_1d.index.date == current_date].iloc[0]
        except:
            daily_row = None
    
    # Day of week (0-6, Monday=0)
    day_of_week = row.name.weekday() if hasattr(row.name, 'weekday') else 2
    
    # Normalized day of week (0-1)
    day_of_week_norm = day_of_week / 6.0
    
    # Weekly trend (from daily data)
    weekly_trend = 0.0
    if daily_row is not None and 'ema_50' in df_1d.columns:
        weekly_trend = (daily_row['Close'] - daily_row['ema_50']) / daily_row['ema_50']
    
    # Volatility
    if idx >= 50:
        recent_closes = df_1h['Close'].iloc[idx-50:idx]
        volatility = recent_closes.std() / recent_closes.mean()
    else:
        volatility = 0.0
    
    # Trend strength
    if idx >= 20:
        trend_strength = (df_1h['ema_20'].iloc[idx] - df_1h['ema_50'].iloc[idx]) / df_1h['ema_50'].iloc[idx]
    else:
        trend_strength = 0.0
    
    # Support/resistance proximity
    if idx >= 50:
        recent_high = df_1h['High'].iloc[idx-50:idx].max()
        recent_low = df_1h['Low'].iloc[idx-50:idx].min()
        range_size = recent_high - recent_low
        if range_size > 0:
            support_resistance_proximity = min(
                abs(row['Close'] - recent_high) / range_size,
                abs(row['Close'] - recent_low) / range_size
            )
        else:
            support_resistance_proximity = 0.5
    else:
        support_resistance_proximity = 0.5
    
    features = {
        'rsi': float(row.get('rsi', 50)),
        'macd_signal': float(row.get('macd_signal', 0)),
        'macd_histogram': float(row.get('macd_histogram', 0)),
        'ema_20': float(row.get('ema_20', row['Close'])),
        'ema_50': float(row.get('ema_50', row['Close'])),
        'ema_200': float(row.get('ema_200', row['Close'])),
        'volume_ratio': float(row.get('volume_ratio', 1.0)),
        'atr': float(row.get('atr', 0)),
        'price_position': float(row.get('price_position', 0.5)),
        'order_block_distance': float(row.get('ob_distance', 0)),
        'liquidity_grab': float(row.get('liquidity_grab', 0)),
        'fvg_present': float(row.get('fvg_present', 0)),
        'day_of_week': float(day_of_week_norm),
        'volatility': float(volatility),
        'trend_strength': float(trend_strength),
        'support_resistance_proximity': float(support_resistance_proximity),
        'risk_reward_ratio': float(5.0 / 2.5),
        'weekly_trend': float(weekly_trend),
        'sector_strength': float(0.0)  # Placeholder - can be enhanced later
    }
    
    return features


def extract_longterm_features(df_1d: pd.DataFrame, df_1wk: pd.DataFrame, idx: int) -> Dict[str, float]:
    """
    Extract long-term trading features from daily and weekly data.
    
    Args:
        df_1d: Daily DataFrame with technical indicators
        df_1wk: Weekly DataFrame for context
        idx: Current index in daily data
        
    Returns:
        Dictionary of features
    """
    if idx >= len(df_1d):
        return {}
    
    row = df_1d.iloc[idx]
    
    # Quarterly trend (last 60 days)
    quarterly_trend = 0.0
    if idx >= 60:
        price_60d_ago = df_1d['Close'].iloc[idx-60]
        quarterly_trend = (row['Close'] - price_60d_ago) / price_60d_ago
    
    # Yearly high/low position
    yearly_high_low_position = 0.5
    if idx >= 252:
        yearly_high = df_1d['High'].iloc[idx-252:idx].max()
        yearly_low = df_1d['Low'].iloc[idx-252:idx].min()
        if yearly_high > yearly_low:
            yearly_high_low_position = (row['Close'] - yearly_low) / (yearly_high - yearly_low)
    
    # Monthly trend (from weekly data)
    monthly_trend = 0.0
    if df_1wk is not None and len(df_1wk) > 0:
        current_date = row.name.date() if hasattr(row.name, 'date') else row.name
        try:
            weekly_idx = df_1wk.index.get_indexer([current_date], method='nearest')[0]
            if weekly_idx >= 4:
                price_4w_ago = df_1wk['Close'].iloc[weekly_idx-4]
                monthly_trend = (df_1wk['Close'].iloc[weekly_idx] - price_4w_ago) / price_4w_ago
        except:
            monthly_trend = 0.0
    
    # Volatility
    if idx >= 100:
        recent_closes = df_1d['Close'].iloc[idx-100:idx]
        volatility = recent_closes.std() / recent_closes.mean()
    else:
        volatility = 0.0
    
    # Trend strength
    if idx >= 50:
        trend_strength = (df_1d['ema_50'].iloc[idx] - df_1d['ema_200'].iloc[idx]) / df_1d['ema_200'].iloc[idx]
    else:
        trend_strength = 0.0
    
    # Support/resistance proximity
    if idx >= 100:
        recent_high = df_1d['High'].iloc[idx-100:idx].max()
        recent_low = df_1d['Low'].iloc[idx-100:idx].min()
        range_size = recent_high - recent_low
        if range_size > 0:
            support_resistance_proximity = min(
                abs(row['Close'] - recent_high) / range_size,
                abs(row['Close'] - recent_low) / range_size
            )
        else:
            support_resistance_proximity = 0.5
    else:
        support_resistance_proximity = 0.5
    
    features = {
        'rsi': float(row.get('rsi', 50)),
        'macd_signal': float(row.get('macd_signal', 0)),
        'macd_histogram': float(row.get('macd_histogram', 0)),
        'ema_50': float(row.get('ema_50', row['Close'])),
        'ema_100': float(row.get('ema_100', row['Close'])),
        'ema_200': float(row.get('ema_200', row['Close'])),
        'volume_ratio': float(row.get('volume_ratio', 1.0)),
        'atr': float(row.get('atr', 0)),
        'price_position': float(row.get('price_position', 0.5)),
        'order_block_distance': float(row.get('ob_distance', 0)),
        'liquidity_grab': float(row.get('liquidity_grab', 0)),
        'monthly_trend': float(monthly_trend),
        'volatility': float(volatility),
        'trend_strength': float(trend_strength),
        'support_resistance_proximity': float(support_resistance_proximity),
        'risk_reward_ratio': float(15.0 / 7.5),
        'quarterly_trend': float(quarterly_trend),
        'yearly_high_low_position': float(yearly_high_low_position),
        'fundamental_score': float(0.0)  # Placeholder - can enhance with fundamental data
    }
    
    return features


# ==================== HELPER FUNCTIONS ====================

def get_higher_timeframe_context(df_lower: pd.DataFrame, df_higher: pd.DataFrame) -> pd.DataFrame:
    """
    Merge lower timeframe data with higher timeframe context.
    
    Args:
        df_lower: Lower timeframe DataFrame
        df_higher: Higher timeframe DataFrame
        
    Returns:
        Merged DataFrame
    """
    # This is a placeholder - in actual implementation, you would align timeframes
    return df_lower


if __name__ == "__main__":
    print("Utility functions loaded successfully")
    print(f"Functions available: validate_data, calculate_technical_indicators, ")
    print(f"detect_order_blocks, find_liquidity_zones, identify_fvg,")
    print(f"extract_intraday_features, extract_swing_features, extract_longterm_features")
