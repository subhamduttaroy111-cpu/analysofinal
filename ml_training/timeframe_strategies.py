"""
Timeframe-specific trading strategies.
Each strategy class implements signal detection logic for its timeframe.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class IntradayStrategy:
    """
    Intraday trading strategy (5-minute charts).
    Focus: Quick moves, order blocks, liquidity grabs, session timing.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize intraday strategy.
        
        Args:
            config: Intraday configuration dictionary
        """
        self.config = config
        self.lookback = config.get('order_block_lookback', 20)
        self.sensitivity = config.get('liquidity_sensitivity', 0.001)
    
    def detect_signal(self, df: pd.DataFrame, idx: int) -> Optional[str]:
        """
        Detect buy/sell signal for intraday trading.
        
        Args:
            df: DataFrame with technical indicators
            idx: Current index
            
        Returns:
            'BUY', 'SELL', or None
        """
        if idx < 50:  # Need enough history
            return None
        
        row = df.iloc[idx]
        
        # Basic conditions
        rsi = row.get('rsi', 50)
        macd_histogram = row.get('macd_histogram', 0)
        ema_9 = row.get('ema_9', row['Close'])
        ema_20 = row.get('ema_20', row['Close'])
        volume_ratio = row.get('volume_ratio', 1.0)
        liquidity_grab = row.get('liquidity_grab', 0)
        ob_distance = row.get('ob_distance', 1.0)
        
        # Time of day filter (avoid first and last 30 minutes)
        if hasattr(row.name, 'hour'):
            hour = row.name.hour
            minute = row.name.minute
            # Skip 9:15-9:45 and 3:00-3:30
            if (hour == 9 and minute < 45) or (hour >= 15):
                return None
        
        # BUY SIGNAL CONDITIONS
        buy_conditions = [
            rsi < 40,  # Oversold
            macd_histogram > 0,  # Bullish MACD
            ema_9 > ema_20,  # Short-term uptrend
            volume_ratio > 1.2,  # Above-average volume
            liquidity_grab == 1,  # Bullish liquidity grab
            ob_distance < 0.02  # Near order block (2%)
        ]
        
        if sum(buy_conditions) >= 4:  # At least 4 conditions met
            return 'BUY'
        
        # SELL SIGNAL CONDITIONS
        sell_conditions = [
            rsi > 60,  # Overbought
            macd_histogram < 0,  # Bearish MACD
            ema_9 < ema_20,  # Short-term downtrend
            volume_ratio > 1.2,  # Above-average volume
            liquidity_grab == -1,  # Bearish liquidity grab
            ob_distance < 0.02  # Near order block (2%)
        ]
        
        if sum(sell_conditions) >= 4:  # At least 4 conditions met
            return 'SELL'
        
        return None
    
    def calculate_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """
        Extract intraday-specific features.
        
        Args:
            df: DataFrame with technical indicators
            idx: Current index
            
        Returns:
            Dictionary of features
        """
        from utils import extract_intraday_features
        return extract_intraday_features(df, idx)


class SwingStrategy:
    """
    Swing trading strategy (1-hour + daily charts).
    Focus: Multi-day trends, larger order blocks, weekly context.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize swing strategy.
        
        Args:
            config: Swing configuration dictionary
        """
        self.config = config
        self.lookback = config.get('order_block_lookback', 100)
        self.sensitivity = config.get('liquidity_sensitivity', 0.002)
    
    def detect_signal(
        self,
        df_1h: pd.DataFrame,
        df_1d: Optional[pd.DataFrame],
        idx: int
    ) -> Optional[str]:
        """
        Detect buy/sell signal for swing trading.
        
        Args:
            df_1h: Hourly DataFrame with technical indicators
            df_1d: Daily DataFrame for context (optional)
            idx: Current index in hourly data
            
        Returns:
            'BUY', 'SELL', or None
        """
        if idx < 200:  # Need enough history
            return None
        
        row_1h = df_1h.iloc[idx]
        
        # Hourly conditions
        rsi = row_1h.get('rsi', 50)
        macd_histogram = row_1h.get('macd_histogram', 0)
        ema_20 = row_1h.get('ema_20', row_1h['Close'])
        ema_50 = row_1h.get('ema_50', row_1h['Close'])
        ema_200 = row_1h.get('ema_200', row_1h['Close'])
        volume_ratio = row_1h.get('volume_ratio', 1.0)
        liquidity_grab = row_1h.get('liquidity_grab', 0)
        ob_distance = row_1h.get('ob_distance', 1.0)
        
        # Daily context (if available)
        daily_trend = 0
        if df_1d is not None and len(df_1d) > 0:
            try:
                current_date = row_1h.name.date() if hasattr(row_1h.name, 'date') else row_1h.name
                daily_row = df_1d.loc[df_1d.index.date == current_date].iloc[0]
                
                # Check daily trend
                if 'ema_50' in df_1d.columns and 'ema_200' in df_1d.columns:
                    if daily_row['ema_50'] > daily_row['ema_200']:
                        daily_trend = 1  # Bullish daily trend
                    else:
                        daily_trend = -1  # Bearish daily trend
            except:
                daily_trend = 0
        
        # BUY SIGNAL CONDITIONS
        buy_conditions = [
            rsi < 45,  # Oversold
            macd_histogram > 0,  # Bullish MACD
            ema_20 > ema_50,  # Medium-term uptrend
            ema_50 > ema_200,  # Long-term uptrend
            volume_ratio > 1.3,  # Above-average volume
            liquidity_grab == 1,  # Bullish liquidity grab
            ob_distance < 0.03,  # Near order block (3%)
            daily_trend >= 0  # Daily not bearish
        ]
        
        if sum(buy_conditions) >= 5:  # At least 5 conditions met
            return 'BUY'
        
        # SELL SIGNAL CONDITIONS
        sell_conditions = [
            rsi > 55,  # Overbought
            macd_histogram < 0,  # Bearish MACD
            ema_20 < ema_50,  # Medium-term downtrend
            ema_50 < ema_200,  # Long-term downtrend
            volume_ratio > 1.3,  # Above-average volume
            liquidity_grab == -1,  # Bearish liquidity grab
            ob_distance < 0.03,  # Near order block (3%)
            daily_trend <= 0  # Daily not bullish
        ]
        
        if sum(sell_conditions) >= 5:  # At least 5 conditions met
            return 'SELL'
        
        return None
    
    def calculate_features(
        self,
        df_1h: pd.DataFrame,
        df_1d: Optional[pd.DataFrame],
        idx: int
    ) -> Dict[str, float]:
        """
        Extract swing-specific features.
        
        Args:
            df_1h: Hourly DataFrame with technical indicators
            df_1d: Daily DataFrame for context
            idx: Current index in hourly data
            
        Returns:
            Dictionary of features
        """
        from utils import extract_swing_features
        return extract_swing_features(df_1h, df_1d, idx)


class LongTermStrategy:
    """
    Long-term investing/position trading (daily + weekly charts).
    Focus: Major trends, monthly patterns, fundamental alignment.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize long-term strategy.
        
        Args:
            config: Long-term configuration dictionary
        """
        self.config = config
        self.lookback = config.get('order_block_lookback', 200)
        self.sensitivity = config.get('liquidity_sensitivity', 0.003)
    
    def detect_signal(
        self,
        df_1d: pd.DataFrame,
        df_1wk: Optional[pd.DataFrame],
        idx: int
    ) -> Optional[str]:
        """
        Detect buy/sell signal for long-term trading.
        
        Args:
            df_1d: Daily DataFrame with technical indicators
            df_1wk: Weekly DataFrame for context (optional)
            idx: Current index in daily data
            
        Returns:
            'BUY', 'SELL', or None
        """
        if idx < 252:  # Need at least 1 year of history
            return None
        
        row_1d = df_1d.iloc[idx]
        
        # Daily conditions
        rsi = row_1d.get('rsi', 50)
        macd_histogram = row_1d.get('macd_histogram', 0)
        ema_50 = row_1d.get('ema_50', row_1d['Close'])
        ema_100 = row_1d.get('ema_100', row_1d['Close'])
        ema_200 = row_1d.get('ema_200', row_1d['Close'])
        volume_ratio = row_1d.get('volume_ratio', 1.0)
        liquidity_grab = row_1d.get('liquidity_grab', 0)
        ob_distance = row_1d.get('ob_distance', 1.0)
        
        # Yearly position
        yearly_high = df_1d['High'].iloc[max(0, idx-252):idx].max()
        yearly_low = df_1d['Low'].iloc[max(0, idx-252):idx].min()
        yearly_position = 0.5
        if yearly_high > yearly_low:
            yearly_position = (row_1d['Close'] - yearly_low) / (yearly_high - yearly_low)
        
        # Weekly context (if available)
        weekly_trend = 0
        if df_1wk is not None and len(df_1wk) > 0:
            try:
                current_date = row_1d.name.date() if hasattr(row_1d.name, 'date') else row_1d.name
                weekly_idx = df_1wk.index.get_indexer([current_date], method='nearest')[0]
                
                if weekly_idx >= 13:  # 13 weeks = 1 quarter
                    weekly_ema_13 = df_1wk['Close'].iloc[weekly_idx-13:weekly_idx].mean()
                    if df_1wk['Close'].iloc[weekly_idx] > weekly_ema_13:
                        weekly_trend = 1
                    else:
                        weekly_trend = -1
            except:
                weekly_trend = 0
        
        # BUY SIGNAL CONDITIONS
        buy_conditions = [
            rsi < 50,  # Not overbought
            macd_histogram > 0,  # Bullish MACD
            ema_50 > ema_100,  # Medium-term uptrend
            ema_100 > ema_200,  # Long-term uptrend
            volume_ratio > 1.5,  # Significantly above-average volume
            liquidity_grab == 1,  # Bullish liquidity grab
            ob_distance < 0.05,  # Near order block (5%)
            yearly_position < 0.7,  # Not near yearly high
            weekly_trend >= 0  # Weekly not bearish
        ]
        
        if sum(buy_conditions) >= 6:  # At least 6 conditions met
            return 'BUY'
        
        # SELL SIGNAL CONDITIONS
        sell_conditions = [
            rsi > 50,  # Not oversold
            macd_histogram < 0,  # Bearish MACD
            ema_50 < ema_100,  # Medium-term downtrend
            ema_100 < ema_200,  # Long-term downtrend
            volume_ratio > 1.5,  # Significantly above-average volume
            liquidity_grab == -1,  # Bearish liquidity grab
            ob_distance < 0.05,  # Near order block (5%)
            yearly_position > 0.3,  # Not near yearly low
            weekly_trend <= 0  # Weekly not bullish
        ]
        
        if sum(sell_conditions) >= 6:  # At least 6 conditions met
            return 'SELL'
        
        return None
    
    def calculate_features(
        self,
        df_1d: pd.DataFrame,
        df_1wk: Optional[pd.DataFrame],
        idx: int
    ) -> Dict[str, float]:
        """
        Extract long-term features.
        
        Args:
            df_1d: Daily DataFrame with technical indicators
            df_1wk: Weekly DataFrame for context
            idx: Current index in daily data
            
        Returns:
            Dictionary of features
        """
        from utils import extract_longterm_features
        return extract_longterm_features(df_1d, df_1wk, idx)


if __name__ == "__main__":
    print("Timeframe strategies loaded successfully")
    print("Available strategies: IntradayStrategy, SwingStrategy, LongTermStrategy")
