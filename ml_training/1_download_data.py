"""
Data Download Script - Downloads market data for all three timeframes.
Downloads sequentially with rate limiting to avoid yfinance API blocks.
"""

import yfinance as yf
import pandas as pd
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_training.config import (
    STOCK_LIST,
    INTRADAY_CONFIG,
    SWING_CONFIG,
    LONGTERM_CONFIG,
    PATHS,
    DOWNLOAD_CONFIG
)

# Setup logging
os.makedirs(PATHS['logs'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs'], 'download_errors.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Handles sequential data downloads for all timeframes."""
    
    def __init__(self):
        self.failed_downloads = []
        self.stats = {
            'intraday': {'success': 0, 'failed': 0},
            'swing': {'success': 0, 'failed': 0},
            'longterm': {'success': 0, 'failed': 0}
        }
    
    def download_with_retry(
        self,
        symbol: str,
        period: str,
        interval: str,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Download data with retry logic and exponential backoff.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            period: Time period (e.g., '60d', '2y', '5y')
            interval: Data interval (e.g., '5m', '1h', '1d')
            max_retries: Maximum retry attempts
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        for attempt in range(max_retries):
            try:
                # Download data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df is None or df.empty:
                    logger.warning(f"Empty data for {symbol} ({interval})")
                    time.sleep(DOWNLOAD_CONFIG['retry_delay'] * (attempt + 1))
                    continue
                
                # Keep only OHLCV columns to save space
                columns_to_keep = DOWNLOAD_CONFIG['columns_to_keep']
                df = df[columns_to_keep]
                
                # Remove NaN rows
                df = df.dropna()
                
                # Remove duplicates
                df = df[~df.index.duplicated(keep='first')]
                
                if len(df) < 10:  # Too little data
                    logger.warning(f"Insufficient data for {symbol} ({interval}): {len(df)} rows")
                    time.sleep(DOWNLOAD_CONFIG['retry_delay'] * (attempt + 1))
                    continue
                
                return df
                
            except Exception as e:
                logger.error(f"Error downloading {symbol} ({interval}), attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(DOWNLOAD_CONFIG['retry_delay'] * (attempt + 1))
                else:
                    return None
        
        return None
    
    def download_intraday(self, symbol: str) -> bool:
        """
        Download intraday data (5-minute candles, 60 days).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if successful, False otherwise
        """
        try:
            df = self.download_with_retry(
                symbol,
                INTRADAY_CONFIG['period'],
                INTRADAY_CONFIG['interval']
            )
            
            if df is not None:
                # Save to CSV
                filename = f"{symbol.replace('.NS', '')}_5m.csv"
                filepath = os.path.join(PATHS['raw_intraday'], filename)
                df.to_csv(filepath)
                
                logger.info(f"✓ Intraday: {symbol} ({len(df)} rows)")
                self.stats['intraday']['success'] += 1
                return True
            else:
                self.failed_downloads.append((symbol, 'intraday', 'No data'))
                self.stats['intraday']['failed'] += 1
                return False
                
        except Exception as e:
            logger.error(f"✗ Intraday: {symbol} - {str(e)}")
            self.failed_downloads.append((symbol, 'intraday', str(e)))
            self.stats['intraday']['failed'] += 1
            return False
    
    def download_swing(self, symbol: str) -> Tuple[bool, bool]:
        """
        Download swing data (1-hour + daily candles, 2 years).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (hourly_success, daily_success)
        """
        success_1h = False
        success_1d = False
        
        # Download 1-hour data
        try:
            df_1h = self.download_with_retry(
                symbol,
                SWING_CONFIG['period'],
                SWING_CONFIG['interval']
            )
            
            if df_1h is not None:
                filename = f"{symbol.replace('.NS', '')}_1h.csv"
                filepath = os.path.join(PATHS['raw_swing'], filename)
                df_1h.to_csv(filepath)
                
                logger.info(f"✓ Swing 1h: {symbol} ({len(df_1h)} rows)")
                success_1h = True
            else:
                self.failed_downloads.append((symbol, 'swing_1h', 'No data'))
                
        except Exception as e:
            logger.error(f"✗ Swing 1h: {symbol} - {str(e)}")
            self.failed_downloads.append((symbol, 'swing_1h', str(e)))
        
        # Small delay between downloads
        time.sleep(0.3)
        
        # Download daily data for context
        try:
            df_1d = self.download_with_retry(
                symbol,
                SWING_CONFIG['period'],
                SWING_CONFIG['timeframe_secondary']
            )
            
            if df_1d is not None:
                filename = f"{symbol.replace('.NS', '')}_1d_swing.csv"
                filepath = os.path.join(PATHS['raw_swing'], filename)
                df_1d.to_csv(filepath)
                
                logger.info(f"✓ Swing 1d: {symbol} ({len(df_1d)} rows)")
                success_1d = True
            else:
                self.failed_downloads.append((symbol, 'swing_1d', 'No data'))
                
        except Exception as e:
            logger.error(f"✗ Swing 1d: {symbol} - {str(e)}")
            self.failed_downloads.append((symbol, 'swing_1d', str(e)))
        
        # Update stats
        if success_1h and success_1d:
            self.stats['swing']['success'] += 1
        else:
            self.stats['swing']['failed'] += 1
        
        return success_1h, success_1d
    
    def download_longterm(self, symbol: str) -> Tuple[bool, bool]:
        """
        Download long-term data (daily + weekly candles, 5 years).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (daily_success, weekly_success)
        """
        success_1d = False
        success_1wk = False
        
        # Download daily data
        try:
            df_1d = self.download_with_retry(
                symbol,
                LONGTERM_CONFIG['period'],
                LONGTERM_CONFIG['interval']
            )
            
            if df_1d is not None:
                filename = f"{symbol.replace('.NS', '')}_1d.csv"
                filepath = os.path.join(PATHS['raw_longterm'], filename)
                df_1d.to_csv(filepath)
                
                logger.info(f"✓ Long-term 1d: {symbol} ({len(df_1d)} rows)")
                success_1d = True
            else:
                self.failed_downloads.append((symbol, 'longterm_1d', 'No data'))
                
        except Exception as e:
            logger.error(f"✗ Long-term 1d: {symbol} - {str(e)}")
            self.failed_downloads.append((symbol, 'longterm_1d', str(e)))
        
        # Small delay between downloads
        time.sleep(0.3)
        
        # Download weekly data for context
        try:
            df_1wk = self.download_with_retry(
                symbol,
                LONGTERM_CONFIG['period'],
                LONGTERM_CONFIG['timeframe_secondary']
            )
            
            if df_1wk is not None:
                filename = f"{symbol.replace('.NS', '')}_1wk.csv"
                filepath = os.path.join(PATHS['raw_longterm'], filename)
                df_1wk.to_csv(filepath)
                
                logger.info(f"✓ Long-term 1wk: {symbol} ({len(df_1wk)} rows)")
                success_1wk = True
            else:
                self.failed_downloads.append((symbol, 'longterm_1wk', 'No data'))
                
        except Exception as e:
            logger.error(f"✗ Long-term 1wk: {symbol} - {str(e)}")
            self.failed_downloads.append((symbol, 'longterm_1wk', str(e)))
        
        # Update stats
        if success_1d and success_1wk:
            self.stats['longterm']['success'] += 1
        else:
            self.stats['longterm']['failed'] += 1
        
        return success_1d, success_1wk
    
    def download_all(self, symbols: List[str]):
        """
        Download data for all symbols sequentially.
        
        Args:
            symbols: List of stock symbols
        """
        total_stocks = len(symbols)
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"Starting data download for {total_stocks} stocks")
        print(f"{'='*70}\n")
        
        for idx, symbol in enumerate(symbols, 1):
            print(f"\n[{idx}/{total_stocks}] Processing: {symbol}")
            print(f"{'-'*70}")
            
            # Download all timeframes for this stock
            intraday_ok = self.download_intraday(symbol)
            time.sleep(DOWNLOAD_CONFIG['request_delay'])
            
            swing_ok = self.download_swing(symbol)
            time.sleep(DOWNLOAD_CONFIG['request_delay'])
            
            longterm_ok = self.download_longterm(symbol)
            time.sleep(DOWNLOAD_CONFIG['request_delay'])
            
            # Display progress
            status = []
            if intraday_ok:
                status.append("Intraday ✓")
            else:
                status.append("Intraday ✗")
            
            if swing_ok[0] and swing_ok[1]:
                status.append("Swing ✓")
            else:
                status.append("Swing ✗")
            
            if longterm_ok[0] and longterm_ok[1]:
                status.append("Long-term ✓")
            else:
                status.append("Long-term ✗")
            
            progress_pct = (idx / total_stocks) * 100
            print(f"Status: [{' | '.join(status)}] - Progress: {progress_pct:.1f}%")
            
            # Estimate remaining time
            elapsed = time.time() - start_time
            avg_time_per_stock = elapsed / idx
            remaining_stocks = total_stocks - idx
            estimated_remaining = avg_time_per_stock * remaining_stocks
            
            if idx < total_stocks:
                print(f"Estimated remaining time: {estimated_remaining/60:.1f} minutes")
        
        # Final summary
        self.print_summary(start_time)
    
    def print_summary(self, start_time: float):
        """Print download summary."""
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"DOWNLOAD COMPLETE")
        print(f"{'='*70}")
        print(f"\nResults:")
        print(f"  ✓ Intraday:   {self.stats['intraday']['success']:3d}/{len(STOCK_LIST)} stocks "
              f"({self.stats['intraday']['success']/len(STOCK_LIST)*100:.1f}%)")
        print(f"  ✓ Swing:      {self.stats['swing']['success']:3d}/{len(STOCK_LIST)} stocks "
              f"({self.stats['swing']['success']/len(STOCK_LIST)*100:.1f}%)")
        print(f"  ✓ Long-term:  {self.stats['longterm']['success']:3d}/{len(STOCK_LIST)} stocks "
              f"({self.stats['longterm']['success']/len(STOCK_LIST)*100:.1f}%)")
        
        print(f"\nExecution time: {elapsed/60:.1f} minutes")
        
        if self.failed_downloads:
            print(f"\nFailed downloads: {len(self.failed_downloads)}")
            print(f"See logs/download_errors.log for details")
            
            # Write failed downloads to file
            with open(os.path.join(PATHS['logs'], 'download_failures.txt'), 'w') as f:
                f.write("Failed Downloads:\n")
                f.write("="*70 + "\n")
                for symbol, timeframe, error in self.failed_downloads:
                    f.write(f"{symbol} - {timeframe}: {error}\n")
        else:
            print("\n✓ All downloads successful!")
        
        print(f"\nData saved to:")
        print(f"  - {PATHS['raw_intraday']}")
        print(f"  - {PATHS['raw_swing']}")
        print(f"  - {PATHS['raw_longterm']}")
        print(f"\n{'='*70}\n")


def main():
    """Main execution function."""
    # Create directories if they don't exist
    for path in [PATHS['raw_intraday'], PATHS['raw_swing'], PATHS['raw_longterm']]:
        os.makedirs(path, exist_ok=True)
    
    # Validate stock list
    if not STOCK_LIST:
        logger.error("Stock list is empty! Please check config.py")
        return
    
    logger.info(f"Loaded {len(STOCK_LIST)} stocks")
    
    # Create downloader and start
    downloader = DataDownloader()
    downloader.download_all(STOCK_LIST)


if __name__ == "__main__":
    main()
