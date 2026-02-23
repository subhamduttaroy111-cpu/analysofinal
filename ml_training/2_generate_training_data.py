"""
Training Data Generation Script - Processes raw data into labeled training datasets.
Generates separate datasets for Intraday, Swing, and Long-term timeframes.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_training.config import (
    INTRADAY_CONFIG,
    SWING_CONFIG,
    LONGTERM_CONFIG,
    PATHS
)
from ml_training.utils import (
    validate_data,
    calculate_technical_indicators,
    detect_order_blocks,
    find_liquidity_zones,
    identify_fvg,
    calculate_signal_outcome,
    extract_intraday_features,
    extract_swing_features,
    extract_longterm_features
)
from ml_training.timeframe_strategies import (
    IntradayStrategy,
    SwingStrategy,
    LongTermStrategy
)

# Setup logging
os.makedirs(PATHS['logs'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs'], 'training_data_generation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """Generates labeled training datasets for all timeframes."""
    
    def __init__(self):
        self.intraday_strategy = IntradayStrategy(INTRADAY_CONFIG)
        self.swing_strategy = SwingStrategy(SWING_CONFIG)
        self.longterm_strategy = LongTermStrategy(LONGTERM_CONFIG)
        
        self.stats = {
            'intraday': {'signals': 0, 'success': 0, 'failure': 0},
            'swing': {'signals': 0, 'success': 0, 'failure': 0},
            'longterm': {'signals': 0, 'success': 0, 'failure': 0}
        }
    
    def process_intraday_file(self, filepath: str) -> List[Dict]:
        """
        Process a single intraday file and extract training examples.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            List of training examples
        """
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            if not validate_data(df, 'intraday'):
                return []
            
            # Calculate technical indicators
            df = calculate_technical_indicators(df, INTRADAY_CONFIG)
            
            # Detect SMC patterns
            df = detect_order_blocks(df, INTRADAY_CONFIG['order_block_lookback'])
            df = find_liquidity_zones(df, INTRADAY_CONFIG['liquidity_sensitivity'])
            df = identify_fvg(df)
            
            # Extract training examples
            examples = []
            
            for idx in range(50, len(df) - 50):  # Need buffer for future lookback
                # Check if strategy detects a signal
                signal = self.intraday_strategy.detect_signal(df, idx)
                
                if signal in ['BUY', 'SELL']:
                    # Extract features
                    features = extract_intraday_features(df, idx)
                    
                    if not features:
                        continue
                    
                        # Calculate outcome
                    # Intraday (5m): 1 day = 6.25 hours = 75 bars
                    holding_period = INTRADAY_CONFIG['holding_period']
                    max_bars = int(holding_period[:-1]) * 75 if holding_period.endswith('d') else 75
                    
                    outcome = calculate_signal_outcome(
                        df,
                        idx,
                        INTRADAY_CONFIG['target_pct'],
                        INTRADAY_CONFIG['sl_pct'],
                        max_bars
                    )
                    
                    # Create example
                    example = {
                        **features,
                        'signal_type': signal,
                        'label': outcome
                    }
                    
                    examples.append(example)
                    
                    # Update stats
                    self.stats['intraday']['signals'] += 1
                    if outcome == 1:
                        self.stats['intraday']['success'] += 1
                    else:
                        self.stats['intraday']['failure'] += 1
            
            return examples
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            return []
    
    def generate_intraday_data(self) -> pd.DataFrame:
        """
        Generate intraday training dataset from all intraday files.
        
        Returns:
            DataFrame with training data
        """
        print(f"\n{'='*70}")
        print("GENERATING INTRADAY TRAINING DATA")
        print(f"{'='*70}\n")
        
        all_examples = []
        files = [f for f in os.listdir(PATHS['raw_intraday']) if f.endswith('.csv')]
        
        for filename in tqdm(files, desc="Processing intraday files"):
            filepath = os.path.join(PATHS['raw_intraday'], filename)
            examples = self.process_intraday_file(filepath)
            all_examples.extend(examples)
        
        if not all_examples:
            logger.warning("No intraday training examples generated!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_examples)
        
        logger.info(f"Generated {len(df)} intraday examples")
        logger.info(f"Success rate: {self.stats['intraday']['success']/max(1, self.stats['intraday']['signals'])*100:.1f}%")
        
        # Balance dataset
        df = self.balance_dataset(df, 'intraday')
        
        return df
    
    def process_swing_file(self, filepath_1h: str, filepath_1d: str) -> List[Dict]:
        """
        Process swing files (1h + 1d) and extract training examples.
        
        Args:
            filepath_1h: Path to 1-hour CSV
            filepath_1d: Path to daily CSV
            
        Returns:
            List of training examples
        """
        try:
            df_1h = pd.read_csv(filepath_1h, index_col=0, parse_dates=True)
            df_1d = None
            
            if os.path.exists(filepath_1d):
                df_1d = pd.read_csv(filepath_1d, index_col=0, parse_dates=True)
                if not validate_data(df_1d, 'swing_daily'):
                    df_1d = None
            
            if not validate_data(df_1h, 'swing'):
                return []
            
            # Calculate technical indicators
            df_1h = calculate_technical_indicators(df_1h, SWING_CONFIG)
            if df_1d is not None:
                df_1d = calculate_technical_indicators(df_1d, SWING_CONFIG)
            
            # Detect SMC patterns
            df_1h = detect_order_blocks(df_1h, SWING_CONFIG['order_block_lookback'])
            df_1h = find_liquidity_zones(df_1h, SWING_CONFIG['liquidity_sensitivity'])
            df_1h = identify_fvg(df_1h)
            
            # Extract training examples
            examples = []
            
            for idx in range(200, len(df_1h) - 100):  # Need buffer
                # Check if strategy detects a signal
                signal = self.swing_strategy.detect_signal(df_1h, df_1d, idx)
                
                if signal in ['BUY', 'SELL']:
                    # Extract features
                    features = extract_swing_features(df_1h, df_1d, idx)
                    
                    if not features:
                        continue
                    
                    # Calculate outcome
                    # Swing (1h): 1 day = ~7 trading hours = 7 bars
                    holding_period = SWING_CONFIG['holding_period']
                    max_bars = int(holding_period[:-1]) * 7 if holding_period.endswith('d') else 100
                    
                    outcome = calculate_signal_outcome(
                        df_1h,
                        idx,
                        SWING_CONFIG['target_pct'],
                        SWING_CONFIG['sl_pct'],
                        max_bars
                    )
                    
                    # Create example
                    example = {
                        **features,
                        'signal_type': signal,
                        'label': outcome
                    }
                    
                    examples.append(example)
                    
                    # Update stats
                    self.stats['swing']['signals'] += 1
                    if outcome == 1:
                        self.stats['swing']['success'] += 1
                    else:
                        self.stats['swing']['failure'] += 1
            
            return examples
            
        except Exception as e:
            logger.error(f"Error processing swing files: {str(e)}")
            return []
    
    def generate_swing_data(self) -> pd.DataFrame:
        """
        Generate swing trading dataset from all swing files.
        
        Returns:
            DataFrame with training data
        """
        print(f"\n{'='*70}")
        print("GENERATING SWING TRAINING DATA")
        print(f"{'='*70}\n")
        
        all_examples = []
        files_1h = [f for f in os.listdir(PATHS['raw_swing']) if f.endswith('_1h.csv')]
        
        for filename_1h in tqdm(files_1h, desc="Processing swing files"):
            symbol = filename_1h.replace('_1h.csv', '')
            filepath_1h = os.path.join(PATHS['raw_swing'], filename_1h)
            filepath_1d = os.path.join(PATHS['raw_swing'], f'{symbol}_1d_swing.csv')
            
            examples = self.process_swing_file(filepath_1h, filepath_1d)
            all_examples.extend(examples)
        
        if not all_examples:
            logger.warning("No swing training examples generated!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_examples)
        
        logger.info(f"Generated {len(df)} swing examples")
        logger.info(f"Success rate: {self.stats['swing']['success']/max(1, self.stats['swing']['signals'])*100:.1f}%")
        
        # Balance dataset
        df = self.balance_dataset(df, 'swing')
        
        return df
    
    def process_longterm_file(self, filepath_1d: str, filepath_1wk: str) -> List[Dict]:
        """
        Process long-term files (daily + weekly) and extract training examples.
        
        Args:
            filepath_1d: Path to daily CSV
            filepath_1wk: Path to weekly CSV
            
        Returns:
            List of training examples
        """
        try:
            df_1d = pd.read_csv(filepath_1d, index_col=0, parse_dates=True)
            df_1wk = None
            
            if os.path.exists(filepath_1wk):
                df_1wk = pd.read_csv(filepath_1wk, index_col=0, parse_dates=True)
                if not validate_data(df_1wk, 'longterm_weekly'):
                    df_1wk = None
            
            if not validate_data(df_1d, 'longterm'):
                return []
            
            # Calculate technical indicators
            df_1d = calculate_technical_indicators(df_1d, LONGTERM_CONFIG)
            if df_1wk is not None:
                df_1wk = calculate_technical_indicators(df_1wk, LONGTERM_CONFIG)
            
            # Detect SMC patterns
            df_1d = detect_order_blocks(df_1d, LONGTERM_CONFIG['order_block_lookback'])
            df_1d = find_liquidity_zones(df_1d, LONGTERM_CONFIG['liquidity_sensitivity'])
            df_1d = identify_fvg(df_1d)
            
            # Extract training examples
            examples = []
            
            for idx in range(252, len(df_1d) - 252):  # Need 1 year buffer on each side
                # Check if strategy detects a signal
                signal = self.longterm_strategy.detect_signal(df_1d, df_1wk, idx)
                
                if signal in ['BUY', 'SELL']:
                    # Extract features
                    features = extract_longterm_features(df_1d, df_1wk, idx)
                    
                    if not features:
                        continue
                    
                    # Calculate outcome
                    # Long-term (1d): 1 day = 1 bar
                    holding_period = LONGTERM_CONFIG['holding_period']
                    max_bars = int(holding_period[:-1]) if holding_period.endswith('d') else 365
                    
                    outcome = calculate_signal_outcome(
                        df_1d,
                        idx,
                        LONGTERM_CONFIG['target_pct'],
                        LONGTERM_CONFIG['sl_pct'],
                        max_bars
                    )
                    
                    # Create example
                    example = {
                        **features,
                        'signal_type': signal,
                        'label': outcome
                    }
                    
                    examples.append(example)
                    
                    # Update stats
                    self.stats['longterm']['signals'] += 1
                    if outcome == 1:
                        self.stats['longterm']['success'] += 1
                    else:
                        self.stats['longterm']['failure'] += 1
            
            return examples
            
        except Exception as e:
            logger.error(f"Error processing long-term files: {str(e)}")
            return []
    
    def generate_longterm_data(self) -> pd.DataFrame:
        """
        Generate long-term trading dataset from all long-term files.
        
        Returns:
            DataFrame with training data
        """
        print(f"\n{'='*70}")
        print("GENERATING LONG-TERM TRAINING DATA")
        print(f"{'='*70}\n")
        
        all_examples = []
        files_1d = [f for f in os.listdir(PATHS['raw_longterm']) if f.endswith('_1d.csv')]
        
        for filename_1d in tqdm(files_1d, desc="Processing long-term files"):
            symbol = filename_1d.replace('_1d.csv', '')
            filepath_1d = os.path.join(PATHS['raw_longterm'], filename_1d)
            filepath_1wk = os.path.join(PATHS['raw_longterm'], f'{symbol}_1wk.csv')
            
            examples = self.process_longterm_file(filepath_1d, filepath_1wk)
            all_examples.extend(examples)
        
        if not all_examples:
            logger.warning("No long-term training examples generated!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_examples)
        
        logger.info(f"Generated {len(df)} long-term examples")
        logger.info(f"Success rate: {self.stats['longterm']['success']/max(1, self.stats['longterm']['signals'])*100:.1f}%")
        
        # Balance dataset
        df = self.balance_dataset(df, 'longterm')
        
        return df
    
    def balance_dataset(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Balance dataset to have reasonable success rate (55-60%).
        
        Args:
            df: DataFrame with training data
            timeframe: Timeframe identifier
            
        Returns:
            Balanced DataFrame
        """
        if df.empty:
            return df
        
        # Count successes and failures
        success_count = (df['label'] == 1).sum()
        failure_count = (df['label'] == 0).sum()
        
        logger.info(f"{timeframe.capitalize()} - Before balancing: {success_count} successes, {failure_count} failures")
        
        # Target: 55-60% success rate
        target_ratio = 0.575  # 57.5% success
        
        if failure_count == 0:
            return df
        
        current_ratio = success_count / (success_count + failure_count)
        
        if current_ratio > target_ratio + 0.05:  # Too many successes
            # Downsample successes
            target_success = int(failure_count * target_ratio / (1 - target_ratio))
            df_success = df[df['label'] == 1].sample(n=min(target_success, success_count), random_state=42)
            df_failure = df[df['label'] == 0]
            df = pd.concat([df_success, df_failure]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        elif current_ratio < target_ratio - 0.05:  # Too many failures
            # Downsample failures
            target_failure = int(success_count * (1 - target_ratio) / target_ratio)
            df_success = df[df['label'] == 1]
            df_failure = df[df['label'] == 0].sample(n=min(target_failure, failure_count), random_state=42)
            df = pd.concat([df_success, df_failure]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Final counts
        final_success = (df['label'] == 1).sum()
        final_failure = (df['label'] == 0).sum()
        final_ratio = final_success / (final_success + final_failure) if (final_success + final_failure) > 0 else 0
        
        logger.info(f"{timeframe.capitalize()} - After balancing: {final_success} successes, {final_failure} failures ({final_ratio*100:.1f}% success rate)")
        
        return df
    
    def print_summary(self):
        """Print generation summary."""
        print(f"\n{'='*70}")
        print("TRAINING DATA GENERATION COMPLETE")
        print(f"{'='*70}\n")
        
        for timeframe in ['intraday', 'swing', 'longterm']:
            stats = self.stats[timeframe]
            total = stats['signals']
            if total > 0:
                success_rate = stats['success'] / total * 100
                print(f"{timeframe.upper()}:")
                print(f"  Total signals: {total}")
                print(f"  Success: {stats['success']} ({success_rate:.1f}%)")
                print(f"  Failure: {stats['failure']} ({100-success_rate:.1f}%)")
                print()


def main():
    """Main execution function."""
    # Create output directory
    os.makedirs(PATHS['processed'], exist_ok=True)
    
    # Create generator
    generator = TrainingDataGenerator()
    
    # Generate datasets
    df_intraday = generator.generate_intraday_data()
    if not df_intraday.empty:
        output_path = os.path.join(PATHS['processed'], 'intraday_training.csv')
        df_intraday.to_csv(output_path, index=False)
        logger.info(f"✓ Saved intraday training data: {len(df_intraday)} examples")
    
    df_swing = generator.generate_swing_data()
    if not df_swing.empty:
        output_path = os.path.join(PATHS['processed'], 'swing_training.csv')
        df_swing.to_csv(output_path, index=False)
        logger.info(f"✓ Saved swing training data: {len(df_swing)} examples")
    
    df_longterm = generator.generate_longterm_data()
    if not df_longterm.empty:
        output_path = os.path.join(PATHS['processed'], 'longterm_training.csv')
        df_longterm.to_csv(output_path, index=False)
        logger.info(f"✓ Saved long-term training data: {len(df_longterm)} examples")
    
    # Print summary
    generator.print_summary()


if __name__ == "__main__":
    main()
