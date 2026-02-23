"""
Calculate Win Rates Script
Reads training data and calculates historical win percentages for each strategy mode.
Results are saved to a JSON file for quick loading during scans.
"""

import pandas as pd
import json
import os
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'win_rates.json')


def calculate_win_rate(csv_path: str) -> dict:
    """Calculate win rate from a training CSV file."""
    try:
        df = pd.read_csv(csv_path)
        
        if 'label' not in df.columns:
            return {'available': False, 'error': 'No label column found'}
        
        total = len(df)
        wins = (df['label'] == 1).sum()
        losses = (df['label'] == 0).sum()
        
        win_rate = (wins / total * 100) if total > 0 else 0
        
        return {
            'available': True,
            'total_trades': int(total),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': round(win_rate, 1)
        }
    
    except Exception as e:
        return {'available': False, 'error': str(e)}


def main():
    print("=" * 60)
    print("CALCULATING WIN RATES FROM HISTORICAL DATA")
    print("=" * 60)
    print()
    
    results = {
        'generated_at': datetime.now().isoformat(),
        'modes': {}
    }
    
    # Calculate for each mode
    modes = {
        'INTRADAY': 'intraday_training.csv',
        'SWING': 'swing_training.csv',
        'LONG_TERM': 'longterm_training.csv'
    }
    
    for mode, filename in modes.items():
        filepath = os.path.join(PROCESSED_DIR, filename)
        
        if os.path.exists(filepath):
            print(f"Processing {mode}...")
            stats = calculate_win_rate(filepath)
            results['modes'][mode] = stats
            
            if stats['available']:
                print(f"  ✓ {stats['total_trades']} trades analyzed")
                print(f"  ✓ Win Rate: {stats['win_rate']}%")
                print(f"  ✓ Wins: {stats['wins']} | Losses: {stats['losses']}")
            else:
                print(f"  ✗ Error: {stats.get('error', 'Unknown')}")
        else:
            print(f"  ✗ {filename} not found")
            results['modes'][mode] = {'available': False, 'error': 'File not found'}
        
        print()
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 60)
    print(f"✓ Results saved to: {OUTPUT_FILE}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
