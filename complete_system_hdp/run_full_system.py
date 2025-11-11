"""
Main Execution Script - Run Complete Regime Detection System
=============================================================

Command-line interface for running the complete regime detection pipeline
on real market data.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from regime_engine import RegimeDetectionEngine


def load_data(file_path: str, date_col: str = None) -> pd.DataFrame:
    """
    Load market data from CSV file
    
    Args:
        file_path: Path to CSV file
        date_col: Name of date column (if None, assumes index)
        
    Returns:
        DataFrame with market data
    """
    print(f"Loading data from {file_path}...")
    
    df = pd.read_csv(file_path)
    
    # Set index
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    print(f"✓ Loaded {len(df)} rows")
    print(f"  Columns: {', '.join(df.columns)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Complete Regime Detection System using Sticky HDP-HMM'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to market data CSV file'
    )
    
    parser.add_argument(
        '--price-col',
        type=str,
        default='close',
        help='Name of price column (default: close)'
    )
    
    parser.add_argument(
        '--date-col',
        type=str,
        default=None,
        help='Name of date column (default: auto-detect)'
    )
    
    parser.add_argument(
        '--multi-asset',
        type=str,
        default=None,
        help='Path to multi-asset returns CSV (optional)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='regime_output',
        help='Output directory for results (default: regime_output)'
    )
    
    parser.add_argument(
        '--kappa',
        type=float,
        default=20.0,
        help='Sticky parameter for HDP-HMM (default: 20.0)'
    )
    
    args = parser.parse_args()
    
    # Load data
    data = load_data(args.data, args.date_col)
    
    # Load multi-asset data if provided
    multi_asset_data = None
    if args.multi_asset:
        multi_asset_data = load_data(args.multi_asset, args.date_col)
    
    # Initialize engine
    print("\n" + "="*80)
    print("INITIALIZING REGIME DETECTION ENGINE")
    print("="*80)
    print(f"Configuration:")
    print(f"  Sticky parameter (kappa): {args.kappa}")
    print(f"  Price column: {args.price_col}")
    print(f"  Multi-asset data: {'Yes' if multi_asset_data is not None else 'No'}")
    
    # Modify HDP config with custom kappa
    from config import HDP_CONFIG
    HDP_CONFIG['kappa'] = args.kappa
    
    engine = RegimeDetectionEngine(hdp_config=HDP_CONFIG)
    
    # Run detection
    print("\n" + "="*80)
    print("RUNNING REGIME DETECTION")
    print("="*80)
    
    results = engine.detect_regimes(
        data=data,
        price_col=args.price_col,
        multi_asset_data=multi_asset_data
    )
    
    # Export results
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    
    engine.export_results(args.output)
    
    # Print final summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    summary = engine.get_summary()
    print(f"\nFinal Regime: {summary['current_regime']}")
    print(f"Description: {summary['description']}")
    print(f"Confidence: {summary['confidence']:.1%}")
    print(f"Consensus: {summary['consensus']}")
    print(f"Transition Probability: {summary['transition_probability']:.1%}")
    
    print(f"\n✓ Results saved to {args.output}/")
    print(f"\nFiles generated:")
    print(f"  - regime_sequence.csv (full regime sequence)")
    print(f"  - current_regime.csv (current regime details)")
    print(f"  - layers_summary.csv (layer-by-layer summary)")
    
    print("\n" + "="*80)
    print("✓ EXECUTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
