"""
Example Usage - Complete Regime Detection System
=================================================

Demonstrates how to use the complete system with both synthetic and real data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from regime_engine import RegimeDetectionEngine


def example_1_synthetic_data():
    """
    Example 1: Synthetic regime-switching data
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: SYNTHETIC REGIME-SWITCHING DATA")
    print("="*80)
    
    np.random.seed(42)
    n_samples = 500
    
    # Create three distinct regimes
    print("\nGenerating synthetic data with 3 regimes...")
    
    # Regime 1: Low volatility trending (0-150)
    regime1_returns = np.random.randn(150) * 0.01 + 0.001
    
    # Regime 2: High volatility mean-reverting (150-300)
    regime2_returns = np.random.randn(150) * 0.03 - regime1_returns[-50:].mean()
    
    # Regime 3: Moderate volatility trending (300-500)
    regime3_returns = np.random.randn(200) * 0.015 + 0.002
    
    returns = np.concatenate([regime1_returns, regime2_returns, regime3_returns])
    prices = 100 * (1 + returns).cumprod()
    
    # Create DataFrame
    data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.abs(np.random.randn(n_samples) * 0.005)),
        'low': prices * (1 - np.abs(np.random.randn(n_samples) * 0.005)),
        'volume': np.random.randint(10000, 50000, n_samples)
    }, index=pd.date_range('2023-01-01', periods=n_samples, freq='D'))
    
    print(f"✓ Generated {len(data)} data points")
    print(f"  True regimes: 0-150 (Low-Vol Trend), 150-300 (High-Vol Reverting), 300-500 (Mod-Vol Trend)")
    
    # Run detection
    engine = RegimeDetectionEngine()
    results = engine.detect_regimes(data, price_col='close')
    
    # Analyze results
    regime_seq = engine.get_regime_sequence()
    print(f"\n✓ Detected {len(regime_seq.unique())} unique regimes")
    print(f"  Regime distribution:")
    for regime_id, count in regime_seq.value_counts().items():
        print(f"    Regime {int(regime_id)}: {count} samples ({count/len(regime_seq)*100:.1f}%)")
    
    # Export
    engine.export_results('example1_output')
    print(f"\n✓ Results exported to example1_output/")
    
    return engine


def example_2_real_nifty_data():
    """
    Example 2: Real NIFTY 50 data
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: REAL NIFTY 50 DATA")
    print("="*80)
    
    # Path to NIFTY 50 data
    data_path = Path(__file__).parent.parent / 'indexes data' / 'NIFTY 50_minute.csv'
    
    if not data_path.exists():
        print(f"\n⊘ NIFTY 50 data not found at {data_path}")
        print("  Skipping this example...")
        return None
    
    # Load data
    print(f"\nLoading NIFTY 50 data...")
    df = pd.read_csv(data_path)
    
    # Process data
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Take recent subset
    df = df.tail(500)  # Last 500 data points
    
    print(f"✓ Loaded {len(df)} data points")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    # Detect price column
    price_col = 'close' if 'close' in df.columns else 'Close' if 'Close' in df.columns else df.columns[0]
    print(f"  Using price column: {price_col}")
    
    # Run detection
    engine = RegimeDetectionEngine()
    results = engine.detect_regimes(df, price_col=price_col)
    
    # Analyze results
    summary = engine.get_summary()
    print(f"\n✓ Current Market Regime: {summary['current_regime']}")
    print(f"  Confidence: {summary['confidence']:.1%}")
    print(f"  Consensus: {summary['consensus']}")
    
    # Export
    engine.export_results('example2_nifty50_output')
    print(f"\n✓ Results exported to example2_nifty50_output/")
    
    return engine


def example_3_multi_asset_analysis():
    """
    Example 3: Multi-asset correlation analysis
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: MULTI-ASSET CORRELATION ANALYSIS")
    print("="*80)
    
    np.random.seed(42)
    n_samples = 400
    n_assets = 5
    
    # Generate correlated returns with regime changes
    print(f"\nGenerating {n_assets} correlated asset returns...")
    
    # Regime 1: Low correlation
    cov1 = np.eye(n_assets) * 0.01 + 0.002
    returns1 = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=cov1,
        size=150
    )
    
    # Regime 2: High correlation (crisis)
    cov2 = np.eye(n_assets) * 0.02 + 0.015
    returns2 = np.random.multivariate_normal(
        mean=-np.ones(n_assets) * 0.001,
        cov=cov2,
        size=150
    )
    
    # Regime 3: Medium correlation
    cov3 = np.eye(n_assets) * 0.01 + 0.008
    returns3 = np.random.multivariate_normal(
        mean=np.ones(n_assets) * 0.0005,
        cov=cov3,
        size=100
    )
    
    returns = np.vstack([returns1, returns2, returns3])
    
    # Create multi-asset DataFrame
    multi_asset_data = pd.DataFrame(
        returns,
        columns=[f'Asset_{i}' for i in range(n_assets)],
        index=pd.date_range('2023-01-01', periods=n_samples, freq='D')
    )
    
    # Create main price data from Asset_0
    prices = 100 * (1 + returns[:, 0]).cumprod()
    main_data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'volume': np.random.randint(10000, 50000, n_samples)
    }, index=multi_asset_data.index)
    
    print(f"✓ Generated {n_samples} samples for {n_assets} assets")
    
    # Run detection with multi-asset data
    engine = RegimeDetectionEngine()
    results = engine.detect_regimes(
        main_data,
        price_col='close',
        multi_asset_data=multi_asset_data
    )
    
    # Check if TICC detected correlation regimes
    if results['layer4_structural']['ticc'].get('available'):
        ticc_results = results['layer4_structural']['ticc']
        print(f"\n✓ TICC detected {ticc_results['n_regimes']} correlation regimes")
        print(f"  Transitions: {ticc_results['transitions']}")
    else:
        print(f"\n⊘ TICC not available (insufficient multi-asset data)")
    
    # Export
    engine.export_results('example3_multiasset_output')
    print(f"\n✓ Results exported to example3_multiasset_output/")
    
    return engine


def example_4_custom_configuration():
    """
    Example 4: Custom configuration parameters
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: CUSTOM CONFIGURATION")
    print("="*80)
    
    # Import config
    from config import HDP_CONFIG, KALMAN_CONFIG, TICC_CONFIG
    
    # Customize HDP config
    custom_hdp_config = HDP_CONFIG.copy()
    custom_hdp_config['kappa'] = 50.0  # Very sticky regimes
    custom_hdp_config['truncation'] = 12  # More potential regimes
    
    print(f"\nCustom Configuration:")
    print(f"  HDP kappa (stickiness): {custom_hdp_config['kappa']}")
    print(f"  HDP truncation: {custom_hdp_config['truncation']}")
    
    # Generate data
    np.random.seed(42)
    n_samples = 300
    returns = np.random.randn(n_samples) * 0.02
    prices = 100 * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'volume': np.random.randint(10000, 50000, n_samples)
    }, index=pd.date_range('2023-01-01', periods=n_samples, freq='D'))
    
    # Run with custom config
    engine = RegimeDetectionEngine(hdp_config=custom_hdp_config)
    results = engine.detect_regimes(data, price_col='close')
    
    print(f"\n✓ Regime detection with custom config complete")
    print(f"  Current regime: {results['final_regime']['current_regime']}")
    print(f"  Transition probability: {results['final_regime']['transition_probability']:.1%}")
    
    # Export
    engine.export_results('example4_custom_output')
    print(f"\n✓ Results exported to example4_custom_output/")
    
    return engine


def run_all_examples():
    """Run all examples"""
    print("\n" + "="*80)
    print("RUNNING ALL EXAMPLES")
    print("="*80)
    
    examples = [
        ("Synthetic Data", example_1_synthetic_data),
        ("Real NIFTY 50 Data", example_2_real_nifty_data),
        ("Multi-Asset Analysis", example_3_multi_asset_analysis),
        ("Custom Configuration", example_4_custom_configuration)
    ]
    
    results = {}
    for name, func in examples:
        try:
            print(f"\n{'='*80}")
            print(f"Running: {name}")
            print(f"{'='*80}")
            results[name] = func()
        except Exception as e:
            print(f"\n✗ Example '{name}' failed: {e}")
            results[name] = None
    
    # Final summary
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)
    
    for name, result in results.items():
        status = "✓ Success" if result is not None else "✗ Failed"
        print(f"  {name}: {status}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        
        if example == "1":
            example_1_synthetic_data()
        elif example == "2":
            example_2_real_nifty_data()
        elif example == "3":
            example_3_multi_asset_analysis()
        elif example == "4":
            example_4_custom_configuration()
        else:
            print(f"Unknown example: {example}")
            print("Usage: python example_usage.py [1|2|3|4|all]")
    else:
        # Run all examples by default
        run_all_examples()
