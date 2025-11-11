"""
Quick Test - Verify Complete System Installation
=================================================

Run this to verify all components are properly installed and working.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        from config import HDP_CONFIG, TICC_CONFIG, KALMAN_CONFIG
        print("  ✓ config.py")
    except Exception as e:
        print(f"  ✗ config.py: {e}")
        return False
    
    try:
        from feature_engineering import FeatureEngineer
        print("  ✓ feature_engineering.py")
    except Exception as e:
        print(f"  ✗ feature_engineering.py: {e}")
        return False
    
    try:
        from layer1_changepoint import Layer1Changepoint
        print("  ✓ layer1_changepoint.py")
    except Exception as e:
        print(f"  ✗ layer1_changepoint.py: {e}")
        return False
    
    try:
        from layer2_kalman import Layer2KalmanFilter
        print("  ✓ layer2_kalman.py")
    except Exception as e:
        print(f"  ✗ layer2_kalman.py: {e}")
        return False
    
    try:
        from layer3_hdp_regime import Layer3HDPRegime
        print("  ✓ layer3_hdp_regime.py")
    except Exception as e:
        print(f"  ✗ layer3_hdp_regime.py: {e}")
        return False
    
    try:
        from layer4_structural import Layer4Structural
        print("  ✓ layer4_structural.py")
    except Exception as e:
        print(f"  ✗ layer4_structural.py: {e}")
        return False
    
    try:
        from state_aggregator import StateAggregator
        print("  ✓ state_aggregator.py")
    except Exception as e:
        print(f"  ✗ state_aggregator.py: {e}")
        return False
    
    try:
        from regime_engine import RegimeDetectionEngine
        print("  ✓ regime_engine.py")
    except Exception as e:
        print(f"  ✗ regime_engine.py: {e}")
        return False
    
    return True


def test_quick_run():
    """Test quick run with minimal data"""
    print("\nTesting quick run with minimal synthetic data...")
    
    try:
        import numpy as np
        import pandas as pd
        from regime_engine import RegimeDetectionEngine
        
        # Create minimal synthetic data
        np.random.seed(42)
        n_samples = 300  # Increased for rolling window features
        
        returns = np.random.randn(n_samples) * 0.02
        prices = 100 * (1 + returns).cumprod()
        
        data = pd.DataFrame({
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'volume': np.random.randint(1000, 10000, n_samples)
        }, index=pd.date_range('2023-01-01', periods=n_samples, freq='D'))
        
        # Initialize engine
        engine = RegimeDetectionEngine()
        
        # Run detection
        print("  Running regime detection...")
        results = engine.detect_regimes(data, price_col='close')
        
        # Check results
        summary = engine.get_summary()
        
        print(f"  ✓ Detection complete")
        print(f"    Current regime: {summary['current_regime']}")
        print(f"    Confidence: {summary['confidence']:.1%}")
        print(f"    Consensus: {summary['consensus']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Quick run failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("="*60)
    print("COMPLETE SYSTEM VERIFICATION TEST")
    print("="*60)
    
    # Test 1: Imports
    print("\n[Test 1: Imports]")
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n✗ Import test failed. Please check dependencies.")
        return
    
    # Test 2: Quick Run
    print("\n[Test 2: Quick Run]")
    run_ok = test_quick_run()
    
    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Quick Run: {'✓ PASS' if run_ok else '✗ FAIL'}")
    
    if imports_ok and run_ok:
        print("\n✓ ALL TESTS PASSED")
        print("\nSystem is ready to use!")
        print("\nNext steps:")
        print("  1. Run examples: python example_usage.py")
        print("  2. Try your data: python run_full_system.py --data your_data.csv")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Please check error messages above.")
    
    print("="*60)


if __name__ == "__main__":
    main()
