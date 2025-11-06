"""
Quick Verification Test
Runs a simple test to verify all components are working
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

print("=" * 80)
print("QUICK VERIFICATION TEST")
print("=" * 80)

print("\n1. Testing imports...")
try:
    from models.markov_switching import MarkovRegimeSwitching
    from models.garch_volatility import GARCHVolatilityRegime
    from models.state_aggregator import StateAggregator
    from utils.data_loader import load_sample_data
    from utils.metrics import ModelEvaluator
    import numpy as np
    import pandas as pd
    print("   ✓ All imports successful!")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

print("\n2. Testing data generation...")
try:
    prices, returns = load_sample_data(n_samples=500)
    print(f"   ✓ Generated {len(returns)} sample returns")
except Exception as e:
    print(f"   ✗ Data generation error: {e}")
    sys.exit(1)

print("\n3. Testing Stream 1 (Markov Switching)...")
try:
    markov = MarkovRegimeSwitching(n_regimes=2)
    markov.fit(returns)
    trend_regimes = markov.predict_regime_id()
    print(f"   ✓ Markov model fitted successfully")
    print(f"   ✓ Predicted {len(trend_regimes)} regime classifications")
except Exception as e:
    print(f"   ✗ Markov model error: {e}")
    sys.exit(1)

print("\n4. Testing Stream 2 (GARCH)...")
try:
    garch = GARCHVolatilityRegime(p=1, q=1)
    garch.fit(returns)
    vol_regimes = garch.predict_regime_id()
    print(f"   ✓ GARCH model fitted successfully")
    print(f"   ✓ Predicted {len(vol_regimes)} regime classifications")
except Exception as e:
    print(f"   ✗ GARCH model error: {e}")
    sys.exit(1)

print("\n5. Testing State Aggregation...")
try:
    aggregator = StateAggregator()
    combined = aggregator.aggregate_states(
        trend_regimes=trend_regimes,
        volatility_regimes=vol_regimes
    )
    print(f"   ✓ State aggregation successful")
    print(f"   ✓ Generated {len(combined)} combined state vectors")
    print(f"   ✓ Final regimes: {combined['final_regime'].unique().tolist()}")
except Exception as e:
    print(f"   ✗ Aggregation error: {e}")
    sys.exit(1)

print("\n6. Testing current state retrieval...")
try:
    current_state = aggregator.get_current_state(combined)
    print(f"   ✓ Current regime: {current_state['final_regime']}")
    print(f"   ✓ Description: {current_state['description']}")
except Exception as e:
    print(f"   ✗ State retrieval error: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL VERIFICATION TESTS PASSED!")
print("=" * 80)
print("\n✅ System is fully operational and ready for comprehensive testing")
print("\nNext steps:")
print("  1. Run: python tests/test_stream1_markov.py")
print("  2. Run: python tests/test_stream2_garch.py")
print("  3. Run: python tests/test_complete_system.py")
print("  4. Or run all: python run_tests.py")
print("\n" + "=" * 80)
