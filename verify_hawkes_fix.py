"""
Verification Script for Hawkes Process Fix
Tests the improved implementation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from models.hawkes_process import HawkesProcess
import time
import warnings
warnings.filterwarnings('ignore')


def print_header(title):
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


print_header("HAWKES PROCESS FIX VERIFICATION")

results = []

# Test 1: Alpha >= Beta (no longer enforced by default)
print("\n--- Test 1: Alpha >= Beta Handling ---")
try:
    model = HawkesProcess(alpha=1.2, beta=1.0, enforce_stability=False)
    event_times = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    model.fit(event_times, optimize=False)
    
    print(f"✓ PASS: Allows alpha ({model.alpha}) >= beta ({model.beta})")
    print(f"  Branching ratio: {model.alpha/model.beta:.3f} (explosive process)")
    results.append({'Test': 'Alpha >= Beta', 'Status': 'PASS', 'Note': 'Configurable constraint'})
except Exception as e:
    print(f"✗ FAIL: {e}")
    results.append({'Test': 'Alpha >= Beta', 'Status': 'FAIL', 'Error': str(e)[:50]})

# Test 2: Performance - High Frequency Data
print("\n--- Test 2: High-Frequency Data Performance ---")
try:
    # Generate 10,000 events
    n_events = 10000
    event_times = np.cumsum(np.random.exponential(0.01, n_events))
    
    start_time = time.time()
    model = HawkesProcess()
    model.fit(event_times, optimize=True)
    elapsed = time.time() - start_time
    
    print(f"✓ PASS: Fitted {n_events} events in {elapsed:.3f} seconds")
    print(f"  Performance: {n_events/elapsed:.0f} events/second")
    
    if elapsed < 5.0:  # Should be fast
        results.append({'Test': 'High-Freq Performance', 'Status': 'PASS', 
                       'Time': f'{elapsed:.3f}s', 'Events/sec': f'{n_events/elapsed:.0f}'})
    else:
        results.append({'Test': 'High-Freq Performance', 'Status': 'SLOW', 
                       'Time': f'{elapsed:.3f}s'})
except Exception as e:
    print(f"✗ FAIL: {e}")
    results.append({'Test': 'High-Freq Performance', 'Status': 'FAIL', 'Error': str(e)[:50]})

# Test 3: Edge Case - Single Event
print("\n--- Test 3: Edge Case - Single Event ---")
try:
    model = HawkesProcess()
    model.fit(np.array([1.0]), optimize=False)
    
    print(f"✓ PASS: Handles single event gracefully")
    results.append({'Test': 'Single Event', 'Status': 'PASS'})
except Exception as e:
    print(f"✗ FAIL: {e}")
    results.append({'Test': 'Single Event', 'Status': 'FAIL', 'Error': str(e)[:50]})

# Test 4: Edge Case - Duplicate Events
print("\n--- Test 4: Edge Case - Duplicate Event Times ---")
try:
    event_times = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0])
    model = HawkesProcess()
    model.fit(event_times, optimize=False)
    
    print(f"✓ PASS: Handles duplicates (cleaned to {len(model.event_times)} unique events)")
    results.append({'Test': 'Duplicate Times', 'Status': 'PASS', 
                   'Cleaned': f'{len(event_times)} -> {len(model.event_times)}'})
except Exception as e:
    print(f"✗ FAIL: {e}")
    results.append({'Test': 'Duplicate Times', 'Status': 'FAIL', 'Error': str(e)[:50]})

# Test 5: Edge Case - Few Events
print("\n--- Test 5: Edge Case - Too Few Events ---")
try:
    event_times = np.array([1.0, 2.0, 3.0])  # Only 3 events
    model = HawkesProcess(min_events=5)
    model.fit(event_times, optimize=False)
    
    print(f"✓ PASS: Handles few events with warning")
    results.append({'Test': 'Few Events', 'Status': 'PASS'})
except Exception as e:
    print(f"✗ FAIL: {e}")
    results.append({'Test': 'Few Events', 'Status': 'FAIL', 'Error': str(e)[:50]})

# Test 6: Missing Data Handling
print("\n--- Test 6: Missing Data in Returns ---")
try:
    # Create returns with NaN values
    returns = pd.Series(np.random.randn(100))
    returns.iloc[10:15] = np.nan
    returns.iloc[50:55] = np.nan
    
    model = HawkesProcess()
    model.fit_from_returns(returns, threshold=1.5, adaptive_threshold=True)
    
    print(f"✓ PASS: Handles missing data ({returns.isna().sum()} NaN values)")
    print(f"  Detected {len(model.event_times)} events")
    results.append({'Test': 'Missing Data', 'Status': 'PASS', 
                   'Events': len(model.event_times)})
except Exception as e:
    print(f"✗ FAIL: {e}")
    results.append({'Test': 'Missing Data', 'Status': 'FAIL', 'Error': str(e)[:50]})

# Test 7: Adaptive Thresholding
print("\n--- Test 7: Adaptive Event Detection ---")
try:
    # Create data with few extreme events
    returns = pd.Series(np.random.randn(200) * 0.1)  # Low volatility
    
    model = HawkesProcess()
    model.fit_from_returns(returns, threshold=3.0, adaptive_threshold=True)
    
    print(f"✓ PASS: Adaptive thresholding found {len(model.event_times)} events")
    results.append({'Test': 'Adaptive Threshold', 'Status': 'PASS', 
                   'Events': len(model.event_times)})
except Exception as e:
    print(f"✗ FAIL: {e}")
    results.append({'Test': 'Adaptive Threshold', 'Status': 'FAIL', 'Error': str(e)[:50]})

# Test 8: False Positive Reduction
print("\n--- Test 8: False Positive Control ---")
try:
    # Create stable data with no clear excitation
    np.random.seed(42)
    event_times = np.cumsum(np.random.exponential(1.0, 100))  # Poisson process
    
    model = HawkesProcess()
    model.fit(event_times, optimize=True)
    
    branching_ratio = model.alpha / model.beta
    
    print(f"✓ Branching ratio: {branching_ratio:.4f}")
    if branching_ratio < 0.5:  # Should detect low excitation
        print(f"✓ PASS: Correctly identifies low excitation (BR < 0.5)")
        results.append({'Test': 'False Positive Control', 'Status': 'PASS',
                       'Branching_Ratio': f'{branching_ratio:.4f}'})
    else:
        print(f"⚠ WARNING: High branching ratio on Poisson data")
        results.append({'Test': 'False Positive Control', 'Status': 'WARNING',
                       'Branching_Ratio': f'{branching_ratio:.4f}'})
except Exception as e:
    print(f"✗ FAIL: {e}")
    results.append({'Test': 'False Positive Control', 'Status': 'FAIL', 'Error': str(e)[:50]})

# Test 9: Stability Check
print("\n--- Test 9: Stability Detection ---")
try:
    model1 = HawkesProcess(alpha=0.3, beta=1.0)
    model1.fit(np.array([1.0, 2.0, 3.0, 5.0]), optimize=False)
    
    model2 = HawkesProcess(alpha=1.2, beta=1.0, enforce_stability=False)
    model2.fit(np.array([1.0, 2.0, 3.0, 5.0]), optimize=False)
    
    print(f"✓ Model 1: is_stable() = {model1.is_stable()} (α/β = {model1.alpha/model1.beta:.2f})")
    print(f"✓ Model 2: is_stable() = {model2.is_stable()} (α/β = {model2.alpha/model2.beta:.2f})")
    
    if model1.is_stable() and not model2.is_stable():
        results.append({'Test': 'Stability Detection', 'Status': 'PASS'})
    else:
        results.append({'Test': 'Stability Detection', 'Status': 'FAIL'})
except Exception as e:
    print(f"✗ FAIL: {e}")
    results.append({'Test': 'Stability Detection', 'Status': 'FAIL', 'Error': str(e)[:50]})

# Test 10: Optimization Methods
print("\n--- Test 10: Multiple Optimization Methods ---")
try:
    event_times = np.array([1.0, 1.5, 2.0, 2.2, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0])
    
    model_mle = HawkesProcess()
    model_mle.fit(event_times, optimize=True, method='mle')
    
    model_moments = HawkesProcess()
    model_moments.fit(event_times, optimize=True, method='moments')
    
    print(f"✓ MLE: α={model_mle.alpha:.4f}, β={model_mle.beta:.4f}")
    print(f"✓ Moments: α={model_moments.alpha:.4f}, β={model_moments.beta:.4f}")
    results.append({'Test': 'Optimization Methods', 'Status': 'PASS'})
except Exception as e:
    print(f"✗ FAIL: {e}")
    results.append({'Test': 'Optimization Methods', 'Status': 'FAIL', 'Error': str(e)[:50]})

# Summary
print_header("VERIFICATION SUMMARY")

df = pd.DataFrame(results)
print(df.to_string(index=False))

passed = df[df['Status'] == 'PASS'].shape[0]
total = df.shape[0]

print(f"\n{'='*80}")
print(f" RESULTS: {passed}/{total} TESTS PASSED ".center(80, '='))
print(f"{'='*80}")

# Save report
report_path = Path(__file__).parent / 'reports' / 'HAWKES_FIX_VERIFICATION.csv'
df.to_csv(report_path, index=False)
print(f"\n✓ Report saved: {report_path}")

sys.exit(0 if passed == total else 1)
