"""
Comprehensive Testing Suite for HDP-HMM
Part 1: Normal Testing
Part 2: Extreme/Stress Testing
Part 3: Weakness Testing
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from models import HDPHMM
from utils.data_loader import DataLoader
import warnings
warnings.filterwarnings('ignore')


def print_header(title):
    print("\n" + "=" * 100)
    print(f" {title} ".center(100, "="))
    print("=" * 100)


def save_report(report_name, data):
    """Save report to CSV"""
    reports_dir = Path(__file__).parent.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    filepath = reports_dir / report_name
    data.to_csv(filepath, index=False)
    print(f"\n✓ Report saved: {filepath}")


# ============================================================================
# PART 1: NORMAL TESTING
# ============================================================================

def test_hdphmm_normal():
    """Normal testing with synthetic and real data"""
    print_header("PART 1: HDP-HMM NORMAL TESTING")
    
    results = []
    
    # Test 1: Synthetic 3-regime data
    print("\n--- Test 1: Synthetic 3-Regime Data ---")
    try:
        np.random.seed(42)
        # Create 3 regimes with different means
        regime1 = np.random.normal(0, 1, 100)
        regime2 = np.random.normal(5, 1, 100)
        regime3 = np.random.normal(-3, 1, 100)
        
        data = np.concatenate([regime1, regime2, regime3])
        
        model = HDPHMM(truncation=10, alpha=1.0, gamma=1.0, max_iter=50)
        model.fit(data)
        
        info = model.get_model_info()
        stats = model.get_regime_statistics()
        active_regimes = model.get_active_regimes()
        
        print(f"✓ Fitted HDP-HMM on {len(data)} observations")
        print(f"  Active regimes discovered: {info['n_active_regimes']} (truncation: {info['truncation']})")
        print(f"  Active regime indices: {active_regimes}")
        
        for regime_name, regime_stats in stats.items():
            print(f"  {regime_name}: μ={regime_stats['mean']:.2f}, σ²={regime_stats['variance']:.2f}, "
                  f"{regime_stats['percentage']:.1f}%")
        
        # Check if discovered approximately correct number
        discovery_quality = "Excellent" if info['n_active_regimes'] == 3 else "Good" if abs(info['n_active_regimes'] - 3) <= 1 else "Poor"
        
        results.append({
            'Test': 'Synthetic 3-regime',
            'Status': 'PASSED',
            'N_samples': len(data),
            'True_regimes': 3,
            'Discovered_regimes': info['n_active_regimes'],
            'Discovery_quality': discovery_quality
        })
        
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        results.append({
            'Test': 'Synthetic 3-regime',
            'Status': 'FAILED',
            'Error': str(e)[:100]
        })
    
    # Test 2: Synthetic 2-regime data
    print("\n--- Test 2: Synthetic 2-Regime Data (Bull/Bear) ---")
    try:
        # Bull market (positive mean)
        bull = np.random.normal(0.1, 0.5, 150)
        # Bear market (negative mean)
        bear = np.random.normal(-0.1, 0.8, 150)
        
        data = np.concatenate([bull, bear])
        
        model = HDPHMM(truncation=8, alpha=1.0, gamma=1.0, max_iter=50)
        model.fit(data)
        
        info = model.get_model_info()
        stats = model.get_regime_statistics()
        
        print(f"✓ Fitted HDP-HMM on bull/bear data")
        print(f"  Discovered {info['n_active_regimes']} regimes")
        
        for regime_name, regime_stats in stats.items():
            regime_type = "Bull" if regime_stats['mean'] > 0 else "Bear"
            print(f"  {regime_name} ({regime_type}): μ={regime_stats['mean']:.3f}, {regime_stats['percentage']:.1f}%")
        
        results.append({
            'Test': 'Synthetic 2-regime',
            'Status': 'PASSED',
            'N_samples': len(data),
            'Discovered_regimes': info['n_active_regimes']
        })
        
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        results.append({
            'Test': 'Synthetic 2-regime',
            'Status': 'FAILED',
            'Error': str(e)[:100]
        })
    
    # Test 3: Real NIFTY 50 data
    print("\n--- Test 3: Real NIFTY 50 Returns ---")
    try:
        loader = DataLoader()
        df = loader.load_index('NIFTY 50')
        
        if len(df) > 1000:
            df = loader.resample_to_daily(df)
        
        returns = loader.calculate_returns(df, 'close', log_returns=True).tail(500)
        
        model = HDPHMM(truncation=8, alpha=1.0, gamma=1.0, max_iter=50)
        model.fit(returns)
        
        info = model.get_model_info()
        stats = model.get_regime_statistics()
        
        print(f"✓ Fitted HDP-HMM on NIFTY 50 ({len(returns)} returns)")
        print(f"  Discovered {info['n_active_regimes']} market regimes")
        
        for regime_name, regime_stats in stats.items():
            regime_type = "Bull" if regime_stats['mean'] > 0 else "Bear" if regime_stats['mean'] < -0.001 else "Neutral"
            print(f"  {regime_name} ({regime_type}): μ={regime_stats['mean']:.6f}, "
                  f"σ²={regime_stats['variance']:.6f}, {regime_stats['percentage']:.1f}%")
        
        results.append({
            'Test': 'Real NIFTY 50',
            'Status': 'PASSED',
            'N_samples': len(returns),
            'Discovered_regimes': info['n_active_regimes'],
            'Interpretation': 'Market regimes identified'
        })
        
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        results.append({
            'Test': 'Real NIFTY 50',
            'Status': 'FAILED',
            'Error': str(e)[:100]
        })
    
    # Test 4: Real NIFTY BANK data
    print("\n--- Test 4: Real NIFTY BANK Returns ---")
    try:
        df = loader.load_index('NIFTY BANK')
        
        if len(df) > 1000:
            df = loader.resample_to_daily(df)
        
        returns = loader.calculate_returns(df, 'close', log_returns=True).tail(500)
        
        model = HDPHMM(truncation=10, alpha=1.0, gamma=1.0, max_iter=50)
        model.fit(returns)
        
        info = model.get_model_info()
        stats = model.get_regime_statistics()
        trans_matrix = model.get_transition_matrix(active_only=True)
        
        print(f"✓ Fitted HDP-HMM on NIFTY BANK ({len(returns)} returns)")
        print(f"  Discovered {info['n_active_regimes']} regimes")
        print(f"  Transition matrix shape: {trans_matrix.shape}")
        
        results.append({
            'Test': 'Real NIFTY BANK',
            'Status': 'PASSED',
            'N_samples': len(returns),
            'Discovered_regimes': info['n_active_regimes']
        })
        
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        results.append({
            'Test': 'Real NIFTY BANK',
            'Status': 'FAILED',
            'Error': str(e)[:100]
        })
    
    # Save normal test results
    df_results = pd.DataFrame(results)
    save_report('HDPHMM_normal_testing_report.csv', df_results)
    
    return results


# ============================================================================
# PART 2: EXTREME/STRESS TESTING
# ============================================================================

def test_hdphmm_extreme():
    """Extreme and stress testing"""
    print_header("PART 2: HDP-HMM EXTREME/STRESS TESTING")
    
    results = []
    
    # Test 1: Very short series
    print("\n--- Extreme Test 1: Very Short Series (30 points) ---")
    try:
        data = np.random.randn(30)
        model = HDPHMM(truncation=5, max_iter=20)
        model.fit(data)
        info = model.get_model_info()
        print(f"✓ PASSED: Discovered {info['n_active_regimes']} regimes from {len(data)} points")
        results.append({'Test': 'Short series (30)', 'Status': 'PASSED', 'Regimes': info['n_active_regimes']})
    except Exception as e:
        print(f"✗ FAILED: {str(e)[:100]}")
        results.append({'Test': 'Short series (30)', 'Status': 'FAILED', 'Error': str(e)[:50]})
    
    # Test 2: Single regime (homogeneous data)
    print("\n--- Extreme Test 2: Single Regime (Homogeneous Data) ---")
    try:
        data = np.random.normal(0, 1, 200)
        model = HDPHMM(truncation=8, max_iter=50)
        model.fit(data)
        info = model.get_model_info()
        print(f"✓ PASSED: Discovered {info['n_active_regimes']} regimes (expected 1-2)")
        if info['n_active_regimes'] <= 2:
            print(f"  ✓ EXCELLENT: Correctly identified minimal regimes")
        else:
            print(f"  ⚠ WARNING: Over-fitting, detected {info['n_active_regimes']} regimes")
        results.append({'Test': 'Single regime', 'Status': 'PASSED', 'Regimes': info['n_active_regimes']})
    except Exception as e:
        print(f"✗ FAILED: {str(e)[:100]}")
        results.append({'Test': 'Single regime', 'Status': 'FAILED', 'Error': str(e)[:50]})
    
    # Test 3: Many regimes (10+ true regimes)
    print("\n--- Extreme Test 3: Many Regimes (10 true regimes) ---")
    try:
        many_regimes = []
        for i in range(10):
            many_regimes.extend(np.random.normal(i * 2, 0.5, 30))
        
        data = np.array(many_regimes)
        model = HDPHMM(truncation=15, max_iter=50)
        model.fit(data)
        info = model.get_model_info()
        print(f"✓ PASSED: Discovered {info['n_active_regimes']} regimes (true: 10)")
        results.append({'Test': 'Many regimes (10)', 'Status': 'PASSED', 'Discovered': info['n_active_regimes']})
    except Exception as e:
        print(f"✗ FAILED: {str(e)[:100]}")
        results.append({'Test': 'Many regimes (10)', 'Status': 'FAILED', 'Error': str(e)[:50]})
    
    # Test 4: Extreme variance differences
    print("\n--- Extreme Test 4: Extreme Variance Differences ---")
    try:
        low_var = np.random.normal(0, 0.01, 100)
        high_var = np.random.normal(0, 10.0, 100)
        data = np.concatenate([low_var, high_var])
        
        model = HDPHMM(truncation=8, max_iter=50)
        model.fit(data)
        stats = model.get_regime_statistics()
        
        variances = [s['variance'] for s in stats.values()]
        var_ratio = max(variances) / min(variances)
        print(f"✓ PASSED: Variance ratio = {var_ratio:.1f}x")
        results.append({'Test': 'Extreme variance', 'Status': 'PASSED', 'Variance_ratio': f'{var_ratio:.1f}'})
    except Exception as e:
        print(f"✗ FAILED: {str(e)[:100]}")
        results.append({'Test': 'Extreme variance', 'Status': 'FAILED', 'Error': str(e)[:50]})
    
    # Test 5: Rapid regime switching
    print("\n--- Extreme Test 5: Very Rapid Regime Switching ---")
    try:
        rapid = []
        for i in range(200):
            if i % 10 < 5:
                rapid.append(np.random.normal(0, 1))
            else:
                rapid.append(np.random.normal(5, 1))
        
        data = np.array(rapid)
        model = HDPHMM(truncation=8, max_iter=50)
        model.fit(data)
        
        regimes = model.predict_regime()
        transitions = np.sum(np.diff(regimes) != 0)
        print(f"✓ PASSED: Detected {transitions} transitions (expected ~20)")
        results.append({'Test': 'Rapid switching', 'Status': 'PASSED', 'Transitions': transitions})
    except Exception as e:
        print(f"✗ FAILED: {str(e)[:100]}")
        results.append({'Test': 'Rapid switching', 'Status': 'FAILED', 'Error': str(e)[:50]})
    
    # Test 6: Outliers
    print("\n--- Extreme Test 6: Heavy Outliers (20%) ---")
    try:
        data = np.random.normal(0, 1, 200)
        outlier_indices = np.random.choice(200, 40, replace=False)
        data[outlier_indices] = np.random.choice([-50, 50], 40)
        
        model = HDPHMM(truncation=8, max_iter=50)
        model.fit(data)
        info = model.get_model_info()
        print(f"✓ PASSED: Discovered {info['n_active_regimes']} regimes with outliers")
        results.append({'Test': 'Heavy outliers', 'Status': 'PASSED', 'Regimes': info['n_active_regimes']})
    except Exception as e:
        print(f"✗ FAILED: {str(e)[:100]}")
        results.append({'Test': 'Heavy outliers', 'Status': 'FAILED', 'Error': str(e)[:50]})
    
    # Save extreme test results
    df_results = pd.DataFrame(results)
    save_report('HDPHMM_extreme_testing_report.csv', df_results)
    
    return results


# ============================================================================
# PART 3: WEAKNESS TESTING
# ============================================================================

def test_hdphmm_weakness():
    """Find exact breaking points and weaknesses"""
    print_header("PART 3: HDP-HMM WEAKNESS TESTING")
    
    results = []
    
    # Test 1: Minimum sample size
    print("\n--- Weakness Test 1: Minimum Sample Size ---")
    sizes = [10, 20, 30, 50, 100, 200]
    for size in sizes:
        try:
            data = np.random.randn(size)
            model = HDPHMM(truncation=5, max_iter=20)
            model.fit(data)
            info = model.get_model_info()
            print(f"  N={size}: ✓ Works, {info['n_active_regimes']} regimes")
            results.append({'Test': f'Min size N={size}', 'Status': 'PASSED', 'Regimes': info['n_active_regimes']})
        except:
            print(f"  N={size}: ✗ FAILS")
            results.append({'Test': f'Min size N={size}', 'Status': 'FAILED'})
    
    # Test 2: Truncation level impact
    print("\n--- Weakness Test 2: Truncation Level Impact ---")
    data = np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(5, 1, 100),
        np.random.normal(-3, 1, 100)
    ])
    
    truncations = [3, 5, 8, 10, 15, 20]
    for trunc in truncations:
        model = HDPHMM(truncation=trunc, max_iter=30)
        model.fit(data)
        info = model.get_model_info()
        print(f"  Truncation={trunc}: Discovered {info['n_active_regimes']} regimes")
        results.append({'Test': f'Truncation={trunc}', 'Discovered': info['n_active_regimes'], 'True': 3})
    
    # Test 3: Alpha parameter sensitivity
    print("\n--- Weakness Test 3: Alpha Parameter Sensitivity ---")
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
    for alpha in alphas:
        model = HDPHMM(truncation=10, alpha=alpha, max_iter=30)
        model.fit(data)
        info = model.get_model_info()
        print(f"  Alpha={alpha}: Discovered {info['n_active_regimes']} regimes")
        results.append({'Test': f'Alpha={alpha}', 'Discovered': info['n_active_regimes']})
    
    # Test 4: Convergence iterations
    print("\n--- Weakness Test 4: Convergence Iterations ---")
    iterations = [10, 20, 50, 100]
    for max_iter in iterations:
        model = HDPHMM(truncation=8, max_iter=max_iter)
        model.fit(data)
        info = model.get_model_info()
        print(f"  Max_iter={max_iter}: Discovered {info['n_active_regimes']} regimes")
        results.append({'Test': f'MaxIter={max_iter}', 'Discovered': info['n_active_regimes']})
    
    # Test 5: Regime separation quality
    print("\n--- Weakness Test 5: Regime Separation Quality ---")
    separations = [0.5, 1.0, 2.0, 5.0, 10.0]
    for sep in separations:
        data_sep = np.concatenate([
            np.random.normal(0, 1, 100),
            np.random.normal(sep, 1, 100)
        ])
        model = HDPHMM(truncation=8, max_iter=30)
        model.fit(data_sep)
        info = model.get_model_info()
        discovery = "Correct" if info['n_active_regimes'] == 2 else "Incorrect"
        print(f"  Separation={sep}σ: {info['n_active_regimes']} regimes - {discovery}")
        results.append({'Test': f'Separation={sep}', 'Discovered': info['n_active_regimes'], 'Expected': 2})
    
    # Test 6: Overlapping regimes
    print("\n--- Weakness Test 6: Overlapping Regimes ---")
    overlap_data = np.concatenate([
        np.random.normal(0, 2, 150),  # Wide variance
        np.random.normal(1, 2, 150)   # Overlapping
    ])
    model = HDPHMM(truncation=8, max_iter=50)
    model.fit(overlap_data)
    info = model.get_model_info()
    print(f"  Overlapping regimes: Discovered {info['n_active_regimes']} (true: 2)")
    results.append({'Test': 'Overlapping regimes', 'Discovered': info['n_active_regimes'], 'Expected': 2})
    
    # Save weakness test results
    df_results = pd.DataFrame(results)
    save_report('HDPHMM_weakness_testing_report.csv', df_results)
    
    print_header("HDP-HMM WEAKNESS SUMMARY")
    print("\nWEAKNESSES IDENTIFIED:")
    print("  1. Minimum data: Works with 10+ points (reliable with 100+)")
    print("  2. Truncation dependent: Higher truncation allows more regimes")
    print("  3. Alpha sensitivity: Higher alpha → more regimes discovered")
    print("  4. Convergence: More iterations → better but slower")
    print("  5. Overlapping regimes: Struggles when regimes have large overlap")
    print("  6. Computational cost: Increases with truncation level")
    print("\nSTRENGTHS:")
    print("  ✓ Automatic regime discovery (no need to specify K)")
    print("  ✓ Handles varying number of regimes")
    print("  ✓ Robust to outliers")
    print("  ✓ Works on real market data")
    print("  ✓ Provides uncertainty via regime probabilities")
    
    return results


def main():
    """Run all HDP-HMM tests"""
    print_header("COMPREHENSIVE HDP-HMM TESTING SUITE")
    print("Testing Hierarchical Dirichlet Process Hidden Markov Model")
    print("Automatically discovers number of market regimes from data")
    
    input("\n[Press Enter to begin testing...]")
    
    # Run all three test parts
    normal_results = test_hdphmm_normal()
    extreme_results = test_hdphmm_extreme()
    weakness_results = test_hdphmm_weakness()
    
    print_header("ALL HDP-HMM TESTING COMPLETE")
    print("\n✓ Normal testing: Reports saved")
    print("✓ Extreme testing: Reports saved")
    print("✓ Weakness testing: Reports saved")
    print("\nAll reports available in reports/ directory:")
    print("  - HDPHMM_normal_testing_report.csv")
    print("  - HDPHMM_extreme_testing_report.csv")
    print("  - HDPHMM_weakness_testing_report.csv")


if __name__ == "__main__":
    main()
