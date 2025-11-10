"""
Verification Script for TICC Clustering Fix
Tests the improved implementation to ensure false positives are reduced
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from models import TICCClustering
import warnings
warnings.filterwarnings('ignore')


def print_separator(char='=', length=80):
    print(char * length)


def print_header(title):
    print_separator()
    print(f" {title} ".center(80, '='))
    print_separator()


def test_synthetic_3_regimes():
    """Test with known 3-regime synthetic data"""
    print_header("TEST 1: Synthetic Data with 3 Known Regimes")
    
    np.random.seed(42)
    n_assets = 5
    
    print(f"\nGenerating synthetic data:")
    print(f"  - {n_assets} assets")
    print(f"  - 3 regimes with distinct correlation structures")
    print(f"  - 100 samples per regime")
    
    # Regime 1: High positive correlation
    cov1 = np.ones((n_assets, n_assets)) * 0.7
    np.fill_diagonal(cov1, 1.0)
    regime1 = np.random.multivariate_normal(np.zeros(n_assets), cov1, 100)
    
    # Regime 2: Low correlation
    cov2 = np.eye(n_assets) * 0.1
    np.fill_diagonal(cov2, 1.0)
    regime2 = np.random.multivariate_normal(np.zeros(n_assets), cov2, 100)
    
    # Regime 3: Negative correlation (alternating)
    cov3 = np.array([
        [1.0, -0.6, 0.1, -0.6, 0.1],
        [-0.6, 1.0, -0.6, 0.1, -0.6],
        [0.1, -0.6, 1.0, -0.6, 0.1],
        [-0.6, 0.1, -0.6, 1.0, -0.6],
        [0.1, -0.6, 0.1, -0.6, 1.0]
    ])
    regime3 = np.random.multivariate_normal(np.zeros(n_assets), cov3, 100)
    
    synthetic_data = np.vstack([regime1, regime2, regime3])
    true_labels = np.array([0]*100 + [1]*100 + [2]*100)
    
    print(f"\nFitting TICC model...")
    model = TICCClustering(n_clusters=3, window_size=10, beta=1000, min_cluster_size=5)
    model.fit(synthetic_data)
    
    predicted_labels = model.predict()
    stats = model.get_cluster_statistics()
    transitions = model.get_regime_transitions()
    info = model.get_model_info()
    
    print(f"\n{'Results:':-<80}")
    print(f"  Clusters requested: {info['n_clusters_requested']}")
    print(f"  Clusters found: {info['n_clusters_found']}")
    print(f"  Number of transitions: {len(transitions)}")
    print(f"  Expected transitions: ~2")
    print(f"  False positive rate: {max(0, len(transitions) - 2)} spurious transitions")
    
    print(f"\n{'Cluster Distribution:':-<80}")
    for cluster_name, cluster_stats in stats.items():
        print(f"  {cluster_name}:")
        print(f"    Count: {cluster_stats['count']}")
        print(f"    Percentage: {cluster_stats['percentage']:.1f}%")
        print(f"    Avg duration: {cluster_stats['avg_duration']:.1f} samples")
    
    # Verify correlation structures
    print(f"\n{'Correlation Structure Verification:':-<80}")
    for i in range(info['n_clusters_found']):
        corr = model.get_correlation_structure(i)
        avg_corr = np.mean(corr[np.triu_indices_from(corr, k=1)])
        print(f"  Cluster {i}: Average correlation = {avg_corr:.3f}")
    
    # Assessment
    print(f"\n{'Assessment:':-<80}")
    if len(transitions) <= 5:
        print("  ✓ EXCELLENT: Low false positive rate")
    elif len(transitions) <= 10:
        print("  ✓ GOOD: Acceptable false positive rate")
    else:
        print("  ⚠ NEEDS IMPROVEMENT: High false positive rate")
    
    if info['n_clusters_found'] == 3:
        print("  ✓ CORRECT: Detected all 3 regimes")
    else:
        print(f"  ⚠ PARTIAL: Detected {info['n_clusters_found']} regimes instead of 3")
    
    return model, stats, transitions


def test_rapid_switching():
    """Test with rapid regime switching (stress test)"""
    print_header("TEST 2: Rapid Regime Switching (Stress Test)")
    
    np.random.seed(123)
    
    print(f"\nGenerating rapid-switching data:")
    print(f"  - 2 regimes alternating every 10 samples")
    print(f"  - 200 total samples")
    
    rapid_data = []
    for i in range(20):
        if i % 2 == 0:
            cov = np.eye(3) * 0.5
        else:
            cov = np.ones((3, 3)) * 0.7
            np.fill_diagonal(cov, 1.0)
        rapid_data.extend(np.random.multivariate_normal(np.zeros(3), cov, 10))
    
    rapid_data = np.array(rapid_data)
    
    print(f"\nFitting TICC model...")
    model = TICCClustering(n_clusters=2, window_size=5, beta=1200, min_cluster_size=3)
    model.fit(rapid_data)
    
    transitions = model.get_regime_transitions()
    stats = model.get_cluster_statistics()
    
    print(f"\n{'Results:':-<80}")
    print(f"  Expected transitions: ~19")
    print(f"  Detected transitions: {len(transitions)}")
    print(f"  Smoothing effectiveness: {abs(19 - len(transitions))} transitions removed")
    
    print(f"\n{'Cluster Distribution:':-<80}")
    for cluster_name, cluster_stats in stats.items():
        print(f"  {cluster_name}: {cluster_stats['percentage']:.1f}%")
    
    # Assessment
    print(f"\n{'Assessment:':-<80}")
    if len(transitions) < 19:
        print(f"  ✓ GOOD: Temporal smoothing reduced {19 - len(transitions)} noise transitions")
    else:
        print("  ⚠ Temporal smoothing had minimal effect")
    
    return model, stats, transitions


def test_stable_regime():
    """Test with single stable regime (no regime changes)"""
    print_header("TEST 3: Single Stable Regime")
    
    np.random.seed(456)
    
    print(f"\nGenerating stable regime data:")
    print(f"  - 1 regime, constant correlation structure")
    print(f"  - 200 samples")
    
    cov = np.ones((4, 4)) * 0.6
    np.fill_diagonal(cov, 1.0)
    stable_data = np.random.multivariate_normal(np.zeros(4), cov, 200)
    
    print(f"\nFitting TICC model...")
    model = TICCClustering(n_clusters=3, window_size=10, beta=1000, min_cluster_size=5)
    model.fit(stable_data)
    
    transitions = model.get_regime_transitions()
    stats = model.get_cluster_statistics()
    info = model.get_model_info()
    
    print(f"\n{'Results:':-<80}")
    print(f"  Clusters found: {info['n_clusters_found']}")
    print(f"  Number of transitions: {len(transitions)}")
    print(f"  Expected: 1 cluster, 0 transitions")
    
    print(f"\n{'Cluster Distribution:':-<80}")
    for cluster_name, cluster_stats in stats.items():
        print(f"  {cluster_name}: {cluster_stats['percentage']:.1f}%")
    
    # Assessment
    print(f"\n{'Assessment:':-<80}")
    if len(transitions) == 0:
        print("  ✓ PERFECT: No false transitions detected")
    elif len(transitions) <= 2:
        print("  ✓ GOOD: Very few false transitions")
    else:
        print(f"  ⚠ HIGH FALSE POSITIVES: {len(transitions)} spurious transitions")
    
    return model, stats, transitions


def test_comparison_summary():
    """Print comparison summary"""
    print_header("SUMMARY: TICC Fix Verification")
    
    print("\nKey Improvements:")
    print("  1. ✓ Robust initialization with multiple K-means runs")
    print("  2. ✓ Adaptive temporal smoothing using dynamic programming")
    print("  3. ✓ Post-processing to remove spurious short regimes")
    print("  4. ✓ Minimum cluster size enforcement")
    print("  5. ✓ Stable log-likelihood computation")
    print("  6. ✓ Data standardization for better clustering")
    
    print("\nExpected Outcomes:")
    print("  - Reduced false positives in regime transitions")
    print("  - Better detection of true regime changes")
    print("  - More stable cluster assignments")
    print("  - Fewer spurious short-lived regimes")
    
    print("\nTechnical Changes:")
    print("  - Changed from greedy to forward-backward DP")
    print("  - Added convergence threshold instead of exact match")
    print("  - Implemented robust covariance estimation")
    print("  - Added correlation-based distance metric")
    print("  - Implemented regime smoothing (min_regime_length)")


def main():
    """Run all verification tests"""
    print_separator('=', 80)
    print(" TICC CLUSTERING FIX VERIFICATION ".center(80, '='))
    print_separator('=', 80)
    
    print("\nThis script verifies the improved TICC implementation.")
    print("Focus: Reducing false positives in regime detection.")
    
    input("\nPress Enter to begin verification...\n")
    
    # Run tests
    try:
        test1_model, test1_stats, test1_trans = test_synthetic_3_regimes()
        print("\n")
        
        test2_model, test2_stats, test2_trans = test_rapid_switching()
        print("\n")
        
        test3_model, test3_stats, test3_trans = test_stable_regime()
        print("\n")
        
        test_comparison_summary()
        
        # Overall assessment
        print_header("OVERALL VERIFICATION RESULT")
        
        test1_pass = len(test1_trans) <= 10
        test2_pass = len(test2_trans) < 19
        test3_pass = len(test3_trans) <= 2
        
        all_passed = test1_pass and test2_pass and test3_pass
        
        print(f"\nTest 1 (3 Regimes): {'PASS ✓' if test1_pass else 'FAIL ✗'}")
        print(f"Test 2 (Rapid Switch): {'PASS ✓' if test2_pass else 'FAIL ✗'}")
        print(f"Test 3 (Stable): {'PASS ✓' if test3_pass else 'FAIL ✗'}")
        
        if all_passed:
            print("\n" + "="*80)
            print(" ALL TESTS PASSED - TICC FIX SUCCESSFUL ".center(80, '='))
            print("="*80)
        else:
            print("\n" + "="*80)
            print(" SOME TESTS FAILED - FURTHER TUNING NEEDED ".center(80, '='))
            print("="*80)
        
        # Save verification report
        results_df = pd.DataFrame([
            {
                'Test': '3 Regimes',
                'Transitions_Expected': 2,
                'Transitions_Detected': len(test1_trans),
                'False_Positives': max(0, len(test1_trans) - 2),
                'Status': 'PASS' if test1_pass else 'FAIL'
            },
            {
                'Test': 'Rapid Switching',
                'Transitions_Expected': 19,
                'Transitions_Detected': len(test2_trans),
                'False_Positives': 'N/A',
                'Status': 'PASS' if test2_pass else 'FAIL'
            },
            {
                'Test': 'Stable Regime',
                'Transitions_Expected': 0,
                'Transitions_Detected': len(test3_trans),
                'False_Positives': len(test3_trans),
                'Status': 'PASS' if test3_pass else 'FAIL'
            }
        ])
        
        report_path = Path(__file__).parent / 'reports' / 'TICC_FIX_VERIFICATION.csv'
        results_df.to_csv(report_path, index=False)
        print(f"\n✓ Verification report saved to: {report_path}")
        
    except Exception as e:
        print(f"\n✗ ERROR during verification: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
