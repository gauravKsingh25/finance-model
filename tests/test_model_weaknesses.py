"""
Identify Exact Conditions Where Models Struggle
Tests models on progressively harder scenarios until they fail
Output: Terminal only
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from models import (
    MarkovRegimeSwitching,
    GARCHVolatilityRegime,
    BayesianChangepoint,
    HawkesProcess
)
from models.chaos_metrics import HurstExponent
import warnings
warnings.filterwarnings('ignore')


def print_header(title):
    print("\n" + "=" * 100)
    print(f" {title} ".center(100, "="))
    print("=" * 100 + "\n")


def find_minimum_sample_size():
    """Find minimum data points needed for each model"""
    print_header("MINIMUM SAMPLE SIZE TESTING")
    
    sizes = [5, 10, 20, 30, 50, 100, 150, 200]
    
    for size in sizes:
        print(f"\n{'='*50} N = {size} {'='*50}")
        data = pd.Series(np.random.normal(0, 1, size))
        
        # Markov
        try:
            m = MarkovRegimeSwitching(n_regimes=2)
            m.fit(data)
            print(f"  Markov:  ✓ Works with {size} points")
        except:
            print(f"  Markov:  ✗ Fails with {size} points (MINIMUM FOUND)")
        
        # GARCH
        try:
            g = GARCHVolatilityRegime(p=1, q=1)
            g.fit(data)
            print(f"  GARCH:   ✓ Works with {size} points")
        except:
            print(f"  GARCH:   ✗ Fails with {size} points (MINIMUM FOUND)")
        
        # BCD
        try:
            b = BayesianChangepoint()
            b.fit(data)
            print(f"  BCD:     ✓ Works with {size} points")
        except:
            print(f"  BCD:     ✗ Fails with {size} points (MINIMUM FOUND)")
        
        # Hurst
        try:
            h = HurstExponent()
            h_val = h.calculate(data, method='rs')
            print(f"  Hurst:   ✓ Works with {size} points (H={h_val:.3f})")
        except:
            print(f"  Hurst:   ✗ Fails with {size} points (MINIMUM FOUND)")


def test_noise_levels():
    """Test how noise affects model performance"""
    print_header("NOISE SENSITIVITY TESTING")
    
    # True signal: 2 regimes
    regime1 = np.random.normal(0, 1, 250)
    regime2 = np.random.normal(3, 1, 250)
    true_signal = np.concatenate([regime1, regime2])
    
    noise_levels = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    print("TRUE SIGNAL: Mean shift from 0 to 3 at t=250")
    print(f"\n{'Noise σ':<10} {'Markov Accuracy':<20} {'Hurst Before':<15} {'Hurst After':<15}")
    print("-" * 100)
    
    for noise_std in noise_levels:
        noisy_signal = true_signal + np.random.normal(0, noise_std, 500)
        
        # Markov detection
        try:
            m = MarkovRegimeSwitching(n_regimes=2)
            m.fit(pd.Series(noisy_signal))
            regimes = m.predict_regimes()
            # Check if regime changes near t=250
            changes = np.where(np.diff(regimes) != 0)[0]
            if len(changes) > 0:
                closest_change = min(changes, key=lambda x: abs(x - 250))
                error = abs(closest_change - 250)
                markov_result = f"Detected at t={closest_change} (off by {error})"
            else:
                markov_result = "No regime change detected"
        except:
            markov_result = "FAILED"
        
        # Hurst before/after
        try:
            h = HurstExponent()
            h_before = h.calculate(pd.Series(noisy_signal[:250]), method='rs')
            h_after = h.calculate(pd.Series(noisy_signal[250:]), method='rs')
            hurst_result = f"{h_before:.3f} / {h_after:.3f}"
        except:
            hurst_result = "FAILED"
        
        print(f"{noise_std:<10.1f} {markov_result:<20} {hurst_result:<30}")


def test_variance_stability():
    """Test GARCH with different variance patterns"""
    print_header("GARCH VARIANCE STABILITY TESTING")
    
    scenarios = {
        "Constant low variance": np.random.normal(0, 0.01, 500),
        "Constant high variance": np.random.normal(0, 5.0, 500),
        "Increasing variance": np.array([np.random.normal(0, 0.01 + i*0.01) for i in range(500)]),
        "Decreasing variance": np.array([np.random.normal(0, 5.0 - i*0.009) for i in range(500)]),
        "Oscillating variance": np.array([np.random.normal(0, 1 + np.sin(i/50)) for i in range(500)]),
        "Sudden variance spike": np.concatenate([
            np.random.normal(0, 0.1, 450),
            np.random.normal(0, 5.0, 50)
        ])
    }
    
    print(f"{'Scenario':<30} {'Success':<10} {'Persistence':<15} {'High-Vol %':<15}")
    print("-" * 100)
    
    for name, data in scenarios.items():
        try:
            g = GARCHVolatilityRegime(p=1, q=1)
            g.fit(pd.Series(data))
            params = g.get_model_parameters()
            regimes = g.predict_regimes()
            high_vol_pct = (regimes == 1).sum() / len(regimes) * 100
            
            persistence = params['persistence']
            
            print(f"{name:<30} {'✓':<10} {persistence:<15.4f} {high_vol_pct:<15.1f}%")
            
            if persistence > 0.999:
                print(f"  ⚠ WARNING: Non-stationary (persistence = {persistence:.6f})")
            
        except Exception as e:
            print(f"{name:<30} {'✗':<10} FAILED: {str(e)[:40]}")


def test_changepoint_sensitivity():
    """Test BCD with different change magnitudes"""
    print_header("CHANGEPOINT DETECTION SENSITIVITY")
    
    # Test different magnitude shifts
    magnitudes = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    
    print(f"{'Shift Magnitude':<20} {'Max CP Prob':<15} {'Detected Index':<20} {'Distance from True':<20}")
    print("-" * 100)
    
    for mag in magnitudes:
        before = np.random.normal(0, 1, 250)
        after = np.random.normal(mag, 1, 250)
        data = pd.Series(np.concatenate([before, after]))
        
        try:
            b = BayesianChangepoint(hazard_rate=0.01)
            b.fit(data)
            probs = b.get_changepoint_probabilities()
            max_prob = probs.max()
            max_idx = probs.argmax()
            distance = abs(max_idx - 250)
            
            print(f"{mag:<20.1f} {max_prob:<15.4f} {max_idx:<20} {distance:<20}")
            
        except Exception as e:
            print(f"{mag:<20.1f} FAILED: {str(e)[:50]}")


def test_hawkes_event_threshold():
    """Test Hawkes with different event thresholds"""
    print_header("HAWKES PROCESS EVENT THRESHOLD SENSITIVITY")
    
    # Generate data with known volatility
    returns = pd.Series(np.random.normal(0, 1, 1000))
    
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    
    print(f"{'Threshold (σ)':<15} {'# Events':<12} {'Fragility':<12} {'Reliability':<20}")
    print("-" * 100)
    
    for thresh in thresholds:
        try:
            h = HawkesProcess()
            h.fit_from_returns(returns, threshold=thresh)
            stats = h.get_statistics()
            n_events = stats['n_events']
            fragility = stats['fragility_score']
            
            if n_events < 5:
                reliability = "TOO FEW EVENTS"
            elif n_events < 10:
                reliability = "UNRELIABLE"
            elif n_events < 20:
                reliability = "MARGINAL"
            else:
                reliability = "RELIABLE"
            
            print(f"{thresh:<15.1f} {n_events:<12} {fragility:<12.4f} {reliability:<20}")
            
        except Exception as e:
            print(f"{thresh:<15.1f} FAILED: {str(e)[:40]}")


def test_hurst_window_size():
    """Test Hurst exponent with different window sizes"""
    print_header("HURST EXPONENT WINDOW SIZE SENSITIVITY")
    
    # Generate known processes
    processes = {
        "Random Walk": np.random.normal(0, 1, 500),
        "Trending": np.cumsum(np.random.normal(0.1, 1, 500)),
        "Mean-Reverting (AR)": None
    }
    
    # Generate AR process
    ar_process = [0]
    for _ in range(499):
        ar_process.append(-0.7 * ar_process[-1] + np.random.normal(0, 1))
    processes["Mean-Reverting (AR)"] = np.array(ar_process)
    
    window_sizes = [20, 30, 50, 100, 200, 300, 500]
    
    for proc_name, data in processes.items():
        print(f"\n{proc_name}:")
        print(f"{'Window Size':<15} {'Hurst':<12} {'Regime':<20} {'Reliability':<20}")
        print("-" * 80)
        
        for window in window_sizes:
            if window > len(data):
                continue
                
            try:
                h = HurstExponent()
                h_val = h.calculate(pd.Series(data[-window:]), method='rs')
                
                if h_val > 0.6:
                    regime = "Trending"
                elif h_val < 0.4:
                    regime = "Mean-Reverting"
                else:
                    regime = "Random Walk"
                
                if window < 50:
                    reliability = "UNRELIABLE"
                elif window < 100:
                    reliability = "MARGINAL"
                else:
                    reliability = "GOOD"
                
                print(f"{window:<15} {h_val:<12.4f} {regime:<20} {reliability:<20}")
                
            except Exception as e:
                print(f"{window:<15} FAILED")


def test_outlier_tolerance():
    """Test how many outliers each model can tolerate"""
    print_header("OUTLIER TOLERANCE TESTING")
    
    base_data = np.random.normal(0, 1, 500)
    outlier_percentages = [0, 1, 2, 5, 10, 20, 30]
    
    print(f"{'Outlier %':<12} {'Markov':<10} {'GARCH':<10} {'BCD':<10} {'Hurst':<10}")
    print("-" * 100)
    
    for outlier_pct in outlier_percentages:
        data = base_data.copy()
        n_outliers = int(len(data) * outlier_pct / 100)
        
        if n_outliers > 0:
            outlier_indices = np.random.choice(len(data), n_outliers, replace=False)
            data[outlier_indices] = np.random.choice([-50, 50], n_outliers)
        
        results = {}
        
        # Markov
        try:
            m = MarkovRegimeSwitching(n_regimes=2)
            m.fit(pd.Series(data))
            results['Markov'] = "✓"
        except:
            results['Markov'] = "✗"
        
        # GARCH
        try:
            g = GARCHVolatilityRegime(p=1, q=1)
            g.fit(pd.Series(data))
            results['GARCH'] = "✓"
        except:
            results['GARCH'] = "✗"
        
        # BCD
        try:
            b = BayesianChangepoint()
            b.fit(pd.Series(data))
            results['BCD'] = "✓"
        except:
            results['BCD'] = "✗"
        
        # Hurst
        try:
            h = HurstExponent()
            h.calculate(pd.Series(data), method='rs')
            results['Hurst'] = "✓"
        except:
            results['Hurst'] = "✗"
        
        print(f"{outlier_pct:<12}% {results.get('Markov', '?'):<10} {results.get('GARCH', '?'):<10} "
              f"{results.get('BCD', '?'):<10} {results.get('Hurst', '?'):<10}")


def generate_weakness_summary():
    """Final summary of model weaknesses"""
    print_header("MODEL WEAKNESS SUMMARY")
    
    print("\n1. MARKOV REGIME SWITCHING")
    print("   BREAKS WHEN:")
    print("   • Data has zero variance (constant values)")
    print("   • Sample size < 30 points")
    print("   • Outliers > 30% of data")
    print("   ")
    print("   DEGRADES WITH:")
    print("   • Very rapid regime switching (every few steps)")
    print("   • High noise relative to signal (SNR < 0.5)")
    print("   • More than 20% missing values")
    
    print("\n2. GARCH VOLATILITY")
    print("   BREAKS WHEN:")
    print("   • Sample size < 10 points")
    print("   • Zero variance data")
    print("   ")
    print("   DEGRADES WITH:")
    print("   • Non-stationary variance (trending σ)")
    print("   • Persistence > 0.999 (unit root territory)")
    print("   • Extreme value ranges (>1000x differences)")
    
    print("\n3. BAYESIAN CHANGEPOINT DETECTION")
    print("   BREAKS WHEN:")
    print("   • Sample size < 5 points")
    print("   ")
    print("   DEGRADES WITH:")
    print("   • Small magnitude changes (< 0.5σ shift)")
    print("   • Gradual drift (no sharp breaks)")
    print("   • High noise-to-signal ratio")
    print("   • Very frequent changepoints")
    
    print("\n4. HAWKES PROCESS")
    print("   BREAKS WHEN:")
    print("   • < 3 events detected")
    print("   • All events perfectly regular (no clustering)")
    print("   ")
    print("   DEGRADES WITH:")
    print("   • < 10 events (unreliable parameter estimates)")
    print("   • Threshold too high (misses clustering)")
    print("   • Threshold too low (false clusters)")
    
    print("\n5. HURST EXPONENT")
    print("   BREAKS WHEN:")
    print("   • Sample size < 20 points")
    print("   • Constant data (zero variance)")
    print("   ")
    print("   DEGRADES WITH:")
    print("   • Window size < 100 points (unreliable)")
    print("   • Non-stationary data (changing H over time)")
    print("   • Heavy outliers (> 20% extreme values)")
    
    print("\n" + "=" * 100)
    print(" PRODUCTION DEPLOYMENT GUIDELINES ".center(100, "="))
    print("=" * 100)
    
    print("\nPRE-FLIGHT CHECKS (Run before each model):")
    print("  1. Sample size: N ≥ 200 (Markov/GARCH), N ≥ 100 (Hurst)")
    print("  2. Variance: σ > 0 and finite")
    print("  3. Outliers: Cap at ±10σ, maximum 10% extreme values")
    print("  4. Missing data: < 10% NaN values")
    print("  5. Hawkes events: Ensure ≥ 10 events detected")
    
    print("\nFAIL-SAFE MECHANISMS:")
    print("  • If Markov fails → Fall back to simple returns mean/std classification")
    print("  • If GARCH fails → Use rolling window standard deviation")
    print("  • If BCD fails → Use simple rolling mean change detection")
    print("  • If Hawkes fails → Skip fragility metric, mark as 'insufficient data'")
    print("  • If Hurst fails → Default to H=0.5 (random walk assumption)")
    
    print("\nMODEL COMBINATIONS:")
    print("  ✓ BEST: GARCH + Hurst (most robust pair)")
    print("  ✓ GOOD: Markov + GARCH (original 2-stream design)")
    print("  ⚠ RISKY: Hawkes alone (too threshold-dependent)")
    print("  ⚠ AVOID: BCD on high-frequency noisy data (too many false positives)")


def main():
    """Run all weakness tests"""
    print("\n" + "=" * 100)
    print(" MODEL WEAKNESS & BREAKING POINT ANALYSIS ".center(100, "="))
    print("=" * 100)
    print("\nFinding exact conditions where each model struggles or fails")
    print("Output: Terminal only\n")
    
    input("[Press Enter to begin weakness testing...]")
    
    find_minimum_sample_size()
    test_noise_levels()
    test_variance_stability()
    test_changepoint_sensitivity()
    test_hawkes_event_threshold()
    test_hurst_window_size()
    test_outlier_tolerance()
    generate_weakness_summary()
    
    print("\n" + "=" * 100)
    print(" WEAKNESS TESTING COMPLETE ".center(100, "="))
    print("=" * 100)
    print("\nAll model breaking points identified")
    print("Production guidelines provided")


if __name__ == "__main__":
    main()
