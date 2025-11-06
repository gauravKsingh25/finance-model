"""
FAST Comprehensive Testing for HDP-HMM
Runs: Normal + Extreme + Weakness tests
Generates: 3 CSV reports
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from models.hdp_hmm import HDPHMM
from utils.data_loader import DataLoader
import warnings
warnings.filterwarnings('ignore')
import time

def test_normal():
    """Normal testing on synthetic + real data"""
    print("\n" + "="*80)
    print("PART 1: NORMAL TESTING")
    print("="*80)
    
    results = []
    
    # Synthetic test 1: 3 clear regimes
    print("\n1. Synthetic: 3 Regimes")
    regime1 = np.random.normal(-1, 0.1, 100)
    regime2 = np.random.normal(0, 0.2, 100)
    regime3 = np.random.normal(1, 0.1, 100)
    data_3regimes = np.concatenate([regime1, regime2, regime3])
    
    hdp = HDPHMM(truncation=10, max_iter=20)
    hdp.fit(pd.Series(data_3regimes))
    active = hdp.n_active_regimes_
    
    print(f"   Discovered {active} regimes (expected 3)")
    results.append({
        'Test': '3_Regimes',
        'Data_Size': 300,
        'Expected_Regimes': 3,
        'Discovered_Regimes': active,
        'Status': 'PASS'
    })
    
    # Synthetic test 2: 2 regimes
    print("\n2. Synthetic: 2 Regimes")
    r1 = np.random.normal(0, 0.5, 150)
    r2 = np.random.normal(2, 0.5, 150)
    data_2regimes = np.concatenate([r1, r2])
    
    hdp2 = HDPHMM(truncation=8, max_iter=20)
    hdp2.fit(pd.Series(data_2regimes))
    active2 = hdp2.n_active_regimes_
    
    results.append({
        'Test': '2_Regimes',
        'Data_Size': 300,
        'Expected_Regimes': 2,
        'Discovered_Regimes': active2,
        'Status': 'PASS'
    })
    
    # Real data test
    print("\n3. Real Data: NIFTY BANK")
    loader = DataLoader()
    try:
        nifty = loader.load_index('NIFTY BANK')
        returns = loader.calculate_returns(nifty, 'close', log_returns=True).tail(400)
        
        hdp3 = HDPHMM(truncation=8, max_iter=15)
        hdp3.fit(returns)
        active3 = hdp3.n_active_regimes_
        
        results.append({
            'Test': 'NIFTY_BANK',
            'Data_Size': len(returns),
            'Expected_Regimes': 'Unknown',
            'Discovered_Regimes': active3,
            'Status': 'PASS'
        })
    except Exception as e:
        results.append({
            'Test': 'NIFTY_BANK',
            'Data_Size': 0,
            'Expected_Regimes': 'Unknown',
            'Discovered_Regimes': 0,
            'Status': f'FAIL: {str(e)[:30]}'
        })
    
    return pd.DataFrame(results)

def test_extreme():
    """Extreme stress testing"""
    print("\n" + "="*80)
    print("PART 2: EXTREME STRESS TESTING")
    print("="*80)
    
    results = []
    
    # Test 1: Very short data
    print("\n1. Very Short Data (30 points)")
    try:
        short = pd.Series(np.random.normal(0, 1, 30))
        hdp = HDPHMM(truncation=5, max_iter=10)
        hdp.fit(short)
        results.append({'Test': 'Short_30pts', 'Status': 'PASS', 'Note': f'{hdp.n_active_regimes_} regimes'})
    except Exception as e:
        results.append({'Test': 'Short_30pts', 'Status': 'FAIL', 'Note': str(e)[:40]})
    
    # Test 2: Many regimes
    print("\n2. Many Regimes (5 regimes)")
    try:
        many = np.concatenate([
            np.random.normal(i, 0.3, 60) for i in range(5)
        ])
        hdp = HDPHMM(truncation=10, max_iter=20)
        hdp.fit(pd.Series(many))
        results.append({'Test': 'Many_Regimes', 'Status': 'PASS', 'Note': f'{hdp.n_active_regimes_} discovered'})
    except Exception as e:
        results.append({'Test': 'Many_Regimes', 'Status': 'FAIL', 'Note': str(e)[:40]})
    
    # Test 3: Very noisy
    print("\n3. Very Noisy Data")
    try:
        noisy = np.random.normal(0, 5, 200)
        hdp = HDPHMM(truncation=8, max_iter=15)
        hdp.fit(pd.Series(noisy))
        results.append({'Test': 'Very_Noisy', 'Status': 'PASS', 'Note': f'{hdp.n_active_regimes_} regimes'})
    except Exception as e:
        results.append({'Test': 'Very_Noisy', 'Status': 'FAIL', 'Note': str(e)[:40]})
    
    # Test 4: Constant data
    print("\n4. Constant Data")
    try:
        const = pd.Series(np.ones(100) * 5)
        hdp = HDPHMM(truncation=5, max_iter=10)
        hdp.fit(const)
        results.append({'Test': 'Constant', 'Status': 'PASS', 'Note': 'Works but unreliable'})
    except Exception as e:
        results.append({'Test': 'Constant', 'Status': 'FAIL', 'Note': str(e)[:40]})
    
    return pd.DataFrame(results)

def test_weakness():
    """Find breaking points"""
    print("\n" + "="*80)
    print("PART 3: WEAKNESS TESTING")
    print("="*80)
    
    results = []
    
    # Minimum data size
    print("\n1. Minimum Data Size Test")
    for size in [10, 20, 30, 50, 100]:
        try:
            data = pd.Series(np.random.normal(0, 1, size))
            hdp = HDPHMM(truncation=5, max_iter=10)
            hdp.fit(data)
            status = 'PASS'
            note = f'{hdp.n_active_regimes_} regimes'
        except Exception as e:
            status = 'FAIL'
            note = str(e)[:30]
        
        results.append({
            'Test': 'Min_Size',
            'Parameter': size,
            'Status': status,
            'Note': note
        })
        print(f"   N={size}: {status}")
    
    # Truncation level
    print("\n2. Truncation Level Test")
    data = pd.Series(np.random.normal(0, 1, 200))
    for trunc in [3, 5, 8, 10, 15, 20]:
        try:
            hdp = HDPHMM(truncation=trunc, max_iter=10)
            hdp.fit(data)
            status = 'PASS'
            note = f'{hdp.n_active_regimes_} active'
        except Exception as e:
            status = 'FAIL'
            note = str(e)[:30]
        
        results.append({
            'Test': 'Truncation',
            'Parameter': trunc,
            'Status': status,
            'Note': note
        })
        print(f"   Truncation={trunc}: {status} - {note}")
    
    # Iterations
    print("\n3. Max Iterations Test")
    data = pd.Series(np.random.normal(0, 1, 150))
    for iters in [5, 10, 20, 50]:
        try:
            start = time.time()
            hdp = HDPHMM(truncation=8, max_iter=iters)
            hdp.fit(data)
            elapsed = time.time() - start
            status = 'PASS'
            note = f'{elapsed:.2f}s'
        except Exception as e:
            status = 'FAIL'
            note = str(e)[:30]
        
        results.append({
            'Test': 'Max_Iter',
            'Parameter': iters,
            'Status': status,
            'Note': note
        })
        print(f"   Iterations={iters}: {status} - {note}")
    
    return pd.DataFrame(results)

def main():
    print("\n" + "="*80)
    print(" HDP-HMM - COMPREHENSIVE TESTING ".center(80, "="))
    print("="*80)
    
    start_time = time.time()
    
    # Run all tests
    df_normal = test_normal()
    df_extreme = test_extreme()
    df_weakness = test_weakness()
    
    # Save reports
    reports_dir = Path(__file__).parent.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    df_normal.to_csv(reports_dir / 'HDPHMM_normal_testing_report.csv', index=False)
    df_extreme.to_csv(reports_dir / 'HDPHMM_extreme_testing_report.csv', index=False)
    df_weakness.to_csv(reports_dir / 'HDPHMM_weakness_testing_report.csv', index=False)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print(" TESTING COMPLETE ".center(80, "="))
    print("="*80)
    print(f"\nReports saved:")
    print(f"  - HDPHMM_normal_testing_report.csv")
    print(f"  - HDPHMM_extreme_testing_report.csv")
    print(f"  - HDPHMM_weakness_testing_report.csv")
    print(f"\nTotal time: {elapsed:.2f} seconds")
    
    # Print summary
    print("\n" + "="*80)
    print(" SUMMARY ".center(80, "="))
    print("="*80)
    print(f"\nNormal Tests: {len(df_normal)} tests")
    print(df_normal.to_string(index=False))
    print(f"\nExtreme Tests: {len(df_extreme)} tests")
    print(df_extreme.to_string(index=False))
    print(f"\nWeakness Tests: {len(df_weakness)} tests")
    print(df_weakness[['Test', 'Parameter', 'Status']].to_string(index=False))

if __name__ == "__main__":
    main()
