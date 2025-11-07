"""
Simple BCD Test Runner - Generates visualizations
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.bayesian_changepoint import BayesianChangepoint
from utils.data_loader import DataLoader

# Test on synthetic data with known changepoints
print("=" * 80)
print("Bayesian Changepoint Detection - Generating Visualizations")
print("=" * 80)

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
data = []
true_changepoints = [250, 500, 750]

for i in range(n_samples):
    if i < 250:
        val = np.random.normal(0, 1)
    elif i < 500:
        val = np.random.normal(2, 1.5)
    elif i < 750:
        val = np.random.normal(-1, 0.8)
    else:
        val = np.random.normal(1, 2)
    data.append(val)

data = pd.Series(data)

print(f"\nSynthetic Data: {len(data)} observations")
print(f"True changepoints at: {true_changepoints}")

# Fit model
model = BayesianChangepoint(window_size=50, threshold=1.5)
model.fit(data)

# Plot
model.plot_changepoints(
    title="Synthetic Data - Bayesian Changepoint Detection",
    save_path='reports/BCD_synthetic_data.png',
    threshold=0.5
)

# Test on real data
print("\n" + "=" * 80)
print("Processing Real Market Data")
print("=" * 80)

loader = DataLoader()
test_symbols = ['NIFTY 50', 'NIFTY BANK', 'NIFTY IT']

for symbol in test_symbols:
    try:
        print(f"\nProcessing: {symbol}")
        df = loader.load_index(symbol)
        
        if len(df) > 5000:
            df = loader.resample_to_daily(df)
        
        returns = loader.calculate_returns(df, 'close', log_returns=True)
        returns = returns.tail(500)
        
        # Fit model
        model = BayesianChangepoint(window_size=30, threshold=1.8)
        model.fit(returns)
        
        # Save CSV report
        report_df = pd.DataFrame({
            'changepoint_probability': model.get_changepoint_probabilities(),
            'changepoint_detected_50': model.detect_changepoints(0.5),
            'changepoint_detected_75': model.detect_changepoints(0.75)
        })
        report_file = f"reports/BCD_{symbol.replace(' ', '_')}_report.csv"
        report_df.to_csv(report_file)
        print(f"  Report saved: {report_file}")
        
        # Save visualization
        plot_file = f"reports/BCD_{symbol.replace(' ', '_')}.png"
        model.plot_changepoints(
            title=f"{symbol} - Bayesian Changepoint Detection",
            save_path=plot_file,
            threshold=0.75
        )
        
        stats = model.get_statistics()
        print(f"  Changepoints (>75%): {stats['n_significant_cp_75']}")
        print(f"  Max probability: {stats['max_cp_prob']:.4f}")
        
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 80)
print("BCD Visualization Generation Complete")
print("=" * 80)
print("\nCheck the reports/ folder for PNG visualizations:")
print("  - BCD_synthetic_data.png")
print("  - BCD_NIFTY_50.png")
print("  - BCD_NIFTY_BANK.png")
print("  - BCD_NIFTY_IT.png")
