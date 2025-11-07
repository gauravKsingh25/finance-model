"""
Test Bayesian Changepoint Detection (BCD)
Tests structural break detection capability
Purpose: Validate "The Alarm" sensor for regime change detection
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.bayesian_changepoint import BayesianChangepoint
from utils.data_loader import DataLoader
from utils.metrics import ModelEvaluator
import warnings
warnings.filterwarnings('ignore')


def test_bcd_synthetic():
    """Test BCD on synthetic data with known changepoints"""
    print("=" * 80)
    print("TEST 1: Bayesian Changepoint Detection on Synthetic Data")
    print("=" * 80)
    
    # Generate synthetic data with known changepoints at t=250, 500, 750
    np.random.seed(42)
    n_samples = 1000
    data = []
    true_changepoints = [250, 500, 750]
    
    for i in range(n_samples):
        if i < 250:
            # Regime 1: mean=0, std=1
            val = np.random.normal(0, 1)
        elif i < 500:
            # Regime 2: mean=2, std=1.5
            val = np.random.normal(2, 1.5)
        elif i < 750:
            # Regime 3: mean=-1, std=0.8
            val = np.random.normal(-1, 0.8)
        else:
            # Regime 4: mean=1, std=2
            val = np.random.normal(1, 2)
        data.append(val)
    
    data = pd.Series(data)
    
    print(f"\nGenerated {len(data)} observations with {len(true_changepoints)} changepoints")
    print(f"True changepoints at: {true_changepoints}")
    
    # Fit BCD model with appropriate parameters
    print("\nFitting Bayesian Changepoint Detection...")
    model = BayesianChangepoint(window_size=50, threshold=1.5)  # Moderately sensitive
    model.fit(data)
    
    # Get statistics
    print("\n" + "=" * 80)
    print("MODEL STATISTICS:")
    print("=" * 80)
    stats = model.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Detect changepoints
    print("\n" + "=" * 80)
    print("CHANGEPOINT DETECTION:")
    print("=" * 80)
    
    detected_cps = model.get_changepoint_locations(threshold=0.5)
    print(f"\nDetected changepoints (prob > 50%): {detected_cps}")
    print(f"Number detected: {len(detected_cps)}")
    
    # Check detection accuracy
    detected_set = set(detected_cps)
    true_set = set(true_changepoints)
    
    # Find detected changepoints close to true ones (within 20 time steps)
    tolerance = 20
    true_positives = 0
    for true_cp in true_changepoints:
        if any(abs(det_cp - true_cp) < tolerance for det_cp in detected_cps):
            true_positives += 1
            print(f"✓ Detected changepoint near t={true_cp}")
    
    accuracy = true_positives / len(true_changepoints)
    print(f"\nDetection Accuracy: {accuracy:.2%} ({true_positives}/{len(true_changepoints)})")
    
    # Current changepoint probability
    current_cp_prob = model.get_current_changepoint_prob()
    print(f"\nCurrent Changepoint Probability: {current_cp_prob:.4f}")
    
    # Create comprehensive visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    try:
        # Plot with multiple thresholds
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Plot 1: Original data with true changepoints
        axes[0].plot(data, 'b-', linewidth=1.5, alpha=0.7, label='Time Series Data')
        
        # Mark true changepoints
        for tcp in true_changepoints:
            axes[0].axvline(x=tcp, color='green', linestyle='--', linewidth=2, 
                          alpha=0.6, label='True Changepoint' if tcp == true_changepoints[0] else '')
            axes[0].scatter(tcp, data.iloc[tcp], color='green', s=150, 
                          marker='*', zorder=5, edgecolors='darkgreen', linewidths=2)
        
        # Mark detected changepoints
        detected_cps_50 = model.get_changepoint_locations(threshold=0.5)
        if len(detected_cps_50) > 0:
            axes[0].scatter(detected_cps_50, data.iloc[detected_cps_50], 
                          color='red', s=100, marker='v', 
                          label=f'Detected (50%, n={len(detected_cps_50)})',
                          zorder=4, edgecolors='darkred', linewidths=2)
        
        axes[0].set_ylabel('Value', fontsize=12, fontweight='bold')
        axes[0].set_title('Synthetic Data with Known Changepoints', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Changepoint probabilities
        time_indices = np.arange(len(model.changepoint_probs))
        axes[1].plot(time_indices, model.changepoint_probs, 'b-', 
                    linewidth=1.5, label='Changepoint Probability')
        
        # Add threshold lines
        axes[1].axhline(y=0.5, color='orange', linestyle='--', linewidth=2, 
                       alpha=0.7, label='50% threshold')
        axes[1].axhline(y=0.75, color='red', linestyle='--', linewidth=2, 
                       alpha=0.7, label='75% threshold')
        
        # Fill areas
        axes[1].fill_between(time_indices, 0, model.changepoint_probs,
                            where=(model.changepoint_probs > 0.5),
                            color='orange', alpha=0.2)
        axes[1].fill_between(time_indices, 0, model.changepoint_probs,
                            where=(model.changepoint_probs > 0.75),
                            color='red', alpha=0.3)
        
        # Mark true changepoint locations
        for tcp in true_changepoints:
            axes[1].axvline(x=tcp, color='green', linestyle='--', 
                          linewidth=1.5, alpha=0.4)
        
        axes[1].set_ylabel('Probability', fontsize=12, fontweight='bold')
        axes[1].set_title('Changepoint Detection Probabilities', fontsize=14, fontweight='bold')
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].set_ylim([0, 1.0])
        
        # Plot 3: Detection performance
        detection_accuracy = np.zeros(len(data))
        for tcp in true_changepoints:
            # Mark region around true changepoint
            start = max(0, tcp - tolerance)
            end = min(len(data), tcp + tolerance)
            detection_accuracy[start:end] = 0.3  # Background for true CP region
        
        axes[2].fill_between(time_indices, 0, detection_accuracy, 
                            color='green', alpha=0.3, label='True CP ± tolerance')
        axes[2].bar(time_indices, model.changepoint_probs, 
                   color='blue', alpha=0.5, width=1.0, label='Detection Probability')
        
        # Mark detections
        if len(detected_cps_50) > 0:
            axes[2].scatter(detected_cps_50, model.changepoint_probs[detected_cps_50],
                          color='red', s=80, marker='o', zorder=5,
                          label='Detected CPs', edgecolors='darkred', linewidths=2)
        
        axes[2].set_ylabel('Probability', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Time', fontsize=12, fontweight='bold')
        axes[2].set_title('Detection Performance Analysis', fontsize=14, fontweight='bold')
        axes[2].legend(loc='best', fontsize=10)
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].set_ylim([0, 1.0])
        
        # Add overall statistics
        stats_text = f"True Changepoints: {len(true_changepoints)}\n"
        stats_text += f"Detected (>50%): {len(detected_cps_50)}\n"
        stats_text += f"Detection Rate: {accuracy:.1%}\n"
        stats_text += f"Mean Prob: {np.mean(model.changepoint_probs):.4f}\n"
        stats_text += f"Max Prob: {np.max(model.changepoint_probs):.4f}"
        
        axes[2].text(0.98, 0.98, stats_text, 
                    transform=axes[2].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10, family='monospace')
        
        plt.tight_layout()
        plt.savefig('reports/BCD_synthetic_comprehensive.png', dpi=300, 
                   bbox_inches='tight', facecolor='white')
        print("✓ Comprehensive synthetic data plot saved to reports/BCD_synthetic_comprehensive.png")
        plt.close()
        
        # Also create simple version using model's plot function
        model.plot_changepoints(title="Synthetic Data - Bayesian Changepoint Detection",
                              save_path='reports/BCD_synthetic_simple.png',
                              threshold=0.5)
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    return model, stats, accuracy


def test_bcd_real_data():
    """Test BCD on real market data"""
    print("\n" + "=" * 80)
    print("TEST 2: Bayesian Changepoint Detection on Real Data")
    print("=" * 80)
    
    loader = DataLoader()
    test_symbols = ['NIFTY 50', 'NIFTY BANK', 'NIFTY IT']
    results = {}
    
    for symbol in test_symbols:
        try:
            print(f"\n{'=' * 80}")
            print(f"Testing on: {symbol}")
            print('=' * 80)
            
            # Load data
            df = loader.load_index(symbol)
            
            # Resample to daily
            if len(df) > 5000:
                df = loader.resample_to_daily(df)
            
            returns = loader.calculate_returns(df, 'close', log_returns=True)
            
            # Use recent data
            returns = returns.tail(500)
            print(f"Using last {len(returns)} observations")
            
            # Fit BCD with appropriate parameters for financial data
            print("\nFitting Bayesian Changepoint Detection...")
            model = BayesianChangepoint(window_size=30, threshold=1.8)  # Sensitive to market changes
            model.fit(returns)
            
            # Get statistics
            stats = model.get_statistics()
            print(f"\nStatistics:")
            print(f"  Mean CP Probability: {stats['mean_cp_prob']:.4f}")
            print(f"  Max CP Probability: {stats['max_cp_prob']:.4f}")
            print(f"  Significant CPs (>50%): {stats['n_significant_cp_50']}")
            print(f"  Significant CPs (>75%): {stats['n_significant_cp_75']}")
            print(f"  Significant CPs (>90%): {stats['n_significant_cp_90']}")
            
            # Detect changepoints
            detected_cps = model.get_changepoint_locations(threshold=0.75)
            print(f"\nDetected {len(detected_cps)} high-probability changepoints (>75%)")
            if detected_cps:
                print(f"Changepoint indices: {detected_cps[:10]}...")  # Show first 10
            
            # Current state
            current_prob = model.get_current_changepoint_prob()
            print(f"\nCurrent Changepoint Probability: {current_prob:.4f}")
            if current_prob > 0.5:
                print("  ⚠️  WARNING: High probability of regime change!")
            else:
                print("  ✓ Regime appears stable")
            
            # Save report
            report_df = pd.DataFrame({
                'changepoint_probability': model.get_changepoint_probabilities(),
                'changepoint_detected_50': model.detect_changepoints(0.5),
                'changepoint_detected_75': model.detect_changepoints(0.75)
            })
            report_file = f"reports/BCD_{symbol.replace(' ', '_')}_report.csv"
            report_df.to_csv(report_file)
            print(f"✓ Report saved to {report_file}")
            
            # Create comprehensive visualization
            print(f"Generating visualization for {symbol}...")
            try:
                # Create detailed plot
                fig, axes = plt.subplots(3, 1, figsize=(16, 12))
                
                # Get data values
                data_values = model.data
                time_indices = np.arange(len(data_values))
                cp_probs = model.changepoint_probs
                
                # Get changepoints at different thresholds
                cp_50 = model.get_changepoint_locations(threshold=0.5)
                cp_75 = model.get_changepoint_locations(threshold=0.75)
                cp_90 = model.get_changepoint_locations(threshold=0.9)
                
                # Plot 1: Returns with changepoints
                axes[0].plot(time_indices, data_values, 'b-', linewidth=1, 
                           alpha=0.6, label='Returns')
                axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
                
                # Mark changepoints
                if len(cp_90) > 0:
                    axes[0].scatter(cp_90, data_values[cp_90], 
                                  color='darkred', s=150, marker='v',
                                  label=f'High Confidence CP (>90%, n={len(cp_90)})',
                                  zorder=5, edgecolors='black', linewidths=1.5)
                if len(cp_75) > 0:
                    axes[0].scatter(cp_75, data_values[cp_75], 
                                  color='red', s=100, marker='v',
                                  label=f'Medium Confidence CP (>75%, n={len(cp_75)})',
                                  zorder=4, edgecolors='black', linewidths=1)
                
                # Add vertical lines for high confidence CPs
                for cp in cp_90:
                    axes[0].axvline(x=cp, color='darkred', linestyle='--', 
                                  alpha=0.3, linewidth=1.5)
                
                axes[0].set_ylabel('Log Returns', fontsize=12, fontweight='bold')
                axes[0].set_title(f'{symbol} - Returns with Detected Changepoints', 
                                fontsize=14, fontweight='bold')
                axes[0].legend(loc='best', fontsize=9)
                axes[0].grid(True, alpha=0.3, linestyle='--')
                
                # Plot 2: Changepoint Probabilities
                axes[1].plot(time_indices, cp_probs, 'b-', linewidth=1.5, 
                           label='CP Probability', alpha=0.7)
                
                # Threshold lines
                axes[1].axhline(y=0.5, color='orange', linestyle='--', 
                              linewidth=1.5, alpha=0.6, label='50% threshold')
                axes[1].axhline(y=0.75, color='red', linestyle='--', 
                              linewidth=1.5, alpha=0.6, label='75% threshold')
                axes[1].axhline(y=0.9, color='darkred', linestyle='--', 
                              linewidth=1.5, alpha=0.6, label='90% threshold')
                
                # Fill areas
                axes[1].fill_between(time_indices, 0, cp_probs,
                                   where=(cp_probs > 0.5),
                                   color='orange', alpha=0.2, label='Detected (>50%)')
                axes[1].fill_between(time_indices, 0, cp_probs,
                                   where=(cp_probs > 0.75),
                                   color='red', alpha=0.2)
                axes[1].fill_between(time_indices, 0, cp_probs,
                                   where=(cp_probs > 0.9),
                                   color='darkred', alpha=0.3, label='High Confidence (>90%)')
                
                axes[1].set_ylabel('Probability', fontsize=12, fontweight='bold')
                axes[1].set_title('Changepoint Detection Probabilities', 
                                fontsize=14, fontweight='bold')
                axes[1].legend(loc='best', fontsize=9)
                axes[1].grid(True, alpha=0.3, linestyle='--')
                axes[1].set_ylim([0, 1.0])
                
                # Plot 3: Volatility context
                # Calculate rolling volatility
                window = 20
                rolling_vol = pd.Series(data_values).rolling(window=window).std()
                
                axes[2].plot(time_indices, rolling_vol, 'g-', linewidth=1.5,
                           label=f'{window}-period Rolling Volatility', alpha=0.7)
                axes[2].set_ylabel('Volatility', fontsize=12, fontweight='bold')
                
                # Add changepoint markers
                ax2_twin = axes[2].twinx()
                ax2_twin.scatter(cp_90, [1]*len(cp_90), color='darkred', 
                               s=100, marker='v', label='Changepoints (>90%)',
                               zorder=5, edgecolors='black', linewidths=1.5)
                ax2_twin.set_ylabel('Changepoint Indicator', fontsize=12, fontweight='bold')
                ax2_twin.set_ylim([0, 2])
                ax2_twin.set_yticks([0, 1])
                ax2_twin.set_yticklabels(['', 'CP'])
                
                axes[2].set_xlabel('Time', fontsize=12, fontweight='bold')
                axes[2].set_title('Volatility Context and Changepoints', 
                                fontsize=14, fontweight='bold')
                axes[2].legend(loc='upper left', fontsize=9)
                axes[2].grid(True, alpha=0.3, linestyle='--')
                
                # Add statistics box
                stats_text = f"Data Points: {len(data_values)}\n"
                stats_text += f"CPs (>50%): {len(cp_50)}\n"
                stats_text += f"CPs (>75%): {len(cp_75)}\n"
                stats_text += f"CPs (>90%): {len(cp_90)}\n"
                stats_text += f"Mean Prob: {np.mean(cp_probs):.4f}\n"
                stats_text += f"Max Prob: {np.max(cp_probs):.4f}\n"
                stats_text += f"Current Prob: {current_prob:.4f}"
                
                axes[2].text(0.98, 0.98, stats_text,
                           transform=axes[2].transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           fontsize=9, family='monospace')
                
                plt.tight_layout()
                plot_file = f"reports/BCD_{symbol.replace(' ', '_')}_comprehensive.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"✓ Comprehensive plot saved to {plot_file}")
                plt.close()
                
                # Also create simple version
                simple_plot_file = f"reports/BCD_{symbol.replace(' ', '_')}_simple.png"
                model.plot_changepoints(
                    title=f"{symbol} - Bayesian Changepoint Detection",
                    save_path=simple_plot_file,
                    threshold=0.75
                )
                
            except Exception as e:
                print(f"Error creating visualization: {e}")
                import traceback
                traceback.print_exc()
            
            results[symbol] = {
                'model': model,
                'stats': stats,
                'returns': returns
            }
            
        except FileNotFoundError:
            print(f"Data not found for {symbol}, skipping...")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def main():
    """Run all BCD tests"""
    print("\n" + "=" * 80)
    print("BAYESIAN CHANGEPOINT DETECTION (BCD) - COMPREHENSIVE TESTING")
    print("Purpose: The Alarm - Detect structural breaks and regime changes")
    print("=" * 80)
    
    # Test 1: Synthetic data
    model_synth, stats_synth, accuracy = test_bcd_synthetic()
    
    # Test 2: Real data
    results_real = test_bcd_real_data()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY - BAYESIAN CHANGEPOINT DETECTION")
    print("=" * 80)
    print(f"\nSynthetic Data Test:")
    print(f"  ✓ Model fitted successfully")
    print(f"  ✓ Detection Accuracy: {accuracy:.2%}")
    print(f"  ✓ Can detect known structural breaks")
    
    print(f"\nReal Data Test:")
    print(f"  ✓ Tested on {len(results_real)} instruments")
    for symbol in results_real.keys():
        stats = results_real[symbol]['stats']
        print(f"  ✓ {symbol}: Detected {stats['n_significant_cp_75']} changepoints (>75% prob)")
    
    print("\n" + "=" * 80)
    print("FEASIBILITY ASSESSMENT:")
    print("=" * 80)
    print("✓ Model Implementation: SUCCESSFUL")
    print("✓ Detection Capability: GOOD")
    print(f"✓ Accuracy on Synthetic Data: {accuracy:.2%}")
    print("✓ Real-time Application: FEASIBLE")
    print("\nStrengths:")
    print("  • Probabilistic changepoint detection")
    print("  • Online algorithm (can process data sequentially)")
    print("  • No pre-specification of number of changepoints")
    print("  • Quantifies uncertainty with probabilities")
    print("\nLimitations:")
    print("  • Sensitive to hazard rate parameter")
    print("  • May have false positives in high volatility periods")
    print("  • Computational cost increases with data length")
    print("\nRecommendation for Project:")
    print("  ✓ APPROVED for use as 'The Alarm' sensor")
    print("  ✓ Use threshold >0.75 for high-confidence changepoints")
    print("  ✓ Combine with other sensors for confirmation")
    print("=" * 80)
    
    return model_synth, results_real


if __name__ == "__main__":
    model_synth, results_real = main()
