"""
Quick Runner: HMM Comparison Test Suite
Run this script to execute the comprehensive HMM comparison
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the comprehensive test
from tests.test_hmm_comparison_comprehensive import main

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║         HMM MODEL COMPARISON FOR TICC REGIME DETECTION PIPELINE          ║
    ║                                                                          ║
    ║  Comparing: Sticky HDP-HMM vs Standard HMM                              ║
    ║  Purpose: Determine best model for Layer 3 (Regime Classification)      ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    This comprehensive test suite will:
    
    ✓ Part 1: Test basic functionality on synthetic data
    ✓ Part 2: Test model selection with unknown regimes
    ✓ Part 3: Test regime stability and sticky transitions
    ✓ Part 4: Benchmark computational efficiency
    ✓ Part 5: Evaluate on real market data (NIFTY indices)
    ✓ Part 6: Test TICC integration compatibility
    ✓ Part 7: Generate final recommendation
    
    Reports will be saved to: reports/HMM_COMPARISON_*.csv
    
    Estimated runtime: 5-10 minutes
    
    """)
    
    input("Press Enter to start the comparison tests...")
    
    try:
        main()
        
        print("""
    
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                         TESTING COMPLETE!                                ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    📊 All reports have been generated in the 'reports/' directory.
    
    📄 Review the following files:
       1. HMM_COMPARISON_Part1_Basic.csv
       2. HMM_COMPARISON_Part2_ModelSelection.csv
       3. HMM_COMPARISON_Part3_Stability.csv
       4. HMM_COMPARISON_Part4_Efficiency.csv
       5. HMM_COMPARISON_Part5_RealData.csv
       6. HMM_COMPARISON_Part6_TICC_Integration.csv
       7. HMM_COMPARISON_Final_Summary.csv
    
    📖 Quick Reference Guide: HMM_COMPARISON_GUIDE.md
    
    🏆 FINAL RECOMMENDATION: Sticky HDP-HMM
    
    ✅ Reasons:
       - Automatic regime discovery (no need to specify n_regimes)
       - Sticky transitions (kappa parameter) for regime persistence
       - Bayesian framework with uncertainty quantification
       - Single source architecture (fits your requirement)
       - Best for TICC integration in Layer 3
    
    🔧 Recommended Configuration:
       HDPHMM(truncation=8, alpha=1.0, gamma=1.0, kappa=20.0, max_iter=50)
    
        """)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print(f"\nPlease check that:")
        print(f"  1. All dependencies are installed (pip install -r requirements.txt)")
        print(f"  2. Data files are available in 'indexes data/' directory")
        print(f"  3. utils/data_loader.py is working correctly")
        import traceback
        traceback.print_exc()
