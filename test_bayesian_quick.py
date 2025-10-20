#!/usr/bin/env python3
"""
Quick test of the Bayesian Hyperparameter Optimization with 1 trial
"""

from auto_hyperparameter_bayesian import BayesianHyperparameterSearch

def main():
    print("="*80)
    print("QUICK BAYESIAN OPTIMIZATION TEST (1 TRIAL)")
    print("="*80)
    
    # Run with just 1 trial for testing
    search = BayesianHyperparameterSearch(n_calls=1, random_state=42)
    
    print("\n2. Running 1 optimization trial...")
    result = search.optimize()
    
    print("\n3. Results:")
    search.print_summary()
    
    print("\n✅ Quick test completed successfully!")
    print("To run full optimization (30 trials): python auto_hyperparameter_bayesian.py")

if __name__ == "__main__":
    main()