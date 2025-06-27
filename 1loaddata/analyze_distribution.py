#!/usr/bin/env python3
"""
Analyze Label Distribution
=========================

Analyze the distribution of labels in processed datasets to check balance.
"""

import numpy as np
import pickle
import os

def analyze_distribution(dataset_dir):
    """Analyze label distribution in a dataset"""
    
    print(f"🔍 ANALYZING DISTRIBUTION: {dataset_dir}")
    print("=" * 60)
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    # Load labels
    try:
        train_labels = np.load(os.path.join(dataset_dir, 'train_labels.npy'))
        test_labels = np.load(os.path.join(dataset_dir, 'test_labels.npy'))
    except Exception as e:
        print(f"❌ Error loading labels: {e}")
        return
    
    print("📊 CURRENT DATASET DISTRIBUTION:")
    print("-" * 40)
    
    # Train distribution
    print("Train labels distribution:")
    train_dist = np.bincount(train_labels)
    total_train = len(train_labels)
    
    for digit, count in enumerate(train_dist):
        percentage = count / total_train * 100
        print(f"   Digit {digit}: {count:4d} trials ({percentage:5.1f}%)")
    
    # Test distribution
    print("\nTest labels distribution:")
    test_dist = np.bincount(test_labels)
    total_test = len(test_labels)
    
    for digit, count in enumerate(test_dist):
        percentage = count / total_test * 100
        print(f"   Digit {digit}: {count:4d} trials ({percentage:5.1f}%)")
    
    # Balance analysis
    print("\n📈 BALANCE ANALYSIS:")
    print("-" * 40)
    
    train_min, train_max = train_dist.min(), train_dist.max()
    test_min, test_max = test_dist.min(), test_dist.max()
    
    train_balance = train_max / train_min if train_min > 0 else float('inf')
    test_balance = test_max / test_min if test_min > 0 else float('inf')
    
    print(f"Train balance ratio: {train_balance:.2f} (max/min)")
    print(f"Test balance ratio: {test_balance:.2f} (max/min)")
    
    # Balance assessment
    if train_balance < 1.2 and test_balance < 1.2:
        balance_status = "✅ EXCELLENT balance (< 1.2)"
    elif train_balance < 1.5 and test_balance < 1.5:
        balance_status = "✅ VERY GOOD balance (< 1.5)"
    elif train_balance < 2.0 and test_balance < 2.0:
        balance_status = "✅ GOOD balance (< 2.0)"
    elif train_balance < 3.0 and test_balance < 3.0:
        balance_status = "⚠️ ACCEPTABLE balance (< 3.0)"
    else:
        balance_status = "❌ POOR balance (> 3.0)"
    
    print(f"Overall assessment: {balance_status}")
    
    # Statistics
    print(f"\n🎯 DATASET STATISTICS:")
    print("-" * 40)
    print(f"Train total: {total_train:,} trials")
    print(f"Test total: {total_test:,} trials")
    print(f"Combined total: {total_train + total_test:,} trials")
    print(f"Train/test ratio: {total_train/total_test:.1f}:1")
    
    # Standard deviation analysis
    train_std = np.std(train_dist)
    test_std = np.std(test_dist)
    train_mean = np.mean(train_dist)
    test_mean = np.mean(test_dist)
    
    train_cv = train_std / train_mean * 100  # Coefficient of variation
    test_cv = test_std / test_mean * 100
    
    print(f"\n📊 VARIABILITY ANALYSIS:")
    print("-" * 40)
    print(f"Train CV: {train_cv:.1f}% (lower is better)")
    print(f"Test CV: {test_cv:.1f}% (lower is better)")
    
    if train_cv < 10 and test_cv < 10:
        variability_status = "✅ EXCELLENT variability (< 10%)"
    elif train_cv < 20 and test_cv < 20:
        variability_status = "✅ GOOD variability (< 20%)"
    else:
        variability_status = "⚠️ HIGH variability (> 20%)"
    
    print(f"Variability assessment: {variability_status}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    if train_balance > 2.0 or test_balance > 2.0:
        print("⚠️ Consider stratified sampling for better balance")
        print("⚠️ Current sampling may lead to biased performance")
    else:
        print("✅ Distribution is acceptable for training")
    
    if train_cv > 20 or test_cv > 20:
        print("⚠️ High variability detected - consider larger sample sizes")
    else:
        print("✅ Variability is within acceptable range")
    
    # Expected performance impact
    print(f"\n🎯 PERFORMANCE IMPACT:")
    print("-" * 40)
    
    if train_balance < 1.5 and test_balance < 1.5:
        print("✅ Balanced distribution should provide stable training")
        print("✅ All digits equally represented for fair evaluation")
    elif train_balance < 2.0 and test_balance < 2.0:
        print("✅ Reasonably balanced - minor impact on performance")
    else:
        print("⚠️ Imbalanced distribution may affect performance")
        print("⚠️ Some digits may be under/over-represented")
    
    return {
        'train_dist': train_dist,
        'test_dist': test_dist,
        'train_balance': train_balance,
        'test_balance': test_balance,
        'train_cv': train_cv,
        'test_cv': test_cv,
        'balance_status': balance_status,
        'variability_status': variability_status
    }


def compare_distributions():
    """Compare distributions across different datasets"""
    
    print("\n🔄 COMPARING DATASET DISTRIBUTIONS:")
    print("=" * 60)
    
    datasets = [
        ("1K Dataset", "preprocessed_simple_real"),
        ("5K Dataset", "preprocessed_5k_dev")
    ]
    
    results = {}
    
    for name, path in datasets:
        if os.path.exists(path):
            print(f"\n📊 {name}:")
            results[name] = analyze_distribution(path)
        else:
            print(f"\n❌ {name}: Not found ({path})")
    
    # Summary comparison
    if len(results) > 1:
        print(f"\n📋 COMPARISON SUMMARY:")
        print("=" * 60)
        
        for name, result in results.items():
            print(f"{name}:")
            print(f"   Balance: {result['train_balance']:.2f} (train), {result['test_balance']:.2f} (test)")
            print(f"   Variability: {result['train_cv']:.1f}% (train), {result['test_cv']:.1f}% (test)")
            print(f"   Status: {result['balance_status']}")


if __name__ == "__main__":
    # Analyze 5K dataset
    analyze_distribution("preprocessed_5k_dev")
    
    # Compare all datasets
    compare_distributions()
