#!/usr/bin/env python3
"""
Check Full Dataset Size
======================

Check the actual size of the full MindBigData dataset to see how much data we have available.
"""

import sys
import os
sys.path.append('mindbigdata2')
from mindbigdata_loader import MindBigDataLoader
import numpy as np

def check_full_dataset_size():
    """Check the size of the full MindBigData dataset"""
    
    print("🔍 CHECKING FULL DATASET SIZE...")
    print("=" * 60)
    
    # Initialize loader
    loader = MindBigDataLoader()
    
    # Load full dataset (no limit)
    print("📂 Loading FULL MindBigData dataset...")
    print("   This may take several minutes...")
    
    dataset_path = r"d:\trainflow\dataset\datasets\EP1.01.txt"
    
    try:
        dataset = loader.load_dataset(dataset_path, max_trials=None)
        
        if dataset:
            print(f"\n✅ FULL DATASET LOADED:")
            print(f"   EEG data shape: {dataset['eeg_data'].shape}")
            print(f"   Labels shape: {dataset['labels'].shape}")
            print(f"   Total trials: {len(dataset['labels'])}")
            
            # Check digit distribution
            labels = dataset['labels']
            valid_labels = labels[labels >= 0]  # Remove random signals (-1)
            random_signals = labels[labels == -1]
            
            print(f"\n📊 DATA BREAKDOWN:")
            print(f"   Valid trials (digits 0-9): {len(valid_labels)}")
            print(f"   Random signals (-1): {len(random_signals)}")
            print(f"   Total trials: {len(labels)}")
            
            if len(valid_labels) > 0:
                digit_dist = np.bincount(valid_labels)
                print(f"\n📈 DIGIT DISTRIBUTION:")
                for digit, count in enumerate(digit_dist):
                    print(f"   Digit {digit}: {count} trials")
                
                print(f"\n🎯 POTENTIAL DATASET SIZES:")
                # Calculate different train/test splits
                n_valid = len(valid_labels)
                
                splits = [
                    (0.8, 0.2, "Standard"),
                    (0.9, 0.1, "Large train"),
                    (0.7, 0.3, "Large test")
                ]
                
                for train_ratio, test_ratio, desc in splits:
                    n_train = int(train_ratio * n_valid)
                    n_test = n_valid - n_train
                    print(f"   {desc} ({train_ratio:.0%}/{test_ratio:.0%}): {n_train} train, {n_test} test")
                
                print(f"\n🔍 COMPARISON WITH CURRENT:")
                print(f"   Current processed: 1,000 trials (sample)")
                print(f"   Available total: {n_valid:,} trials")
                print(f"   Utilization: {1000/n_valid*100:.1f}% of available data")
                
                if n_valid > 10000:
                    print(f"\n💡 RECOMMENDATIONS:")
                    print(f"   - Process larger dataset for better performance")
                    print(f"   - Consider 5,000-10,000 trials for development")
                    print(f"   - Use full dataset for final training")
                    
                    # Suggest practical sizes
                    practical_sizes = [2000, 5000, 10000, 20000]
                    print(f"\n📋 PRACTICAL DATASET SIZES:")
                    for size in practical_sizes:
                        if size <= n_valid:
                            train_size = int(0.8 * size)
                            test_size = size - train_size
                            print(f"   {size:,} trials: {train_size:,} train, {test_size:,} test")
            
            print(f"\n🚀 SCALING POTENTIAL:")
            print(f"   Current: 800 train + 200 test = 1,000 total")
            print(f"   Available: {n_valid:,} total trials")
            print(f"   Scale factor: {n_valid/1000:.1f}x larger dataset available")
            
        else:
            print("❌ Failed to load dataset")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_full_dataset_size()
