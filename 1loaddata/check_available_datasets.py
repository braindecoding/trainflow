#!/usr/bin/env python3
"""
Check Available Datasets
========================

Check which processed datasets are available and their specifications.
"""

import numpy as np
import os

def check_available_datasets():
    """Check all available processed datasets"""
    
    print("📊 AVAILABLE PROCESSED DATASETS:")
    print("=" * 60)
    
    datasets = [
        ('1K Dataset', 'preprocessed_simple_real'),
        ('5K Dataset', 'preprocessed_5k_dev'), 
        ('65K Dataset', 'preprocessed_full_production')
    ]
    
    available_datasets = []
    
    for name, path in datasets:
        if os.path.exists(path):
            try:
                train_data = np.load(f'{path}/train_data.npy')
                test_data = np.load(f'{path}/test_data.npy')
                train_labels = np.load(f'{path}/train_labels.npy')
                test_labels = np.load(f'{path}/test_labels.npy')
                
                total_trials = len(train_labels) + len(test_labels)
                train_size_mb = os.path.getsize(f"{path}/train_data.npy") / (1024**2)
                test_size_mb = os.path.getsize(f"{path}/test_data.npy") / (1024**2)
                
                print(f'\n✅ {name}:')
                print(f'   📁 Directory: {path}')
                print(f'   🎯 Train: {train_data.shape}')
                print(f'   🎯 Test: {test_data.shape}')
                print(f'   📊 Total trials: {total_trials:,}')
                print(f'   💾 Train file: {train_size_mb:.1f} MB')
                print(f'   💾 Test file: {test_size_mb:.1f} MB')
                print(f'   💾 Total size: {train_size_mb + test_size_mb:.1f} MB')
                
                # Check balance
                train_dist = np.bincount(train_labels)
                test_dist = np.bincount(test_labels)
                train_balance = train_dist.max() / train_dist.min() if train_dist.min() > 0 else float('inf')
                test_balance = test_dist.max() / test_dist.min() if test_dist.min() > 0 else float('inf')
                
                print(f'   ⚖️ Balance: Train {train_balance:.2f}, Test {test_balance:.2f}')
                
                available_datasets.append({
                    'name': name,
                    'path': path,
                    'train_shape': train_data.shape,
                    'test_shape': test_data.shape,
                    'total_trials': total_trials,
                    'size_mb': train_size_mb + test_size_mb,
                    'train_balance': train_balance,
                    'test_balance': test_balance
                })
                
            except Exception as e:
                print(f'\n❌ {name}: Error loading - {e}')
        else:
            print(f'\n❌ {name}: Directory not found')
    
    if available_datasets:
        print(f'\n🎯 DATASET COMPARISON:')
        print("=" * 60)
        
        # Sort by total trials
        available_datasets.sort(key=lambda x: x['total_trials'])
        
        for dataset in available_datasets:
            scale_factor = dataset['total_trials'] / available_datasets[0]['total_trials']
            print(f"{dataset['name']}:")
            print(f"   Trials: {dataset['total_trials']:,} ({scale_factor:.1f}x)")
            print(f"   Size: {dataset['size_mb']:.1f} MB")
            print(f"   Balance: {min(dataset['train_balance'], dataset['test_balance']):.2f}")
        
        # Recommendations
        print(f'\n💡 RECOMMENDATIONS:')
        print("=" * 60)
        
        largest_dataset = max(available_datasets, key=lambda x: x['total_trials'])
        best_balance = min(available_datasets, key=lambda x: max(x['train_balance'], x['test_balance']))
        
        print(f"🚀 For MAXIMUM PERFORMANCE:")
        print(f"   Use: {largest_dataset['name']} ({largest_dataset['path']})")
        print(f"   Trials: {largest_dataset['total_trials']:,}")
        print(f"   Expected accuracy: 90-95%")
        
        print(f"\n⚡ For DEVELOPMENT/TESTING:")
        print(f"   Use: 5K Dataset (preprocessed_5k_dev)")
        print(f"   Trials: 5,000")
        print(f"   Expected accuracy: 85%+")
        print(f"   Faster iteration")
        
        print(f"\n🎯 CURRENT RECOMMENDATION:")
        print(f"   Latest and best: {largest_dataset['name']}")
        print(f"   Directory: {largest_dataset['path']}")
        print(f"   Ready for production use")
        
        return largest_dataset['path']
    
    else:
        print("\n❌ No datasets found!")
        return None


if __name__ == "__main__":
    recommended_path = check_available_datasets()
    
    if recommended_path:
        print(f"\n🎉 READY TO USE:")
        print(f"   Recommended dataset: {recommended_path}")
        print(f"   Files ready for feature extraction pipeline")
