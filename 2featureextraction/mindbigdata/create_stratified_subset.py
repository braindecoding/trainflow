#!/usr/bin/env python3
"""
Create Stratified Subset from MindBigData
========================================

Create a balanced 1500-sample subset with stratified sampling
ensuring equal representation from all digits (0-9) and
proper train/validation/test splits.

Target:
- 1500 total samples (150 per digit)
- 70% train (1050 samples, 105 per digit)
- 15% validation (225 samples, 22-23 per digit)
- 15% test (225 samples, 22-23 per digit)
"""

import numpy as np
import pickle
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_original_data():
    """Load original preprocessed MindBigData"""
    print("📂 Loading original MindBigData...")
    
    data_path = '../../1loaddata/mindbigdata/mindbigdata_processed_data_correct.pkl'
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Original data loaded:")
    print(f"   Training: {data['training']['eeg'].shape}")
    print(f"   Validation: {data['validation']['eeg'].shape}")
    print(f"   Test: {data['test']['eeg'].shape}")
    
    return data

def analyze_label_distribution(data):
    """Analyze label distribution in original data"""
    print("\n📊 Analyzing original label distribution...")
    
    all_labels = []
    all_labels.extend(data['training']['labels'])
    all_labels.extend(data['validation']['labels'])
    all_labels.extend(data['test']['labels'])
    
    label_counts = Counter(all_labels)
    total_samples = len(all_labels)
    
    print(f"   Total samples: {total_samples:,}")
    print(f"   Label distribution:")
    for digit in sorted(label_counts.keys()):
        count = label_counts[digit]
        percentage = count / total_samples * 100
        print(f"     Digit {digit}: {count:,} samples ({percentage:.1f}%)")
    
    return label_counts, total_samples

def create_stratified_subset(data, target_samples=1500, samples_per_digit=150):
    """Create stratified subset with equal samples per digit"""
    print(f"\n🎯 Creating stratified subset...")
    print(f"   Target: {target_samples} samples ({samples_per_digit} per digit)")
    
    # Combine all data
    all_eeg = np.vstack([
        data['training']['eeg'],
        data['validation']['eeg'],
        data['test']['eeg']
    ])
    
    all_labels = np.concatenate([
        data['training']['labels'],
        data['validation']['labels'],
        data['test']['labels']
    ])
    
    all_images = np.vstack([
        data['training']['images'],
        data['validation']['images'],
        data['test']['images']
    ])
    
    print(f"   Combined data shape: {all_eeg.shape}")
    print(f"   Combined labels shape: {all_labels.shape}")
    
    # Stratified sampling per digit
    subset_indices = []
    subset_counts = {}
    
    for digit in range(10):  # Digits 0-9
        digit_indices = np.where(all_labels == digit)[0]
        available_samples = len(digit_indices)
        
        if available_samples < samples_per_digit:
            print(f"   ⚠️ Digit {digit}: Only {available_samples} available (need {samples_per_digit})")
            selected_indices = digit_indices  # Use all available
        else:
            # Randomly sample required number
            selected_indices = np.random.choice(
                digit_indices, 
                size=samples_per_digit, 
                replace=False
            )
        
        subset_indices.extend(selected_indices)
        subset_counts[digit] = len(selected_indices)
        print(f"   Digit {digit}: Selected {len(selected_indices)} samples")
    
    # Extract subset
    subset_eeg = all_eeg[subset_indices]
    subset_labels = all_labels[subset_indices]
    subset_images = all_images[subset_indices]
    
    print(f"\n✅ Subset created:")
    print(f"   EEG shape: {subset_eeg.shape}")
    print(f"   Labels shape: {subset_labels.shape}")
    print(f"   Images shape: {subset_images.shape}")
    
    return subset_eeg, subset_labels, subset_images, subset_counts

def create_train_val_test_splits(eeg_data, labels, images, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create stratified train/validation/test splits"""
    print(f"\n📊 Creating train/validation/test splits...")
    print(f"   Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%")
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp, img_train, img_temp = train_test_split(
        eeg_data, labels, images,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=42
    )
    
    # Second split: val vs test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test, img_val, img_test = train_test_split(
        X_temp, y_temp, img_temp,
        test_size=(1 - val_test_ratio),
        stratify=y_temp,
        random_state=42
    )
    
    # Verify splits
    print(f"\n✅ Splits created:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    # Check label distribution
    for split_name, split_labels in [('Training', y_train), ('Validation', y_val), ('Test', y_test)]:
        label_counts = Counter(split_labels)
        print(f"   {split_name} distribution:")
        for digit in sorted(label_counts.keys()):
            print(f"     Digit {digit}: {label_counts[digit]} samples")
    
    return {
        'training': {'eeg': X_train, 'labels': y_train, 'images': img_train},
        'validation': {'eeg': X_val, 'labels': y_val, 'images': img_val},
        'test': {'eeg': X_test, 'labels': y_test, 'images': img_test}
    }

def save_stratified_subset(subset_data, metadata):
    """Save stratified subset to file"""
    print(f"\n💾 Saving stratified subset...")
    
    # Enhanced metadata
    enhanced_metadata = metadata.copy()
    enhanced_metadata.update({
        'subset_info': {
            'total_samples': sum(len(split['eeg']) for split in subset_data.values()),
            'samples_per_split': {
                split: len(data['eeg']) for split, data in subset_data.items()
            },
            'creation_method': 'stratified_sampling',
            'target_samples_per_digit': 150,
            'random_seed': 42
        }
    })
    
    output_data = {
        'training': subset_data['training'],
        'validation': subset_data['validation'],
        'test': subset_data['test'],
        'metadata': enhanced_metadata
    }
    
    output_path = 'mindbigdata_stratified_subset_1500.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"   ✅ Saved to: {output_path}")
    
    # Summary
    total_samples = sum(len(split['eeg']) for split in subset_data.values())
    print(f"\n📈 Stratified Subset Summary:")
    print(f"   Total samples: {total_samples}")
    print(f"   Memory footprint: ~{total_samples * 39368 * 8 / (1024**3):.2f} GB (after feature extraction)")
    print(f"   Processing time estimate: ~15 minutes (vs 15+ hours)")
    print(f"   Perfect for prototyping and testing!")

def visualize_distribution(subset_data):
    """Visualize label distribution across splits"""
    print(f"\n📊 Creating distribution visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (split_name, split_data) in enumerate(subset_data.items()):
        labels = split_data['labels']
        label_counts = Counter(labels)
        
        digits = sorted(label_counts.keys())
        counts = [label_counts[d] for d in digits]
        
        axes[idx].bar(digits, counts, alpha=0.7)
        axes[idx].set_title(f'{split_name.capitalize()} Set\n({len(labels)} samples)')
        axes[idx].set_xlabel('Digit')
        axes[idx].set_ylabel('Count')
        axes[idx].set_xticks(digits)
    
    plt.tight_layout()
    plt.savefig('stratified_subset_distribution.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved visualization: stratified_subset_distribution.png")

def main():
    """Main stratified subset creation pipeline"""
    print("🎯 MINDBIGDATA STRATIFIED SUBSET CREATION")
    print("=" * 80)
    print("📝 Creating balanced 1500-sample subset for efficient processing")
    print("🔧 Stratified sampling with equal digit representation")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load original data
    original_data = load_original_data()
    
    # Analyze distribution
    label_counts, total_samples = analyze_label_distribution(original_data)
    
    # Create stratified subset
    subset_eeg, subset_labels, subset_images, subset_counts = create_stratified_subset(
        original_data, target_samples=1500, samples_per_digit=150
    )
    
    # Create train/val/test splits
    subset_splits = create_train_val_test_splits(
        subset_eeg, subset_labels, subset_images
    )
    
    # Save subset
    save_stratified_subset(subset_splits, original_data['metadata'])
    
    # Visualize distribution
    visualize_distribution(subset_splits)
    
    print(f"\n🚀 Stratified subset creation completed!")
    print(f"   Ready for ultra-fast feature extraction!")
    print(f"   Next: Run feature extraction on subset (~15 minutes)")

if __name__ == "__main__":
    main()
