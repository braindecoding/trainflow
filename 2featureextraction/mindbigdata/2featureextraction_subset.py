#!/usr/bin/env python3
"""
Fast Feature Extraction for Stratified Subset
=============================================

Ultra-fast feature extraction for 1500-sample stratified subset
using UltraHighDimExtractor. Optimized for speed and memory efficiency.

Input:  1500 samples (150 per digit) in (n_trials, 14, 260) format
Output: 1500 samples with 35,000+ ultra-high dimensional features
Time:   ~15 minutes (vs 15+ hours for full dataset)
"""

import numpy as np
import pickle
import sys
import os
import time
import torch
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Fast Feature Extraction for Stratified Subset")
print(f"   Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Add UltraHighDimExtractor to path
sys.path.append('../UltraHighDimExtractor')

try:
    from core.ultra_extractor import UltraHighDimExtractor
    from utils.validation import validate_eeg_data
    print("✅ UltraHighDimExtractor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import UltraHighDimExtractor: {e}")
    sys.exit(1)


def load_stratified_subset():
    """Load stratified subset data"""
    print("📂 Loading stratified subset...")
    
    subset_path = 'mindbigdata_stratified_subset_1500.pkl'
    
    if not os.path.exists(subset_path):
        print(f"❌ Subset not found: {subset_path}")
        print("   Please run create_stratified_subset.py first")
        return None
    
    with open(subset_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Stratified subset loaded:")
    print(f"   Training: {data['training']['eeg'].shape}")
    print(f"   Validation: {data['validation']['eeg'].shape}")
    print(f"   Test: {data['test']['eeg'].shape}")
    
    # Verify balanced distribution
    for split_name, split_data in data.items():
        if split_name != 'metadata':
            labels = split_data['labels']
            unique, counts = np.unique(labels, return_counts=True)
            print(f"   {split_name.capitalize()} distribution: {dict(zip(unique, counts))}")
    
    return data


def create_fast_extractor():
    """Create UltraHighDimExtractor optimized for subset"""
    print("\n🔧 Creating fast UltraHighDimExtractor...")
    
    extractor = UltraHighDimExtractor(
        target_dimensions=35000,                    # Target 35K+ features
        wavelets=['db4', 'db8', 'coif5'],         # Multiple wavelets
        max_dwt_levels=6,                          # Deep decomposition
        max_wpd_levels=5,                          # Comprehensive analysis
        feature_types=['statistical', 'energy', 'entropy', 'morphological'],
        sampling_rate=128.0,                       # MindBigData sampling rate
        optimize_for='image_reconstruction',       # EEG-to-digit task
        n_jobs=1                                   # Single thread for stability
    )
    
    print(f"   Target dimensions: {extractor.target_dimensions:,}")
    print(f"   Wavelets: {extractor.wavelets}")
    print(f"   Optimized for subset processing")
    
    return extractor


def extract_features_fast(eeg_data, labels, split_name, extractor):
    """Fast feature extraction for subset"""
    print(f"\n🧠 Extracting features from {split_name} set...")
    print(f"   Input shape: {eeg_data.shape}")
    
    # Validate EEG data
    validated_data = validate_eeg_data(eeg_data)
    print(f"   Validated data shape: {validated_data.shape}")
    
    # Extract features (single batch for subset)
    start_time = time.time()
    features = extractor.fit_transform(validated_data)
    extraction_time = time.time() - start_time
    
    print(f"   ✅ Extracted {features.shape[1]:,} features in {extraction_time:.2f}s")
    print(f"   Output shape: {features.shape}")
    print(f"   Processing speed: {features.shape[0]/extraction_time:.1f} samples/second")
    
    # Quality check
    n_nan = np.isnan(features).sum()
    n_inf = np.isinf(features).sum()
    print(f"   Quality: {n_nan} NaN, {n_inf} Inf values")
    
    return features


def analyze_feature_quality(features, split_name):
    """Quick quality analysis for subset features"""
    print(f"\n📊 Analyzing {split_name} feature quality...")
    
    # Basic statistics
    mean_val = np.mean(features)
    std_val = np.std(features)
    min_val = np.min(features)
    max_val = np.max(features)
    
    print(f"   Mean: {mean_val:.6f}")
    print(f"   Std: {std_val:.6f}")
    print(f"   Range: [{min_val:.6f}, {max_val:.6f}]")
    
    # Feature diversity
    feature_stds = np.std(features, axis=0)
    active_features = np.sum(feature_stds > 1e-6)
    
    print(f"   Active features: {active_features:,}/{features.shape[1]:,}")
    print(f"   Feature diversity: {active_features/features.shape[1]*100:.1f}%")
    
    return {
        'mean': mean_val,
        'std': std_val,
        'range': [min_val, max_val],
        'active_features': active_features,
        'diversity': active_features/features.shape[1]
    }


def save_subset_features(features_data, metadata):
    """Save extracted features for subset"""
    print(f"\n💾 Saving subset features...")
    
    # Enhanced metadata
    enhanced_metadata = metadata.copy()
    enhanced_metadata.update({
        'feature_extraction': {
            'method': 'UltraHighDimExtractor',
            'n_features': features_data['training']['features'].shape[1],
            'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'subset_optimized': True,
            'processing_time': 'fast_subset_extraction'
        }
    })
    
    output_data = {
        'training': features_data['training'],
        'validation': features_data['validation'],
        'test': features_data['test'],
        'metadata': enhanced_metadata
    }
    
    # Save features
    output_path = 'mindbigdata_subset_features_1500.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"   ✅ Features saved to: {output_path}")
    
    # Summary
    total_samples = sum(data['features'].shape[0] for data in features_data.values())
    total_features = features_data['training']['features'].shape[1]
    
    print(f"\n📈 Subset Feature Extraction Summary:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Features per sample: {total_features:,}")
    print(f"   Memory usage: ~{(total_samples * total_features * 8) / (1024**3):.2f} GB")
    print(f"   Perfect for contrastive learning prototyping!")


def main():
    """Main fast feature extraction pipeline"""
    print("🎯 FAST FEATURE EXTRACTION FOR STRATIFIED SUBSET")
    print("=" * 80)
    print("📝 Processing 1500 balanced samples with UltraHighDimExtractor")
    print("🔧 Optimized for speed and memory efficiency")
    print("=" * 80)
    
    # Load stratified subset
    data = load_stratified_subset()
    if data is None:
        return
    
    # Create fast extractor
    extractor = create_fast_extractor()
    
    # Extract features from all splits
    features_data = {}
    quality_metrics = {}
    total_start_time = time.time()
    
    for split_name in ['training', 'validation', 'test']:
        eeg_data = data[split_name]['eeg']
        labels = data[split_name]['labels']
        images = data[split_name]['images']
        
        # Extract features
        features = extract_features_fast(eeg_data, labels, split_name, extractor)
        
        # Analyze quality
        quality = analyze_feature_quality(features, split_name)
        
        # Store results
        features_data[split_name] = {
            'features': features,
            'labels': labels,
            'images': images
        }
        quality_metrics[split_name] = quality
    
    total_time = time.time() - total_start_time
    
    # Update metadata with quality metrics
    enhanced_metadata = data['metadata'].copy()
    enhanced_metadata['feature_extraction']['quality_metrics'] = quality_metrics
    enhanced_metadata['feature_extraction']['total_processing_time'] = f"{total_time:.2f}s"
    
    # Save extracted features
    save_subset_features(features_data, enhanced_metadata)
    
    print(f"\n🚀 Fast feature extraction completed!")
    print(f"   Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"   Speedup: ~{(15*3600)/total_time:.0f}x faster than full dataset")
    print(f"   Ready for contrastive learning with CLIP embeddings!")
    print(f"\n🎯 Next Steps:")
    print(f"   1. Implement EEG-to-digit CLIP model")
    print(f"   2. Train contrastive learning")
    print(f"   3. Evaluate digit reconstruction")


if __name__ == "__main__":
    main()
