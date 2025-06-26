#!/usr/bin/env python3
"""
MindBigData Feature Extraction using UltraHighDimExtractor
=========================================================

Extract ultra-high dimensional features from preprocessed MindBigData EEG signals
using the UltraHighDimExtractor for EEG-to-digit reconstruction tasks.

Input:  (n_trials, 14, 128) - Preprocessed EEG data
Output: (n_trials, 35000+) - Ultra-high dimensional features
"""

import numpy as np
import pickle
import sys
import os
import time
from pathlib import Path
import logging
import torch
import torch.nn.functional as F

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add UltraHighDimExtractor to path
sys.path.append('../UltraHighDimExtractor')

try:
    from core.ultra_extractor import UltraHighDimExtractor
    from utils.validation import validate_eeg_data
    from utils.metrics import FeatureQualityMetrics
    print("✅ UltraHighDimExtractor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import UltraHighDimExtractor: {e}")
    print("Please ensure UltraHighDimExtractor is in the correct path")
    sys.exit(1)


def load_preprocessed_data():
    """Load preprocessed MindBigData from 1loaddata step"""
    print("📂 Loading preprocessed MindBigData...")
    
    # Path to preprocessed data
    data_path = '../../1loaddata/mindbigdata/mindbigdata_processed_data_correct.pkl'
    
    if not os.path.exists(data_path):
        print(f"❌ Preprocessed data not found at: {data_path}")
        print("Please run 1loaddata/mindbigdata/1process_mindbigdata_data.py first")
        return None
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Data loaded successfully")
        print(f"   Training: {data['training']['eeg'].shape}")
        print(f"   Validation: {data['validation']['eeg'].shape}")
        print(f"   Test: {data['test']['eeg'].shape}")
        print(f"   Format: {data['metadata'].get('format', 'Not specified')}")
        print(f"   Channels: {data['metadata'].get('n_channels', 'Unknown')}")
        print(f"   Timepoints: {data['metadata'].get('n_timepoints', 'Unknown')}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def extract_features_from_split_gpu(eeg_data, labels, split_name, extractor, batch_size=1000):
    """Extract features from a data split with GPU acceleration"""
    print(f"\n🧠 Extracting features from {split_name} set (GPU-accelerated)...")
    print(f"   Input shape: {eeg_data.shape}")
    print(f"   Batch size: {batch_size}")

    # Validate EEG data format
    validated_data = validate_eeg_data(eeg_data)
    print(f"   Validated data shape: {validated_data.shape}")

    n_samples = validated_data.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"   Processing {n_samples} samples in {n_batches} batches...")

    # Extract features in batches
    start_time = time.time()
    all_features = []

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)

        batch_data = validated_data[start_idx:end_idx]

        # Move to GPU if available
        if torch.cuda.is_available():
            try:
                # Convert to tensor and move to GPU
                batch_tensor = torch.tensor(batch_data, device=device, dtype=torch.float32)

                # Process on GPU (if extractor supports it)
                with torch.amp.autocast('cuda'):  # Mixed precision for speed
                    batch_features = extractor.fit_transform(batch_tensor.cpu().numpy())

                # Clear GPU cache
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"   ⚠️ GPU processing failed, falling back to CPU: {e}")
                batch_features = extractor.fit_transform(batch_data)
        else:
            batch_features = extractor.fit_transform(batch_data)

        all_features.append(batch_features)

        # Progress update
        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            progress = (batch_idx + 1) / n_batches * 100
            print(f"   Progress: {progress:.1f}% ({batch_idx + 1}/{n_batches} batches)")

    # Concatenate all features
    features = np.vstack(all_features)
    extraction_time = time.time() - start_time

    print(f"   ✅ Extracted {features.shape[1]:,} features in {extraction_time:.2f}s")
    print(f"   Output shape: {features.shape}")
    print(f"   Processing speed: {features.shape[1]/extraction_time:.0f} features/second")
    print(f"   Samples/second: {n_samples/extraction_time:.0f}")

    # Quality check
    n_nan = np.isnan(features).sum()
    n_inf = np.isinf(features).sum()
    print(f"   Quality: {n_nan} NaN, {n_inf} Inf values")

    return features


def extract_features_from_split(eeg_data, labels, split_name, extractor):
    """Extract features from a data split (CPU fallback)"""
    print(f"\n🧠 Extracting features from {split_name} set...")
    print(f"   Input shape: {eeg_data.shape}")

    # Validate EEG data format
    validated_data = validate_eeg_data(eeg_data)
    print(f"   Validated data shape: {validated_data.shape}")

    # Extract ultra-high dimensional features
    start_time = time.time()
    features = extractor.fit_transform(validated_data)
    extraction_time = time.time() - start_time

    print(f"   ✅ Extracted {features.shape[1]:,} features in {extraction_time:.2f}s")
    print(f"   Output shape: {features.shape}")
    print(f"   Processing speed: {features.shape[1]/extraction_time:.0f} features/second")

    # Quality check
    n_nan = np.isnan(features).sum()
    n_inf = np.isinf(features).sum()
    print(f"   Quality: {n_nan} NaN, {n_inf} Inf values")

    return features


def create_feature_extractor(data_info):
    """Create and configure UltraHighDimExtractor"""
    print("\n🔧 Creating UltraHighDimExtractor...")

    # Get actual data dimensions
    n_channels = data_info.get('n_channels', 14)
    n_timepoints = data_info.get('n_timepoints', 260)
    sampling_rate = data_info.get('sampling_rate', 128)

    print(f"   Detected format: ({n_channels} channels, {n_timepoints} timepoints)")

    # Configuration optimized for actual MindBigData format
    extractor = UltraHighDimExtractor(
        target_dimensions=35000,                    # Target 35K+ features
        wavelets=['db4', 'db8', 'coif5'],         # Multiple wavelets for diversity
        max_dwt_levels=6,                          # Deep DWT decomposition
        max_wpd_levels=5,                          # Deep WPD decomposition
        feature_types=['statistical', 'energy', 'entropy', 'morphological'],
        sampling_rate=float(sampling_rate),        # Actual sampling rate
        optimize_for='image_reconstruction',       # Optimized for digit reconstruction
        n_jobs=1                                   # Single thread for stability
    )
    
    print(f"   Target dimensions: {extractor.target_dimensions:,}")
    print(f"   Wavelets: {extractor.wavelets}")
    print(f"   Max DWT levels: {extractor.max_dwt_levels}")
    print(f"   Max WPD levels: {extractor.max_wpd_levels}")
    print(f"   Sampling rate: {extractor.sampling_rate} Hz")
    
    return extractor


def analyze_feature_quality(features, split_name):
    """Analyze quality of extracted features"""
    print(f"\n📊 Analyzing {split_name} feature quality...")
    
    try:
        metrics = FeatureQualityMetrics()
        quality_scores = metrics.compute_comprehensive_metrics(features)
        
        print(f"   SNR estimate: {quality_scores.get('snr', 'N/A'):.3f}")
        print(f"   Stability: {quality_scores.get('stability', 'N/A'):.3f}")
        print(f"   Information content: {quality_scores.get('information_content', 'N/A'):.3f}")
        print(f"   Redundancy: {quality_scores.get('redundancy', 'N/A'):.3f}")
        
        return quality_scores
        
    except Exception as e:
        print(f"   ⚠️ Quality analysis failed: {e}")
        return {}


def save_extracted_features(features_data, metadata):
    """Save extracted features to file"""
    print(f"\n💾 Saving extracted features...")
    
    # Create output data structure
    output_data = {
        'training': features_data['training'],
        'validation': features_data['validation'],
        'test': features_data['test'],
        'metadata': metadata,
        'extraction_info': {
            'extractor': 'UltraHighDimExtractor',
            'version': '1.0.0',
            'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_format': '(n_trials, 14, 128)',
            'output_format': f"(n_trials, {features_data['training']['features'].shape[1]})"
        }
    }
    
    # Save to file
    output_path = 'mindbigdata_ultrahighdim_features.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"   ✅ Features saved to: {output_path}")
    
    # Print summary
    total_trials = sum(data['features'].shape[0] for data in features_data.values())
    total_features = features_data['training']['features'].shape[1]
    
    print(f"\n📈 Feature Extraction Summary:")
    print(f"   Total trials: {total_trials:,}")
    print(f"   Features per trial: {total_features:,}")
    print(f"   Total feature matrix size: {total_trials:,} × {total_features:,}")
    print(f"   Memory usage: ~{(total_trials * total_features * 8) / (1024**3):.2f} GB")


def main():
    """Main feature extraction pipeline"""
    print("🎯 MINDBIGDATA ULTRA-HIGH DIMENSIONAL FEATURE EXTRACTION")
    print("=" * 80)
    print("📝 Extracting 35,000+ features from preprocessed EEG data")
    print("🔧 Using UltraHighDimExtractor for EEG-to-digit reconstruction")
    print("=" * 80)
    
    # Load preprocessed data
    data = load_preprocessed_data()
    if data is None:
        return
    
    # Create feature extractor
    extractor = create_feature_extractor(data['metadata'])
    
    # Extract features from all splits
    features_data = {}
    quality_metrics = {}

    # Determine batch size based on GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Adaptive batch size based on GPU memory
        if gpu_memory_gb >= 16:
            batch_size = 2000
        elif gpu_memory_gb >= 8:
            batch_size = 1000
        else:
            batch_size = 500
        print(f"🔧 GPU detected ({gpu_memory_gb:.1f}GB), using batch size: {batch_size}")
        use_gpu = True
    else:
        batch_size = 100  # Small batch for CPU
        print(f"🔧 No GPU detected, using CPU with batch size: {batch_size}")
        use_gpu = False

    for split_name in ['training', 'validation', 'test']:
        eeg_data = data[split_name]['eeg']
        labels = data[split_name]['labels']
        images = data[split_name]['images']

        # Extract features (GPU-accelerated if available)
        if use_gpu and len(eeg_data) > 1000:  # Use GPU for large datasets
            features = extract_features_from_split_gpu(eeg_data, labels, split_name, extractor, batch_size)
        else:
            features = extract_features_from_split(eeg_data, labels, split_name, extractor)

        # Analyze quality
        quality = analyze_feature_quality(features, split_name)

        # Store results
        features_data[split_name] = {
            'features': features,
            'labels': labels,
            'images': images
        }
        quality_metrics[split_name] = quality
    
    # Update metadata
    enhanced_metadata = data['metadata'].copy()
    enhanced_metadata.update({
        'feature_extraction': {
            'method': 'UltraHighDimExtractor',
            'n_features': features_data['training']['features'].shape[1],
            'extraction_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'quality_metrics': quality_metrics
        }
    })
    
    # Save extracted features
    save_extracted_features(features_data, enhanced_metadata)
    
    print(f"\n🚀 Feature extraction completed successfully!")
    print(f"   Ready for EEG-to-digit reconstruction modeling!")


if __name__ == "__main__":
    main()
