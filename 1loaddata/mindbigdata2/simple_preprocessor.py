#!/usr/bin/env python3
"""
Simple MindBigData Preprocessor
==============================

Simplified preprocessing for MindBigData without MVNN complexity.
Direct path from TSV to preprocessed EEG data ready for feature extraction.
"""

import numpy as np
import os
import pickle
from pathlib import Path
import logging
import argparse
from mindbigdata_loader import MindBigDataLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleMindBigDataPreprocessor:
    """Simple preprocessor for MindBigData"""
    
    def __init__(self):
        """Initialize simple preprocessor"""
        self.loader = MindBigDataLoader()
        logger.info("🔧 Simple MindBigData Preprocessor initialized")
    
    def preprocess_dataset(self, tsv_path: str, output_dir: str, 
                          max_trials: int = None, train_ratio: float = 0.8):
        """
        Simple preprocessing pipeline
        
        Args:
            tsv_path: Path to MindBigData TSV file
            output_dir: Output directory
            max_trials: Maximum trials to process
            train_ratio: Train/test split ratio
            
        Returns:
            Dictionary with preprocessed data
        """
        logger.info("🚀 Starting simple preprocessing pipeline...")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load MindBigData
        logger.info("📂 Loading MindBigData...")
        dataset = self.loader.load_dataset(tsv_path, max_trials=max_trials)
        
        if not dataset:
            raise ValueError("Failed to load dataset")
        
        eeg_data = dataset['eeg_data']  # (n_trials, 14, 256)
        labels = dataset['labels']      # (n_trials,)
        
        logger.info(f"✅ Loaded dataset:")
        logger.info(f"   EEG data: {eeg_data.shape}")
        logger.info(f"   Labels: {labels.shape}")
        logger.info(f"   Unique digits: {np.unique(labels)}")
        
        # Filter out random signals (-1)
        valid_mask = labels >= 0
        eeg_data = eeg_data[valid_mask]
        labels = labels[valid_mask]
        
        logger.info(f"📊 After filtering random signals:")
        logger.info(f"   EEG data: {eeg_data.shape}")
        logger.info(f"   Labels: {labels.shape}")
        logger.info(f"   Digit distribution: {np.bincount(labels)}")
        
        # Train/test split
        n_trials = len(eeg_data)
        n_train = int(train_ratio * n_trials)
        
        # Shuffle data
        indices = np.random.permutation(n_trials)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        train_data = eeg_data[train_indices]
        train_labels = labels[train_indices]
        test_data = eeg_data[test_indices]
        test_labels = labels[test_indices]
        
        logger.info(f"🔄 Train/test split:")
        logger.info(f"   Train: {train_data.shape}")
        logger.info(f"   Test: {test_data.shape}")
        
        # Basic preprocessing: normalize data
        logger.info("⚡ Applying basic preprocessing...")
        
        # Z-score normalization per channel
        train_data_norm = self.normalize_data(train_data)
        test_data_norm = self.normalize_data(test_data)
        
        # Prepare output data
        preprocessed_data = {
            'train_data': train_data_norm,
            'train_labels': train_labels,
            'test_data': test_data_norm,
            'test_labels': test_labels,
            'channels': dataset['channels'],
            'sampling_rate': dataset['sampling_rate'],
            'signal_duration': dataset.get('signal_duration', 2.0),
            'n_trials_train': len(train_data_norm),
            'n_trials_test': len(test_data_norm),
            'n_channels': len(dataset['channels']),
            'n_timepoints': train_data_norm.shape[2]
        }
        
        # Save preprocessed data
        logger.info("💾 Saving preprocessed data...")
        
        # Save as pickle
        output_file = os.path.join(output_dir, 'preprocessed_mindbigdata_simple.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(preprocessed_data, f)
        
        # Save as numpy arrays (for compatibility)
        np.save(os.path.join(output_dir, 'train_data.npy'), train_data_norm)
        np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
        np.save(os.path.join(output_dir, 'test_data.npy'), test_data_norm)
        np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)
        
        # Save metadata
        metadata = {
            'channels': dataset['channels'],
            'sampling_rate': dataset['sampling_rate'],
            'signal_duration': dataset.get('signal_duration', 2.0),
            'data_shape': {
                'train': train_data_norm.shape,
                'test': test_data_norm.shape
            },
            'digit_distribution': {
                'train': np.bincount(train_labels).tolist(),
                'test': np.bincount(test_labels).tolist()
            }
        }
        
        metadata_file = os.path.join(output_dir, 'metadata.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"✅ Preprocessing completed!")
        logger.info(f"   Output directory: {output_dir}")
        logger.info(f"   Files saved:")
        logger.info(f"     - preprocessed_mindbigdata_simple.pkl")
        logger.info(f"     - train_data.npy, train_labels.npy")
        logger.info(f"     - test_data.npy, test_labels.npy")
        logger.info(f"     - metadata.pkl")
        
        return preprocessed_data
    
    def normalize_data(self, data):
        """
        Normalize EEG data using z-score normalization per channel
        
        Args:
            data: EEG data (n_trials, n_channels, n_timepoints)
            
        Returns:
            Normalized data
        """
        logger.info("🔄 Normalizing data (z-score per channel)...")
        
        normalized_data = np.zeros_like(data)
        
        for trial_idx in range(data.shape[0]):
            for ch_idx in range(data.shape[1]):
                channel_data = data[trial_idx, ch_idx, :]
                
                # Z-score normalization
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                
                if std_val > 0:
                    normalized_data[trial_idx, ch_idx, :] = (channel_data - mean_val) / std_val
                else:
                    normalized_data[trial_idx, ch_idx, :] = channel_data - mean_val
        
        return normalized_data


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simple MindBigData Preprocessor')
    parser.add_argument('--tsv_path', type=str, required=True,
                       help='Path to MindBigData TSV file')
    parser.add_argument('--output_dir', type=str, default='./preprocessed_simple',
                       help='Output directory')
    parser.add_argument('--max_trials', type=int, default=None,
                       help='Maximum trials to process')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Train/test split ratio')
    
    args = parser.parse_args()
    
    print("🧠 SIMPLE MINDBIGDATA PREPROCESSOR")
    print("=" * 50)
    
    try:
        # Initialize preprocessor
        preprocessor = SimpleMindBigDataPreprocessor()
        
        # Run preprocessing
        results = preprocessor.preprocess_dataset(
            tsv_path=args.tsv_path,
            output_dir=args.output_dir,
            max_trials=args.max_trials,
            train_ratio=args.train_ratio
        )
        
        print(f"\n🎉 Preprocessing completed successfully!")
        print(f"   Train data: {results['train_data'].shape}")
        print(f"   Test data: {results['test_data'].shape}")
        print(f"   Channels: {results['n_channels']}")
        print(f"   Sampling rate: {results['sampling_rate']} Hz")
        print(f"   Output: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
