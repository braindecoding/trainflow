#!/usr/bin/env python3
"""
MindBigData to Things-EEG2 Preprocessing Adapter
===============================================

Adapter to make MindBigData compatible with Things-EEG2 preprocessing pipeline.
Converts MindBigData format to match expected input/output of preprocessing.py

Key adaptations:
- Convert TSV format to MNE-compatible structure
- Map 14 EPOC channels to processing pipeline
- Adapt single trials to session-like structure
- Convert digit codes to condition-like organization
"""

import numpy as np
import mne
import os
import pickle
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import argparse

# Import our MindBigData loader
import sys
sys.path.append('..')
from mindbigdata_loader import MindBigDataLoader

# Import Things-EEG2 preprocessing functions
from preprocessing_utils import epoching, mvnn, save_prepr

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MindBigDataAdapter:
    """Adapter for MindBigData to Things-EEG2 preprocessing"""
    
    def __init__(self):
        """Initialize adapter"""
        self.loader = MindBigDataLoader()
        
        # EPOC channels (14)
        self.epoc_channels = [
            'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
            'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
        ]
        
        # Digit conditions (0-9)
        self.digit_conditions = list(range(10))
        
        logger.info("🔄 MindBigData to Things-EEG2 Adapter initialized")
    
    def load_mindbigdata(self, tsv_path: str, max_trials: Optional[int] = None) -> Dict:
        """Load MindBigData from TSV file"""
        logger.info(f"📂 Loading MindBigData from: {tsv_path}")
        
        dataset = self.loader.load_dataset(tsv_path, max_trials=max_trials)
        
        if not dataset:
            raise ValueError("Failed to load MindBigData dataset")
        
        logger.info(f"✅ Loaded MindBigData:")
        logger.info(f"   Shape: {dataset['eeg_data'].shape}")
        logger.info(f"   Labels: {len(np.unique(dataset['labels']))} unique digits")
        
        return dataset
    
    def create_mne_structure(self, dataset: Dict) -> Tuple[Dict, Dict]:
        """
        Convert MindBigData to MNE-like structure for Things-EEG2 processing
        
        Args:
            dataset: MindBigData dataset
            
        Returns:
            Tuple of (test_data, train_data) in Things-EEG2 format
        """
        logger.info("🔄 Converting to MNE-compatible structure...")
        
        eeg_data = dataset['eeg_data']  # (n_trials, 14, 256)
        labels = dataset['labels']      # (n_trials,)
        
        # Split into train/test (80/20)
        n_trials = len(eeg_data)
        n_train = int(0.8 * n_trials)
        
        # Shuffle data
        indices = np.random.permutation(n_trials)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        train_data = eeg_data[train_indices]
        train_labels = labels[train_indices]
        test_data = eeg_data[test_indices]
        test_labels = labels[test_indices]
        
        logger.info(f"   Train: {train_data.shape}")
        logger.info(f"   Test: {test_data.shape}")
        
        # Organize by conditions (digits)
        train_by_condition = self.organize_by_conditions(train_data, train_labels)
        test_by_condition = self.organize_by_conditions(test_data, test_labels)
        
        # Create MNE info structure
        info = mne.create_info(
            ch_names=self.epoc_channels,
            sfreq=dataset['sampling_rate'],
            ch_types=['eeg'] * len(self.epoc_channels)
        )
        
        # Convert to MNE epochs format
        train_epochs = self.create_mne_epochs(train_by_condition, info)
        test_epochs = self.create_mne_epochs(test_by_condition, info)
        
        return test_epochs, train_epochs
    
    def organize_by_conditions(self, eeg_data: np.ndarray, labels: np.ndarray) -> Dict:
        """Organize data by digit conditions"""
        logger.info("📊 Organizing data by digit conditions...")
        
        conditions = {}
        
        for digit in self.digit_conditions:
            digit_mask = labels == digit
            digit_data = eeg_data[digit_mask]
            
            if len(digit_data) > 0:
                conditions[digit] = digit_data
                logger.info(f"   Digit {digit}: {len(digit_data)} trials")
            else:
                logger.warning(f"   Digit {digit}: No trials found")
        
        return conditions
    
    def create_mne_epochs(self, conditions: Dict, info: mne.Info) -> Dict:
        """Create MNE epochs from condition data"""
        logger.info("🧠 Creating MNE epochs structure...")
        
        epochs_by_condition = {}
        
        for digit, digit_data in conditions.items():
            if len(digit_data) == 0:
                continue
            
            # Create events for this condition
            n_trials = len(digit_data)
            events = np.zeros((n_trials, 3), dtype=int)
            events[:, 0] = np.arange(n_trials) * info['sfreq']  # Sample indices
            events[:, 2] = digit  # Event ID (digit)
            
            # Create epochs
            epochs = mne.EpochsArray(
                digit_data,  # (n_trials, n_channels, n_times)
                info,
                events=events,
                event_id={str(digit): digit},
                tmin=0.0,
                verbose=False
            )
            
            epochs_by_condition[digit] = epochs
        
        logger.info(f"✅ Created epochs for {len(epochs_by_condition)} conditions")
        return epochs_by_condition
    
    def adapt_to_things_eeg2_format(self, test_epochs: Dict, train_epochs: Dict) -> Tuple:
        """
        Adapt to Things-EEG2 expected format
        
        Returns format expected by mvnn() and save_prepr()
        """
        logger.info("🔄 Adapting to Things-EEG2 format...")
        
        # Convert epochs dict to format expected by Things-EEG2
        # Things-EEG2 expects: sessions × conditions × repetitions × channels × times
        
        # For MindBigData, we simulate 1 session with multiple conditions
        epoched_test = {}
        epoched_train = {}
        
        # Test data
        epoched_test[0] = {}  # Session 0
        for digit, epochs in test_epochs.items():
            data = epochs.get_data()  # (n_trials, n_channels, n_times)
            epoched_test[0][digit] = data
        
        # Train data  
        epoched_train[0] = {}  # Session 0
        img_conditions_train = []
        
        for digit, epochs in train_epochs.items():
            data = epochs.get_data()  # (n_trials, n_channels, n_times)
            epoched_train[0][digit] = data
            img_conditions_train.extend([digit] * len(data))
        
        # Get channel names and times from first available epochs
        first_epochs = next(iter(train_epochs.values()))
        ch_names = first_epochs.ch_names
        times = first_epochs.times
        
        logger.info(f"✅ Adapted to Things-EEG2 format:")
        logger.info(f"   Test conditions: {len(epoched_test[0])}")
        logger.info(f"   Train conditions: {len(epoched_train[0])}")
        logger.info(f"   Channels: {len(ch_names)}")
        logger.info(f"   Time points: {len(times)}")
        
        return epoched_test, epoched_train, img_conditions_train, ch_names, times
    
    def run_preprocessing_pipeline(self, tsv_path: str, output_dir: str, 
                                 max_trials: Optional[int] = None):
        """
        Run complete preprocessing pipeline
        
        Args:
            tsv_path: Path to MindBigData TSV file
            output_dir: Output directory for preprocessed data
            max_trials: Maximum trials to process (for testing)
        """
        logger.info("🚀 Running MindBigData preprocessing pipeline...")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load MindBigData
        dataset = self.load_mindbigdata(tsv_path, max_trials)
        
        # Convert to MNE structure
        test_epochs, train_epochs = self.create_mne_structure(dataset)
        
        # Adapt to Things-EEG2 format
        epoched_test, epoched_train, img_conditions_train, ch_names, times = \
            self.adapt_to_things_eeg2_format(test_epochs, train_epochs)
        
        # Apply MVNN (multivariate noise normalization)
        logger.info("⚡ Applying MVNN...")
        
        # Create args object for MVNN
        class Args:
            def __init__(self):
                self.mvnn_dim = 'epochs'
                self.n_ses = 1  # Single session for MindBigData
        
        args = Args()
        whitened_test, whitened_train = mvnn(args, epoched_test, epoched_train)
        
        # Save preprocessed data
        logger.info("💾 Saving preprocessed data...")
        
        # Set seed for reproducibility
        seed = 20200220
        
        # Modify args for saving
        args.project_dir = output_dir
        args.sub = 1  # Single subject for MindBigData
        
        save_prepr(args, whitened_test, whitened_train, img_conditions_train, 
                  ch_names, times, seed)
        
        logger.info(f"✅ Preprocessing completed!")
        logger.info(f"   Output saved to: {output_dir}")
        
        return {
            'whitened_test': whitened_test,
            'whitened_train': whitened_train,
            'img_conditions_train': img_conditions_train,
            'ch_names': ch_names,
            'times': times,
            'output_dir': output_dir
        }


def main():
    """Main function for testing"""
    parser = argparse.ArgumentParser(description='MindBigData Preprocessing Adapter')
    parser.add_argument('--tsv_path', type=str, required=True,
                       help='Path to MindBigData TSV file')
    parser.add_argument('--output_dir', type=str, default='./preprocessed_mindbigdata',
                       help='Output directory')
    parser.add_argument('--max_trials', type=int, default=None,
                       help='Maximum trials to process (for testing)')
    
    args = parser.parse_args()
    
    print("🧠 MINDBIGDATA TO THINGS-EEG2 PREPROCESSING ADAPTER")
    print("=" * 80)
    
    try:
        # Initialize adapter
        adapter = MindBigDataAdapter()
        
        # Run preprocessing
        results = adapter.run_preprocessing_pipeline(
            tsv_path=args.tsv_path,
            output_dir=args.output_dir,
            max_trials=args.max_trials
        )
        
        print(f"\n🎉 Preprocessing completed successfully!")
        print(f"   Output directory: {results['output_dir']}")
        print(f"   Channels: {len(results['ch_names'])}")
        print(f"   Time points: {len(results['times'])}")
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
