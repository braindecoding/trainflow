#!/usr/bin/env python3
"""
MindBigData EEG Preprocessing Pipeline
=====================================

Adapted from Things-EEG2 preprocessing pipeline for MindBigData format.
Implements advanced EEG preprocessing techniques for digit classification.

Key adaptations:
- 14-channel EEG (vs 63-channel Things-EEG2)
- 128Hz sampling rate (vs 250Hz)
- Single trial format (vs multi-session)
- Digit labels (vs image conditions)

References:
- Things-EEG2: https://www.sciencedirect.com/science/article/pii/S1053811922008758
- MindBigData: https://mindbigdata.com/opendb/
"""

import numpy as np
import pandas as pd
import mne
import os
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MindBigDataPreprocessor:
    """Advanced EEG preprocessing for MindBigData"""
    
    def __init__(self, 
                 sfreq=128,
                 epoch_tmin=-0.2,
                 epoch_tmax=0.8,
                 baseline=(None, 0),
                 filter_low=0.5,
                 filter_high=50.0,
                 apply_mvnn=True,
                 random_seed=42):
        """
        Initialize MindBigData preprocessor
        
        Args:
            sfreq: Target sampling frequency (Hz)
            epoch_tmin: Epoch start time (s)
            epoch_tmax: Epoch end time (s)
            baseline: Baseline correction window
            filter_low: High-pass filter frequency (Hz)
            filter_high: Low-pass filter frequency (Hz)
            apply_mvnn: Whether to apply multivariate noise normalization
            random_seed: Random seed for reproducibility
        """
        self.sfreq = sfreq
        self.epoch_tmin = epoch_tmin
        self.epoch_tmax = epoch_tmax
        self.baseline = baseline
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.apply_mvnn = apply_mvnn
        self.random_seed = random_seed
        
        # MindBigData channel configuration (EPOC 14 channels)
        # EPOC headset layout (Emotiv EPOC):
        # AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
        self.ch_names = [
            'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
            'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
        ]
        
        # Channel types
        self.ch_types = ['eeg'] * len(self.ch_names)
        
        # Set random seed
        np.random.seed(random_seed)
        
        logger.info("🧠 MindBigData Preprocessor initialized")
        logger.info(f"   Channels: {len(self.ch_names)}")
        logger.info(f"   Sampling rate: {sfreq} Hz")
        logger.info(f"   Epoch: {epoch_tmin} to {epoch_tmax} s")
    
    def load_mindbigdata_trial(self, trial_data, digit_label):
        """
        Load single MindBigData trial and convert to MNE format
        
        Args:
            trial_data: EEG data array (n_channels, n_timepoints)
            digit_label: Digit label (0-9)
            
        Returns:
            MNE Epochs object
        """
        # Ensure correct shape
        if trial_data.shape[0] != len(self.ch_names):
            logger.warning(f"Channel mismatch: expected {len(self.ch_names)}, got {trial_data.shape[0]}")
            # Pad or truncate channels if needed
            if trial_data.shape[0] < len(self.ch_names):
                padding = np.zeros((len(self.ch_names) - trial_data.shape[0], trial_data.shape[1]))
                trial_data = np.vstack([trial_data, padding])
            else:
                trial_data = trial_data[:len(self.ch_names)]
        
        # Create MNE info structure
        info = mne.create_info(
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            ch_types=self.ch_types
        )
        
        # Create raw data (add batch dimension)
        raw_data = trial_data.reshape(len(self.ch_names), -1)
        
        # Create MNE Raw object
        raw = mne.io.RawArray(raw_data, info, verbose=False)
        
        # Create events (single event at time 0)
        n_samples = raw_data.shape[1]
        event_time = n_samples // 2  # Event in the middle
        events = np.array([[event_time, 0, digit_label]])
        
        # Create epochs
        epochs = mne.Epochs(
            raw, events,
            event_id={str(digit_label): digit_label},
            tmin=self.epoch_tmin,
            tmax=self.epoch_tmax,
            baseline=self.baseline,
            preload=True,
            verbose=False
        )
        
        return epochs
    
    def apply_filtering(self, epochs):
        """Apply bandpass filtering"""
        logger.info(f"🔧 Applying bandpass filter: {self.filter_low}-{self.filter_high} Hz")
        
        epochs.filter(
            l_freq=self.filter_low,
            h_freq=self.filter_high,
            fir_design='firwin',
            verbose=False
        )
        
        return epochs
    
    def apply_artifact_rejection(self, epochs, reject_criteria=None):
        """Apply artifact rejection"""
        if reject_criteria is None:
            # Default rejection criteria for EEG (in volts)
            reject_criteria = {
                'eeg': 100e-6  # 100 µV
            }
        
        logger.info("🔍 Applying artifact rejection...")
        
        # Apply rejection
        epochs.drop_bad(reject=reject_criteria, verbose=False)
        
        n_rejected = len(epochs.drop_log) - len(epochs)
        logger.info(f"   Rejected {n_rejected} epochs due to artifacts")
        
        return epochs
    
    def apply_mvnn_method(self, epochs_list):
        """
        Apply Multivariate Noise Normalization (MVNN)
        
        Args:
            epochs_list: List of MNE Epochs objects
            
        Returns:
            List of whitened epochs
        """
        logger.info("⚡ Applying Multivariate Noise Normalization...")
        
        # Collect all epoch data for covariance estimation
        all_data = []
        for epochs in epochs_list:
            data = epochs.get_data()  # (n_epochs, n_channels, n_times)
            all_data.append(data)
        
        # Concatenate all data
        combined_data = np.concatenate(all_data, axis=0)
        
        # Compute covariance matrix for each time point
        n_epochs, n_channels, n_times = combined_data.shape
        cov_matrices = []
        
        for t in range(n_times):
            data_t = combined_data[:, :, t]  # (n_epochs, n_channels)
            cov_t = np.cov(data_t.T)  # (n_channels, n_channels)
            cov_matrices.append(cov_t)
        
        # Average covariance matrices across time
        avg_cov = np.mean(cov_matrices, axis=0)
        
        # Compute whitening matrix (inverse square root of covariance)
        eigenvals, eigenvecs = np.linalg.eigh(avg_cov)
        
        # Regularization to avoid numerical issues
        eigenvals = np.maximum(eigenvals, 1e-12)
        
        # Whitening matrix
        whitening_matrix = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
        
        # Apply whitening to each epoch set
        whitened_epochs = []
        for epochs in epochs_list:
            data = epochs.get_data()  # (n_epochs, n_channels, n_times)
            
            # Apply whitening
            whitened_data = np.zeros_like(data)
            for e in range(data.shape[0]):
                for t in range(data.shape[2]):
                    whitened_data[e, :, t] = whitening_matrix @ data[e, :, t]
            
            # Create new epochs with whitened data
            whitened_epochs_obj = epochs.copy()
            whitened_epochs_obj._data = whitened_data
            whitened_epochs.append(whitened_epochs_obj)
        
        logger.info(f"✅ MVNN applied to {len(epochs_list)} epoch sets")
        return whitened_epochs
    
    def preprocess_trial(self, trial_data, digit_label):
        """
        Preprocess single trial
        
        Args:
            trial_data: EEG data array (n_channels, n_timepoints)
            digit_label: Digit label (0-9)
            
        Returns:
            Preprocessed epoch data
        """
        # Load trial as MNE epochs
        epochs = self.load_mindbigdata_trial(trial_data, digit_label)
        
        # Apply filtering
        epochs = self.apply_filtering(epochs)
        
        # Apply artifact rejection
        epochs = self.apply_artifact_rejection(epochs)
        
        return epochs
    
    def preprocess_dataset(self, eeg_data, labels, apply_mvnn_override=None):
        """
        Preprocess entire dataset
        
        Args:
            eeg_data: Array of EEG trials (n_trials, n_channels, n_timepoints)
            labels: Array of digit labels (n_trials,)
            apply_mvnn: Whether to apply MVNN (overrides instance setting)
            
        Returns:
            Preprocessed data dictionary
        """
        logger.info(f"🚀 Preprocessing dataset: {len(eeg_data)} trials")
        
        if apply_mvnn_override is None:
            apply_mvnn = self.apply_mvnn
        else:
            apply_mvnn = apply_mvnn_override
        
        # Process each trial
        epochs_list = []
        valid_labels = []
        
        for i, (trial, label) in enumerate(zip(eeg_data, labels)):
            try:
                epochs = self.preprocess_trial(trial, label)
                
                # Check if epochs survived artifact rejection
                if len(epochs) > 0:
                    epochs_list.append(epochs)
                    valid_labels.append(label)
                else:
                    logger.warning(f"Trial {i} rejected due to artifacts")
                    
            except Exception as e:
                logger.error(f"Error processing trial {i}: {e}")
                continue
            
            if (i + 1) % 100 == 0:
                logger.info(f"   Processed {i + 1}/{len(eeg_data)} trials")
        
        logger.info(f"✅ Successfully processed {len(epochs_list)}/{len(eeg_data)} trials")
        
        # Apply MVNN if requested
        if apply_mvnn and len(epochs_list) > 0:
            epochs_list = self.apply_mvnn_method(epochs_list)
        
        # Extract final data
        processed_data = []
        final_labels = []
        
        for epochs, label in zip(epochs_list, valid_labels):
            data = epochs.get_data()  # (n_epochs, n_channels, n_times)
            
            # Should be single epoch per trial
            if data.shape[0] == 1:
                processed_data.append(data[0])  # (n_channels, n_times)
                final_labels.append(label)
            else:
                logger.warning(f"Unexpected epoch count: {data.shape[0]}")
        
        processed_data = np.array(processed_data)
        final_labels = np.array(final_labels)
        
        # Get time points and channel names
        if epochs_list:
            times = epochs_list[0].times
            ch_names = epochs_list[0].ch_names
        else:
            times = None
            ch_names = self.ch_names
        
        result = {
            'data': processed_data,
            'labels': final_labels,
            'times': times,
            'ch_names': ch_names,
            'sfreq': self.sfreq,
            'preprocessing_params': {
                'filter_low': self.filter_low,
                'filter_high': self.filter_high,
                'epoch_tmin': self.epoch_tmin,
                'epoch_tmax': self.epoch_tmax,
                'baseline': self.baseline,
                'mvnn_applied': apply_mvnn,
                'random_seed': self.random_seed
            }
        }
        
        logger.info(f"🎉 Preprocessing completed!")
        logger.info(f"   Final data shape: {processed_data.shape}")
        logger.info(f"   Labels shape: {final_labels.shape}")
        
        return result


def test_preprocessor():
    """Test the MindBigData preprocessor"""
    print("🧪 Testing MindBigData Preprocessor...")

    # Create synthetic data with longer time series for proper epoching
    n_trials, n_channels, n_timepoints = 10, 14, 256  # Longer time series
    eeg_data = np.random.randn(n_trials, n_channels, n_timepoints) * 1e-5  # Realistic EEG amplitudes
    labels = np.random.randint(0, 10, n_trials)
    
    # Initialize preprocessor with more lenient settings for testing
    preprocessor = MindBigDataPreprocessor(
        epoch_tmin=-0.1,  # Shorter epoch window
        epoch_tmax=0.5,   # Shorter epoch window
        filter_low=1.0,   # Higher low-pass to avoid edge effects
        filter_high=40.0  # Lower high-pass
    )
    
    # Preprocess data
    result = preprocessor.preprocess_dataset(eeg_data, labels)
    
    print(f"✅ Test completed:")
    print(f"   Input: {eeg_data.shape}")
    print(f"   Output: {result['data'].shape}")
    print(f"   Labels: {result['labels'].shape}")
    print(f"   Channels: {len(result['ch_names'])}")
    if result['times'] is not None:
        print(f"   Time points: {len(result['times'])}")
    else:
        print(f"   Time points: None (no valid epochs)")


if __name__ == "__main__":
    test_preprocessor()
