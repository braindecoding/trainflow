#!/usr/bin/env python3
"""
Domain-Specific Feature Extraction for EEG-to-Image Tasks
========================================================

Specialized feature extraction methods optimized for cross-modal
EEG-to-image reconstruction and classification tasks.

Key innovations:
- Visual cortex inspired features
- Cross-modal alignment features  
- Temporal-spatial coherence
- High-frequency edge preservation
- Phase-amplitude coupling
"""

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class EEGImageFeatureExtractor:
    """Domain-specific feature extractor for EEG-to-image tasks"""
    
    def __init__(self, sampling_rate: int = 128, n_channels: int = 14):
        """
        Initialize domain-specific extractor
        
        Args:
            sampling_rate: EEG sampling rate in Hz
            n_channels: Number of EEG channels
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.scaler = StandardScaler()
        
        # Frequency bands relevant for visual processing (adjusted for sampling rate)
        nyquist = sampling_rate / 2
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, min(50, nyquist - 1)),
            'high_gamma': (min(50, nyquist - 10), nyquist - 1) if nyquist > 50 else None
        }

        # Remove high_gamma if sampling rate is too low
        if self.freq_bands['high_gamma'] is None or self.freq_bands['high_gamma'][0] >= self.freq_bands['high_gamma'][1]:
            del self.freq_bands['high_gamma']
        
        logger.info(f"🧠 EEG-Image Feature Extractor initialized")
        logger.info(f"   Sampling rate: {sampling_rate} Hz")
        logger.info(f"   Channels: {n_channels}")
    
    def extract_visual_cortex_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract features inspired by visual cortex processing
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels, n_timepoints)
            
        Returns:
            Visual cortex inspired features
        """
        logger.info("👁️ Extracting visual cortex inspired features...")
        
        features = []
        
        for sample in eeg_data:
            sample_features = []
            
            # 1. Edge detection like features (high frequency components)
            for ch in range(sample.shape[0]):
                channel_data = sample[ch]
                
                # High-frequency power (edge-like activity)
                freqs, psd = signal.welch(channel_data, fs=self.sampling_rate, nperseg=64)
                high_freq_mask = freqs > 30  # Gamma and above
                high_freq_power = np.sum(psd[high_freq_mask])
                sample_features.append(high_freq_power)
                
                # Gradient features (spatial derivatives)
                gradient = np.gradient(channel_data)
                sample_features.extend([
                    np.mean(np.abs(gradient)),
                    np.std(gradient),
                    np.max(np.abs(gradient))
                ])
                
                # Second derivative (curvature)
                second_deriv = np.gradient(gradient)
                sample_features.extend([
                    np.mean(np.abs(second_deriv)),
                    np.std(second_deriv)
                ])
            
            # 2. Orientation selectivity (directional features)
            # Simulate orientation columns with different temporal filters
            orientations = [0, 45, 90, 135]  # degrees
            for orientation in orientations:
                # Create orientation-selective filter
                filter_length = min(32, len(channel_data) // 4)
                t = np.linspace(0, 2*np.pi, filter_length)
                oriented_filter = np.sin(t + np.radians(orientation))
                
                # Apply to each channel and compute response
                orientation_response = 0
                for ch in range(sample.shape[0]):
                    response = np.convolve(sample[ch], oriented_filter, mode='valid')
                    orientation_response += np.mean(response**2)
                
                sample_features.append(orientation_response / sample.shape[0])
            
            # 3. Spatial frequency analysis
            for ch in range(sample.shape[0]):
                channel_data = sample[ch]
                
                # Different spatial frequency bands
                for band_name, (low, high) in self.freq_bands.items():
                    # Bandpass filter
                    sos = signal.butter(4, [low, high], btype='band', 
                                      fs=self.sampling_rate, output='sos')
                    filtered = signal.sosfilt(sos, channel_data)
                    
                    # Power in this band
                    band_power = np.mean(filtered**2)
                    sample_features.append(band_power)
            
            features.append(sample_features)
        
        features = np.array(features)
        logger.info(f"✅ Visual cortex features: {features.shape}")
        return features
    
    def extract_cross_modal_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract features optimized for cross-modal alignment
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels, n_timepoints)
            
        Returns:
            Cross-modal alignment features
        """
        logger.info("🔄 Extracting cross-modal alignment features...")
        
        features = []
        
        for sample in eeg_data:
            sample_features = []
            
            # 1. Global coherence features
            # Measure how synchronized the brain activity is
            coherence_matrix = np.corrcoef(sample)
            
            # Extract coherence statistics
            triu_indices = np.triu_indices(coherence_matrix.shape[0], k=1)
            coherence_values = coherence_matrix[triu_indices]
            
            sample_features.extend([
                np.mean(coherence_values),
                np.std(coherence_values),
                np.max(coherence_values),
                np.min(coherence_values),
                np.median(coherence_values)
            ])
            
            # 2. Phase synchronization
            # Compute instantaneous phase for each channel
            phases = []
            for ch in range(sample.shape[0]):
                analytic_signal = signal.hilbert(sample[ch])
                phase = np.angle(analytic_signal)
                phases.append(phase)
            
            phases = np.array(phases)
            
            # Phase locking value between channels
            for i in range(sample.shape[0]):
                for j in range(i+1, sample.shape[0]):
                    phase_diff = phases[i] - phases[j]
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    sample_features.append(plv)
            
            # 3. Cross-frequency coupling
            # Measure coupling between different frequency bands
            for ch in range(sample.shape[0]):
                channel_data = sample[ch]
                
                # Extract different frequency components
                freq_components = {}
                for band_name, (low, high) in self.freq_bands.items():
                    sos = signal.butter(4, [low, high], btype='band',
                                      fs=self.sampling_rate, output='sos')
                    filtered = signal.sosfilt(sos, channel_data)
                    freq_components[band_name] = filtered
                
                # Compute phase-amplitude coupling
                # Theta-gamma coupling (important for visual processing)
                theta_phase = np.angle(signal.hilbert(freq_components['theta']))
                gamma_amplitude = np.abs(signal.hilbert(freq_components['gamma']))
                
                # Modulation index
                n_bins = 18
                phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
                mean_amplitude = np.zeros(n_bins)
                
                for i in range(n_bins):
                    mask = (theta_phase >= phase_bins[i]) & (theta_phase < phase_bins[i+1])
                    if np.any(mask):
                        mean_amplitude[i] = np.mean(gamma_amplitude[mask])
                
                # Compute modulation index
                if np.sum(mean_amplitude) > 0:
                    p = mean_amplitude / np.sum(mean_amplitude)
                    p = p[p > 0]  # Remove zeros
                    mi = (np.log(n_bins) + np.sum(p * np.log(p))) / np.log(n_bins)
                else:
                    mi = 0
                
                sample_features.append(mi)
            
            features.append(sample_features)
        
        features = np.array(features)
        logger.info(f"✅ Cross-modal features: {features.shape}")
        return features
    
    def extract_temporal_coherence_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract temporal coherence features
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels, n_timepoints)
            
        Returns:
            Temporal coherence features
        """
        logger.info("⏱️ Extracting temporal coherence features...")
        
        features = []
        
        for sample in eeg_data:
            sample_features = []
            
            # 1. Autocorrelation features
            for ch in range(sample.shape[0]):
                channel_data = sample[ch]
                
                # Autocorrelation at different lags
                max_lag = min(32, len(channel_data) // 4)
                autocorr = np.correlate(channel_data, channel_data, mode='full')
                mid = len(autocorr) // 2
                
                # Extract autocorrelation at specific lags
                lags = [1, 2, 4, 8, 16]
                for lag in lags:
                    if mid + lag < len(autocorr):
                        sample_features.append(autocorr[mid + lag])
                
                # Autocorrelation decay rate
                autocorr_positive = autocorr[mid:mid+max_lag]
                if len(autocorr_positive) > 1:
                    decay_rate = -np.polyfit(range(len(autocorr_positive)), 
                                           np.log(np.abs(autocorr_positive) + 1e-10), 1)[0]
                    sample_features.append(decay_rate)
            
            # 2. Temporal stability
            # Measure how stable the signal is over time
            window_size = len(sample[0]) // 4
            for ch in range(sample.shape[0]):
                channel_data = sample[ch]
                
                # Sliding window variance
                variances = []
                for i in range(0, len(channel_data) - window_size, window_size // 2):
                    window = channel_data[i:i+window_size]
                    variances.append(np.var(window))
                
                if variances:
                    sample_features.extend([
                        np.mean(variances),
                        np.std(variances),
                        np.max(variances) / (np.mean(variances) + 1e-10)
                    ])
            
            # 3. Rhythmicity features
            for ch in range(sample.shape[0]):
                channel_data = sample[ch]
                
                # Spectral entropy (measure of rhythmicity)
                freqs, psd = signal.welch(channel_data, fs=self.sampling_rate, nperseg=64)
                psd_norm = psd / np.sum(psd)
                psd_norm = psd_norm[psd_norm > 0]
                spectral_entropy = -np.sum(psd_norm * np.log(psd_norm))
                sample_features.append(spectral_entropy)
                
                # Peak frequency stability
                peak_freq = freqs[np.argmax(psd)]
                sample_features.append(peak_freq)
                
                # Spectral centroid
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                sample_features.append(spectral_centroid)
            
            features.append(sample_features)
        
        features = np.array(features)
        logger.info(f"✅ Temporal coherence features: {features.shape}")
        return features
    
    def extract_spatial_coherence_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract spatial coherence features
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels, n_timepoints)
            
        Returns:
            Spatial coherence features
        """
        logger.info("🗺️ Extracting spatial coherence features...")
        
        features = []
        
        for sample in eeg_data:
            sample_features = []
            
            # 1. Spatial correlation patterns
            spatial_corr = np.corrcoef(sample)
            
            # Extract spatial correlation statistics
            triu_indices = np.triu_indices(spatial_corr.shape[0], k=1)
            corr_values = spatial_corr[triu_indices]
            
            sample_features.extend([
                np.mean(corr_values),
                np.std(corr_values),
                np.max(corr_values),
                np.min(corr_values),
                stats.skew(corr_values),
                stats.kurtosis(corr_values)
            ])
            
            # 2. Spatial gradients
            # Simulate spatial gradients across channels
            for t in range(0, sample.shape[1], sample.shape[1] // 8):
                spatial_snapshot = sample[:, t]
                
                # Compute spatial gradient (simplified)
                spatial_gradient = np.gradient(spatial_snapshot)
                sample_features.extend([
                    np.mean(np.abs(spatial_gradient)),
                    np.std(spatial_gradient),
                    np.max(np.abs(spatial_gradient))
                ])
            
            # 3. Global field power
            # Measure of global brain activity
            gfp = np.std(sample, axis=0)
            sample_features.extend([
                np.mean(gfp),
                np.std(gfp),
                np.max(gfp),
                np.min(gfp)
            ])
            
            # 4. Spatial complexity
            # Measure how complex the spatial pattern is
            for t in range(0, sample.shape[1], sample.shape[1] // 4):
                spatial_snapshot = sample[:, t]
                
                # Spatial entropy
                hist, _ = np.histogram(spatial_snapshot, bins=10)
                hist_norm = hist / np.sum(hist)
                hist_norm = hist_norm[hist_norm > 0]
                spatial_entropy = -np.sum(hist_norm * np.log(hist_norm))
                sample_features.append(spatial_entropy)
            
            features.append(sample_features)
        
        features = np.array(features)
        logger.info(f"✅ Spatial coherence features: {features.shape}")
        return features
    
    def extract_high_frequency_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract high-frequency features for edge preservation
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels, n_timepoints)
            
        Returns:
            High-frequency features
        """
        logger.info("⚡ Extracting high-frequency features...")
        
        features = []
        
        for sample in eeg_data:
            sample_features = []
            
            # 1. High-frequency power
            for ch in range(sample.shape[0]):
                channel_data = sample[ch]
                
                # Gamma band power
                low, high = self.freq_bands['gamma']
                sos = signal.butter(4, [low, high], btype='band',
                                  fs=self.sampling_rate, output='sos')
                gamma_filtered = signal.sosfilt(sos, channel_data)
                gamma_power = np.mean(gamma_filtered**2)
                sample_features.append(gamma_power)
                
                # High gamma band power (if available)
                if 'high_gamma' in self.freq_bands:
                    low, high = self.freq_bands['high_gamma']
                    sos = signal.butter(4, [low, high],
                                      btype='band', fs=self.sampling_rate, output='sos')
                    high_gamma_filtered = signal.sosfilt(sos, channel_data)
                    high_gamma_power = np.mean(high_gamma_filtered**2)
                    sample_features.append(high_gamma_power)
            
            # 2. Sharp transients (edge-like events)
            for ch in range(sample.shape[0]):
                channel_data = sample[ch]
                
                # Detect sharp transients using derivative
                first_deriv = np.diff(channel_data)
                second_deriv = np.diff(first_deriv)
                
                # Count sharp transients
                threshold = np.std(second_deriv) * 3
                sharp_transients = np.sum(np.abs(second_deriv) > threshold)
                sample_features.append(sharp_transients)
                
                # Mean absolute derivative
                sample_features.append(np.mean(np.abs(first_deriv)))
                sample_features.append(np.mean(np.abs(second_deriv)))
            
            # 3. Spectral edge frequency
            for ch in range(sample.shape[0]):
                channel_data = sample[ch]
                
                freqs, psd = signal.welch(channel_data, fs=self.sampling_rate, nperseg=64)
                
                # Spectral edge frequency (95% of power)
                cumulative_power = np.cumsum(psd)
                total_power = cumulative_power[-1]
                edge_freq_idx = np.argmax(cumulative_power >= 0.95 * total_power)
                edge_freq = freqs[edge_freq_idx]
                sample_features.append(edge_freq)
                
                # High frequency ratio
                high_freq_mask = freqs > 20
                high_freq_power = np.sum(psd[high_freq_mask])
                total_power = np.sum(psd)
                hf_ratio = high_freq_power / (total_power + 1e-10)
                sample_features.append(hf_ratio)
            
            features.append(sample_features)
        
        features = np.array(features)
        logger.info(f"✅ High-frequency features: {features.shape}")
        return features
    
    def extract_all_domain_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract all domain-specific features
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels, n_timepoints)
            
        Returns:
            Combined domain-specific features
        """
        logger.info("🧠 Extracting all domain-specific features...")
        
        all_features = []
        
        # Extract each type of features
        visual_features = self.extract_visual_cortex_features(eeg_data)
        all_features.append(visual_features)
        
        cross_modal_features = self.extract_cross_modal_features(eeg_data)
        all_features.append(cross_modal_features)
        
        temporal_features = self.extract_temporal_coherence_features(eeg_data)
        all_features.append(temporal_features)
        
        spatial_features = self.extract_spatial_coherence_features(eeg_data)
        all_features.append(spatial_features)
        
        hf_features = self.extract_high_frequency_features(eeg_data)
        all_features.append(hf_features)
        
        # Combine all features
        combined_features = np.hstack(all_features)
        
        # Normalize features
        combined_features = self.scaler.fit_transform(combined_features)
        
        logger.info(f"✅ All domain features extracted: {combined_features.shape}")
        return combined_features


def test_domain_extractor():
    """Test the domain-specific feature extractor"""
    print("🧪 Testing Domain-Specific Feature Extractor...")
    
    # Create synthetic EEG data
    n_samples, n_channels, n_timepoints = 10, 14, 128
    eeg_data = np.random.randn(n_samples, n_channels, n_timepoints)
    
    # Initialize extractor
    extractor = EEGImageFeatureExtractor(sampling_rate=128, n_channels=14)
    
    # Extract features
    features = extractor.extract_all_domain_features(eeg_data)
    
    print(f"✅ Test completed:")
    print(f"   Input: {eeg_data.shape}")
    print(f"   Output: {features.shape}")
    print(f"   Features per sample: {features.shape[1]}")


if __name__ == "__main__":
    test_domain_extractor()
