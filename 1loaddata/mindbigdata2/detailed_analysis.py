#!/usr/bin/env python3
"""
Detailed Analysis of MindBigData Preprocessing Results
=====================================================

Advanced analysis including statistical tests, correlation analysis, and quality metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
import pandas as pd
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load preprocessed data and metadata"""
    print("📂 Loading preprocessed data...")
    
    train_data = np.load("outputs_final/train_data.npy")
    train_labels = np.load("outputs_final/train_labels.npy")
    test_data = np.load("outputs_final/test_data.npy")
    test_labels = np.load("outputs_final/test_labels.npy")
    
    with open("outputs_final/preprocessing_info.pkl", "rb") as f:
        info = pickle.load(f)
    
    print(f"✅ Data loaded: Train {train_data.shape}, Test {test_data.shape}")
    return train_data, train_labels, test_data, test_labels, info

def plot_statistical_analysis(train_data, train_labels):
    """Statistical analysis of EEG data"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('📊 Statistical Analysis of EEG Data', fontsize=16, fontweight='bold')
    
    # 1. Distribution of signal amplitudes by digit
    ax = axes[0, 0]
    
    # Sample data for each digit
    digit_amplitudes = {}
    for digit in range(10):
        digit_indices = np.where(train_labels == digit)[0]
        if len(digit_indices) > 0:
            # Take sample of trials for this digit
            sample_indices = digit_indices[:min(100, len(digit_indices))]
            amplitudes = []
            for idx in sample_indices:
                # RMS amplitude across channels and time
                rms_amp = np.sqrt(np.mean(train_data[idx]**2))
                amplitudes.append(rms_amp)
            digit_amplitudes[digit] = amplitudes
    
    # Create box plot
    data_for_plot = []
    for digit, amps in digit_amplitudes.items():
        for amp in amps:
            data_for_plot.append({'Digit': digit, 'RMS_Amplitude': amp})
    
    df_amp = pd.DataFrame(data_for_plot)
    sns.boxplot(data=df_amp, x='Digit', y='RMS_Amplitude', ax=ax)
    ax.set_title('📈 RMS Amplitude Distribution by Digit', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMS Amplitude (μV)')
    ax.grid(True, alpha=0.3)
    
    # 2. Correlation matrix between channels
    ax = axes[0, 1]
    
    # Calculate correlation matrix using sample data
    sample_indices = np.random.choice(len(train_data), min(1000, len(train_data)), replace=False)
    
    # Flatten time dimension and calculate correlations
    channel_data = []
    for ch in range(train_data.shape[1]):
        ch_signals = []
        for idx in sample_indices:
            ch_signals.extend(train_data[idx, ch])
        channel_data.append(ch_signals)
    
    corr_matrix = np.corrcoef(channel_data)
    
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                     'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title('🔗 Channel Correlation Matrix', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(channel_names)))
    ax.set_yticks(range(len(channel_names)))
    ax.set_xticklabels(channel_names, rotation=45)
    ax.set_yticklabels(channel_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient')
    
    # 3. Signal variance by digit
    ax = axes[0, 2]
    
    digit_variances = {}
    for digit in range(10):
        digit_indices = np.where(train_labels == digit)[0]
        if len(digit_indices) > 0:
            sample_indices = digit_indices[:min(100, len(digit_indices))]
            variances = []
            for idx in sample_indices:
                # Variance across channels and time
                var = np.var(train_data[idx])
                variances.append(var)
            digit_variances[digit] = variances
    
    # Create violin plot
    data_for_plot = []
    for digit, vars in digit_variances.items():
        for var in vars:
            data_for_plot.append({'Digit': digit, 'Variance': var})
    
    df_var = pd.DataFrame(data_for_plot)
    sns.violinplot(data=df_var, x='Digit', y='Variance', ax=ax)
    ax.set_title('📊 Signal Variance by Digit', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance (μV²)')
    ax.grid(True, alpha=0.3)
    
    # 4. Temporal dynamics
    ax = axes[1, 0]
    
    # Calculate average signal evolution over time for each digit
    time_axis = np.linspace(0, train_data.shape[2]/250, train_data.shape[2])
    
    for digit in range(0, 10, 2):  # Plot every other digit for clarity
        digit_indices = np.where(train_labels == digit)[0]
        if len(digit_indices) > 0:
            sample_indices = digit_indices[:min(50, len(digit_indices))]
            
            # Average across trials and channels
            avg_signal = np.mean([np.mean(train_data[idx], axis=0) for idx in sample_indices], axis=0)
            ax.plot(time_axis, avg_signal, label=f'Digit {digit}', linewidth=2)
    
    ax.set_title('⏱️ Average Temporal Dynamics', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Average Amplitude (μV)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. PCA Analysis
    ax = axes[1, 1]
    
    # Prepare data for PCA (flatten spatial-temporal features)
    sample_size = min(2000, len(train_data))
    sample_indices = np.random.choice(len(train_data), sample_size, replace=False)
    
    # Flatten each trial to 1D feature vector
    features = []
    labels_sample = []
    for idx in sample_indices:
        feature_vector = train_data[idx].flatten()
        features.append(feature_vector)
        labels_sample.append(train_labels[idx])
    
    features = np.array(features)
    labels_sample = np.array(labels_sample)
    
    # Apply PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    # Plot PCA results
    scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1], 
                        c=labels_sample, cmap='tab10', alpha=0.6, s=20)
    ax.set_title(f'🔍 PCA Analysis (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    
    # Add colorbar for digits
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Digit')
    
    # 6. Data quality assessment
    ax = axes[1, 2]
    
    # Calculate various quality metrics
    quality_metrics = {
        'SNR (dB)': [],
        'Peak-to-Peak (μV)': [],
        'Zero Crossings': [],
        'Spectral Centroid (Hz)': []
    }
    
    fs = 250  # Sampling frequency
    
    for idx in sample_indices[:100]:  # Sample for efficiency
        # SNR calculation
        signal_power = np.var(train_data[idx])
        # Estimate noise from high-frequency content
        sos = signal.butter(4, 40, btype='low', fs=fs, output='sos')
        filtered_signal = signal.sosfilt(sos, train_data[idx].flatten())
        noise_estimate = train_data[idx].flatten() - filtered_signal
        noise_power = np.var(noise_estimate)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            quality_metrics['SNR (dB)'].append(snr_db)
        
        # Peak-to-peak amplitude
        pp_amp = np.ptp(train_data[idx])
        quality_metrics['Peak-to-Peak (μV)'].append(pp_amp)
        
        # Zero crossings (average across channels)
        avg_signal = np.mean(train_data[idx], axis=0)
        zero_crossings = np.sum(np.diff(np.sign(avg_signal)) != 0)
        quality_metrics['Zero Crossings'].append(zero_crossings)
        
        # Spectral centroid
        freqs, psd = signal.welch(avg_signal, fs, nperseg=min(256, len(avg_signal)))
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
        quality_metrics['Spectral Centroid (Hz)'].append(spectral_centroid)
    
    # Create quality metrics summary
    quality_df = pd.DataFrame(quality_metrics)
    
    # Normalize metrics for radar plot
    normalized_metrics = {}
    for metric in quality_df.columns:
        values = quality_df[metric].values
        if len(values) > 0:
            normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
            normalized_metrics[metric] = np.mean(normalized)
    
    # Simple bar plot of normalized quality metrics
    if normalized_metrics:
        metrics = list(normalized_metrics.keys())
        values = list(normalized_metrics.values())
        
        bars = ax.bar(range(len(metrics)), values, color=plt.cm.viridis(np.linspace(0, 1, len(metrics))))
        ax.set_title('📊 Data Quality Metrics (Normalized)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('Normalized Score')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('mindbigdata_statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(train_data, train_labels, test_data, test_labels, info):
    """Create a comprehensive summary report"""
    
    print("\n📋 COMPREHENSIVE DATA ANALYSIS REPORT")
    print("=" * 60)
    
    # Basic statistics
    total_trials = len(train_data) + len(test_data)
    train_ratio = len(train_data) / total_trials
    
    print(f"📊 DATASET OVERVIEW:")
    print(f"   Total trials: {total_trials:,}")
    print(f"   Train trials: {len(train_data):,} ({train_ratio:.1%})")
    print(f"   Test trials: {len(test_data):,} ({1-train_ratio:.1%})")
    print(f"   Channels: {train_data.shape[1]} (EPOC)")
    print(f"   Time points: {train_data.shape[2]}")
    print(f"   Duration: {train_data.shape[2]/250:.1f} seconds")
    print(f"   Sampling rate: 250 Hz")
    
    # Class distribution
    print(f"\n🎯 CLASS DISTRIBUTION:")
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    
    print(f"   {'Digit':<6} {'Train':<8} {'Test':<8} {'Total':<8} {'Train %':<8}")
    print(f"   {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for digit in range(10):
        train_count = train_counts.get(digit, 0)
        test_count = test_counts.get(digit, 0)
        total_count = train_count + test_count
        train_pct = (train_count / len(train_data)) * 100 if len(train_data) > 0 else 0
        
        print(f"   {digit:<6} {train_count:<8} {test_count:<8} {total_count:<8} {train_pct:<8.1f}")
    
    # Signal quality metrics
    print(f"\n📈 SIGNAL QUALITY METRICS:")
    
    # Sample data for analysis
    sample_indices = np.random.choice(len(train_data), min(1000, len(train_data)), replace=False)
    
    # Calculate metrics
    amplitudes = []
    variances = []
    snr_values = []
    
    for idx in sample_indices:
        # RMS amplitude
        rms_amp = np.sqrt(np.mean(train_data[idx]**2))
        amplitudes.append(rms_amp)
        
        # Variance
        var = np.var(train_data[idx])
        variances.append(var)
        
        # SNR estimation
        signal_power = np.var(train_data[idx])
        sos = signal.butter(4, 40, btype='low', fs=250, output='sos')
        filtered_signal = signal.sosfilt(sos, train_data[idx].flatten())
        noise_estimate = train_data[idx].flatten() - filtered_signal
        noise_power = np.var(noise_estimate)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            snr_values.append(snr_db)
    
    print(f"   RMS Amplitude: {np.mean(amplitudes):.2f} ± {np.std(amplitudes):.2f} μV")
    print(f"   Signal Variance: {np.mean(variances):.2f} ± {np.std(variances):.2f} μV²")
    if snr_values:
        print(f"   SNR: {np.mean(snr_values):.1f} ± {np.std(snr_values):.1f} dB")
    
    # Data integrity checks
    print(f"\n🔍 DATA INTEGRITY:")
    
    # Check for NaN or infinite values
    train_has_nan = np.isnan(train_data).any()
    train_has_inf = np.isinf(train_data).any()
    test_has_nan = np.isnan(test_data).any()
    test_has_inf = np.isinf(test_data).any()
    
    print(f"   Train data - NaN: {'❌' if train_has_nan else '✅'}, Inf: {'❌' if train_has_inf else '✅'}")
    print(f"   Test data - NaN: {'❌' if test_has_nan else '✅'}, Inf: {'❌' if test_has_inf else '✅'}")
    
    # Check label integrity
    train_labels_valid = all(0 <= label <= 9 for label in train_labels)
    test_labels_valid = all(0 <= label <= 9 for label in test_labels)
    
    print(f"   Train labels valid: {'✅' if train_labels_valid else '❌'}")
    print(f"   Test labels valid: {'✅' if test_labels_valid else '❌'}")
    
    # Memory usage
    train_size_mb = train_data.nbytes / (1024**2)
    test_size_mb = test_data.nbytes / (1024**2)
    total_size_mb = train_size_mb + test_size_mb
    
    print(f"\n💾 MEMORY USAGE:")
    print(f"   Train data: {train_size_mb:.1f} MB")
    print(f"   Test data: {test_size_mb:.1f} MB")
    print(f"   Total: {total_size_mb:.1f} MB")
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"   Data is ready for feature extraction and model training")

def main():
    """Main analysis function"""
    print("🔬 DETAILED MINDBIGDATA ANALYSIS")
    print("=" * 50)
    
    # Load data
    train_data, train_labels, test_data, test_labels, info = load_data()
    
    # Create statistical analysis plots
    print("\n📊 Creating statistical analysis...")
    plot_statistical_analysis(train_data, train_labels)
    
    # Create summary report
    create_summary_report(train_data, train_labels, test_data, test_labels, info)
    
    print(f"\n📁 Analysis complete! Saved:")
    print(f"   - mindbigdata_statistical_analysis.png")

if __name__ == "__main__":
    main()
