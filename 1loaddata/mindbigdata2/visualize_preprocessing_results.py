#!/usr/bin/env python3
"""
MindBigData Preprocessing Results Visualization
==============================================

Comprehensive visualization of preprocessed EEG data from MindBigData.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
import pandas as pd
from scipy import signal
from scipy.stats import zscore, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load preprocessed data and metadata"""
    print("📂 Loading preprocessed data...")
    
    # Load data
    train_data = np.load("outputs_final/train_data.npy")
    train_labels = np.load("outputs_final/train_labels.npy")
    test_data = np.load("outputs_final/test_data.npy")
    test_labels = np.load("outputs_final/test_labels.npy")
    
    # Load preprocessing info
    with open("outputs_final/preprocessing_info.pkl", "rb") as f:
        info = pickle.load(f)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Train: {train_data.shape} | Labels: {train_labels.shape}")
    print(f"   Test:  {test_data.shape} | Labels: {test_labels.shape}")
    
    return train_data, train_labels, test_data, test_labels, info

def plot_data_overview(train_data, train_labels, test_data, test_labels, info):
    """Plot data overview and statistics"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🧠 MindBigData Preprocessing Results Overview', fontsize=16, fontweight='bold')
    
    # 1. Dataset size comparison
    ax = axes[0, 0]
    sizes = [len(train_data), len(test_data)]
    labels = ['Train', 'Test']
    colors = ['#FF6B6B', '#4ECDC4']
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                      startangle=90, textprops={'fontsize': 12})
    ax.set_title('📊 Train/Test Split', fontsize=14, fontweight='bold')
    
    # Add count annotations
    for i, (wedge, autotext) in enumerate(zip(wedges, autotexts)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 0.7 * np.cos(np.radians(angle))
        y = 0.7 * np.sin(np.radians(angle))
        ax.annotate(f'{sizes[i]} trials', xy=(x, y), ha='center', va='center', 
                   fontsize=10, fontweight='bold')
    
    # 2. Class distribution - Train
    ax = axes[0, 1]
    train_counts = Counter(train_labels)
    digits = sorted(train_counts.keys())
    counts = [train_counts[d] for d in digits]
    
    bars = ax.bar(digits, counts, color=plt.cm.Set3(np.linspace(0, 1, len(digits))))
    ax.set_title('🎯 Train Class Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Digit')
    ax.set_ylabel('Number of Trials')
    ax.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Class distribution - Test
    ax = axes[0, 2]
    test_counts = Counter(test_labels)
    digits = sorted(test_counts.keys())
    counts = [test_counts[d] for d in digits]
    
    bars = ax.bar(digits, counts, color=plt.cm.Set3(np.linspace(0, 1, len(digits))))
    ax.set_title('🎯 Test Class Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Digit')
    ax.set_ylabel('Number of Trials')
    ax.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Data shape information
    ax = axes[1, 0]
    ax.axis('off')
    
    info_text = f"""
📋 DATA SPECIFICATIONS:
    
🔹 Train Shape: {train_data.shape}
🔹 Test Shape: {test_data.shape}
🔹 Channels: {train_data.shape[1]} (EPOC)
🔹 Time Points: {train_data.shape[2]}
🔹 Sampling Rate: 250 Hz
🔹 Duration: {train_data.shape[2]/250:.1f} seconds
🔹 Total Trials: {len(train_data) + len(test_data):,}
🔹 Classes: {len(np.unique(train_labels))} digits (0-9)
    """
    
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
            facecolor="lightblue", alpha=0.8))
    
    # 5. Signal amplitude distribution
    ax = axes[1, 1]
    
    # Sample some data for amplitude analysis
    sample_indices = np.random.choice(len(train_data), min(100, len(train_data)), replace=False)
    sample_data = train_data[sample_indices]
    
    amplitudes = sample_data.flatten()
    ax.hist(amplitudes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title('📈 Signal Amplitude Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Amplitude (μV)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_amp = np.mean(amplitudes)
    std_amp = np.std(amplitudes)
    ax.axvline(mean_amp, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_amp:.2f}')
    ax.axvline(mean_amp + std_amp, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_amp:.2f}')
    ax.axvline(mean_amp - std_amp, color='orange', linestyle='--', alpha=0.7)
    ax.legend()
    
    # 6. Channel-wise statistics
    ax = axes[1, 2]
    
    # Calculate channel means and stds
    channel_means = np.mean(train_data, axis=(0, 2))
    channel_stds = np.std(train_data, axis=(0, 2))
    
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                     'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    x_pos = np.arange(len(channel_names))
    
    bars1 = ax.bar(x_pos - 0.2, channel_means, 0.4, label='Mean', alpha=0.8, color='lightcoral')
    bars2 = ax.bar(x_pos + 0.2, channel_stds, 0.4, label='Std', alpha=0.8, color='lightblue')
    
    ax.set_title('📊 Channel-wise Statistics', fontsize=14, fontweight='bold')
    ax.set_xlabel('EEG Channels')
    ax.set_ylabel('Amplitude (μV)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(channel_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mindbigdata_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_sample_signals(train_data, train_labels):
    """Plot sample EEG signals for each digit"""
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    fig.suptitle('🧠 Sample EEG Signals by Digit Class', fontsize=16, fontweight='bold')
    
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                     'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    time_axis = np.linspace(0, train_data.shape[2]/250, train_data.shape[2])
    
    for digit in range(10):
        row = digit // 5
        col = digit % 5
        ax = axes[row, col]
        
        # Find samples for this digit
        digit_indices = np.where(train_labels == digit)[0]
        if len(digit_indices) > 0:
            # Take first sample
            sample_idx = digit_indices[0]
            sample_data = train_data[sample_idx]
            
            # Plot a few representative channels
            selected_channels = [0, 3, 6, 9, 13]  # AF3, FC5, O1, T8, AF4
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_channels)))
            
            for i, ch_idx in enumerate(selected_channels):
                signal_data = sample_data[ch_idx]
                ax.plot(time_axis, signal_data, color=colors[i], 
                       label=channel_names[ch_idx], alpha=0.8, linewidth=1.5)
            
            ax.set_title(f'Digit {digit}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (μV)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Set consistent y-axis limits
            ax.set_ylim(-50, 50)
        else:
            ax.text(0.5, 0.5, f'No data\nfor digit {digit}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_title(f'Digit {digit}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('mindbigdata_sample_signals.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_spectral_analysis(train_data, train_labels):
    """Plot spectral analysis of EEG signals"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🌊 Spectral Analysis of EEG Signals', fontsize=16, fontweight='bold')
    
    # Sampling parameters
    fs = 250  # Hz
    nperseg = min(256, train_data.shape[2])
    
    # 1. Average power spectral density
    ax = axes[0, 0]
    
    # Calculate PSD for a sample of data
    sample_indices = np.random.choice(len(train_data), min(50, len(train_data)), replace=False)
    
    all_psds = []
    for idx in sample_indices:
        # Average across channels
        signal_avg = np.mean(train_data[idx], axis=0)
        freqs, psd = signal.welch(signal_avg, fs, nperseg=nperseg)
        all_psds.append(psd)
    
    # Average PSD across samples
    mean_psd = np.mean(all_psds, axis=0)
    std_psd = np.std(all_psds, axis=0)
    
    ax.loglog(freqs, mean_psd, 'b-', linewidth=2, label='Mean PSD')
    ax.fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd, 
                    alpha=0.3, color='blue', label='±1σ')
    
    # Mark frequency bands
    ax.axvspan(0.5, 4, alpha=0.2, color='red', label='Delta (0.5-4 Hz)')
    ax.axvspan(4, 8, alpha=0.2, color='orange', label='Theta (4-8 Hz)')
    ax.axvspan(8, 13, alpha=0.2, color='yellow', label='Alpha (8-13 Hz)')
    ax.axvspan(13, 30, alpha=0.2, color='green', label='Beta (13-30 Hz)')
    ax.axvspan(30, 100, alpha=0.2, color='purple', label='Gamma (30-100 Hz)')
    
    ax.set_title('📊 Average Power Spectral Density', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Frequency band power by digit
    ax = axes[0, 1]
    
    # Define frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 50)
    }
    
    band_powers = {band: [] for band in bands}
    digits_for_bands = []
    
    # Sample data for each digit
    for digit in range(10):
        digit_indices = np.where(train_labels == digit)[0]
        if len(digit_indices) > 0:
            # Take a few samples
            sample_indices = digit_indices[:min(10, len(digit_indices))]
            
            for idx in sample_indices:
                # Average across channels
                signal_avg = np.mean(train_data[idx], axis=0)
                freqs, psd = signal.welch(signal_avg, fs, nperseg=nperseg)
                
                # Calculate band powers
                for band_name, (low, high) in bands.items():
                    band_mask = (freqs >= low) & (freqs <= high)
                    band_power = np.trapz(psd[band_mask], freqs[band_mask])
                    band_powers[band_name].append(band_power)
                    
                digits_for_bands.append(digit)
    
    # Create DataFrame for easier plotting
    band_data = []
    for band_name, powers in band_powers.items():
        for power, digit in zip(powers, digits_for_bands):
            band_data.append({'Band': band_name, 'Power': power, 'Digit': digit})
    
    df_bands = pd.DataFrame(band_data)
    
    # Box plot of band powers by digit
    if not df_bands.empty:
        sns.boxplot(data=df_bands, x='Digit', y='Power', hue='Band', ax=ax)
        ax.set_title('🎵 Frequency Band Power by Digit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Digit')
        ax.set_ylabel('Band Power')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Channel-wise spectral analysis
    ax = axes[1, 0]
    
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                     'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    # Select a few representative channels
    selected_channels = [0, 6, 7, 13]  # AF3, O1, O2, AF4
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_channels)))
    
    for i, ch_idx in enumerate(selected_channels):
        # Calculate average PSD for this channel
        channel_psds = []
        for idx in sample_indices:
            freqs, psd = signal.welch(train_data[idx, ch_idx], fs, nperseg=nperseg)
            channel_psds.append(psd)
        
        mean_psd = np.mean(channel_psds, axis=0)
        ax.loglog(freqs, mean_psd, color=colors[i], linewidth=2, 
                 label=channel_names[ch_idx])
    
    ax.set_title('📡 Channel-wise Power Spectral Density', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Signal quality metrics
    ax = axes[1, 1]

    # Calculate signal quality metrics
    snr_values = []
    kurtosis_values = []

    for idx in sample_indices:
        # Calculate average SNR across channels
        channel_snrs = []
        channel_kurtosis = []

        for ch in range(train_data.shape[1]):
            # Calculate SNR for this channel
            signal_power = np.var(train_data[idx, ch])

            # Estimate noise as high-frequency content
            sos = signal.butter(4, 40, btype='low', fs=fs, output='sos')
            filtered_signal = signal.sosfilt(sos, train_data[idx, ch])
            noise_estimate = train_data[idx, ch] - filtered_signal
            noise_power = np.var(noise_estimate)

            if noise_power > 0:
                snr = signal_power / noise_power
                channel_snrs.append(10 * np.log10(snr))  # Convert to dB

            # Calculate kurtosis for this channel
            channel_kurtosis.append(kurtosis(train_data[idx, ch]))

        # Average across channels for this trial
        if channel_snrs:
            snr_values.append(np.mean(channel_snrs))
            kurtosis_values.append(np.mean(channel_kurtosis))

    # Plot signal quality metrics
    if len(snr_values) > 0 and len(kurtosis_values) > 0:
        ax.scatter(snr_values, kurtosis_values, alpha=0.6, s=50)
        ax.set_title('📈 Signal Quality Metrics', fontsize=14, fontweight='bold')
        ax.set_xlabel('Average SNR (dB)')
        ax.set_ylabel('Average Kurtosis')
        ax.grid(True, alpha=0.3)

        # Add quality regions
        ax.axhspan(-2, 2, alpha=0.2, color='green', label='Normal Kurtosis')
        if len(snr_values) > 0:
            snr_range = max(snr_values) - min(snr_values)
            mid_snr = (max(snr_values) + min(snr_values)) / 2
            ax.axvspan(mid_snr - snr_range*0.2, mid_snr + snr_range*0.2,
                      alpha=0.2, color='blue', label='Good SNR Range')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No valid SNR data', transform=ax.transAxes,
               ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('mindbigdata_spectral_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main visualization function"""
    print("🎨 MINDBIGDATA PREPROCESSING VISUALIZATION")
    print("=" * 60)
    
    # Load data
    train_data, train_labels, test_data, test_labels, info = load_data()
    
    # Create visualizations
    print("\n📊 Creating overview plots...")
    plot_data_overview(train_data, train_labels, test_data, test_labels, info)
    
    print("\n🧠 Creating sample signal plots...")
    plot_sample_signals(train_data, train_labels)
    
    print("\n🌊 Creating spectral analysis plots...")
    plot_spectral_analysis(train_data, train_labels)
    
    print("\n🎉 Visualization complete!")
    print("📁 Saved plots:")
    print("   - mindbigdata_overview.png")
    print("   - mindbigdata_sample_signals.png") 
    print("   - mindbigdata_spectral_analysis.png")

if __name__ == "__main__":
    main()
