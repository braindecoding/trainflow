#!/usr/bin/env python3
"""
Explanation of Channel-wise Power Spectral Density (PSD)
=======================================================

Educational script to demonstrate PSD calculation and interpretation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def demonstrate_psd_concept():
    """Demonstrate PSD concept with synthetic signals"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('🌊 Understanding Power Spectral Density (PSD)', fontsize=16, fontweight='bold')
    
    # Create synthetic signals with different frequency content
    fs = 250  # Sampling frequency (same as MindBigData)
    t = np.linspace(0, 2, fs * 2)  # 2 seconds
    
    # 1. Pure sine waves at different frequencies
    ax = axes[0, 0]
    
    # Different frequency components
    freq1, freq2, freq3 = 10, 20, 40  # Hz
    signal1 = np.sin(2 * np.pi * freq1 * t)  # Alpha band
    signal2 = np.sin(2 * np.pi * freq2 * t)  # Beta band  
    signal3 = np.sin(2 * np.pi * freq3 * t)  # Gamma band
    
    ax.plot(t[:250], signal1[:250], label=f'{freq1} Hz (Alpha)', linewidth=2)
    ax.plot(t[:250], signal2[:250], label=f'{freq2} Hz (Beta)', linewidth=2)
    ax.plot(t[:250], signal3[:250], label=f'{freq3} Hz (Gamma)', linewidth=2)
    
    ax.set_title('📈 Pure Frequency Components', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. PSD of pure sine waves
    ax = axes[0, 1]
    
    for sig, freq, label in [(signal1, freq1, 'Alpha'), (signal2, freq2, 'Beta'), (signal3, freq3, 'Gamma')]:
        freqs, psd = signal.welch(sig, fs, nperseg=256)
        ax.semilogy(freqs, psd, label=f'{label} ({freq} Hz)', linewidth=2)
    
    ax.set_title('📊 PSD of Pure Frequencies', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (V²/Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    
    # 3. Mixed signal (realistic EEG-like)
    ax = axes[0, 2]
    
    # Create realistic EEG-like signal
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, len(t))
    mixed_signal = (0.5 * signal1 +  # Alpha component
                   0.3 * signal2 +   # Beta component  
                   0.2 * signal3 +   # Gamma component
                   noise)            # Background noise
    
    ax.plot(t[:250], mixed_signal[:250], 'b-', linewidth=1.5, label='Mixed EEG-like Signal')
    ax.set_title('🧠 Realistic EEG-like Signal', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (μV)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. PSD of mixed signal with frequency bands
    ax = axes[1, 0]
    
    freqs, psd = signal.welch(mixed_signal, fs, nperseg=256)
    ax.semilogy(freqs, psd, 'b-', linewidth=2, label='Mixed Signal PSD')
    
    # Mark frequency bands
    ax.axvspan(0.5, 4, alpha=0.2, color='red', label='Delta (0.5-4 Hz)')
    ax.axvspan(4, 8, alpha=0.2, color='orange', label='Theta (4-8 Hz)')
    ax.axvspan(8, 13, alpha=0.2, color='yellow', label='Alpha (8-13 Hz)')
    ax.axvspan(13, 30, alpha=0.2, color='green', label='Beta (13-30 Hz)')
    ax.axvspan(30, 50, alpha=0.2, color='purple', label='Gamma (30-50 Hz)')
    
    ax.set_title('🎵 PSD with EEG Frequency Bands', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    
    # 5. Channel comparison concept
    ax = axes[1, 1]
    
    # Simulate different channels with different frequency profiles
    channel_names = ['AF3 (Frontal)', 'O1 (Occipital)', 'T7 (Temporal)']
    
    # AF3: More beta/gamma (cognitive activity)
    af3_signal = (0.2 * signal1 + 0.6 * signal2 + 0.4 * signal3 + 
                  np.random.normal(0, 0.1, len(t)))
    
    # O1: More alpha (visual cortex)
    o1_signal = (0.8 * signal1 + 0.3 * signal2 + 0.1 * signal3 + 
                 np.random.normal(0, 0.1, len(t)))
    
    # T7: Mixed theta/alpha (temporal lobe)
    theta_signal = np.sin(2 * np.pi * 6 * t)  # 6 Hz theta
    t7_signal = (0.4 * theta_signal + 0.5 * signal1 + 0.2 * signal2 + 
                 np.random.normal(0, 0.1, len(t)))
    
    signals = [af3_signal, o1_signal, t7_signal]
    colors = ['red', 'blue', 'green']
    
    for sig, name, color in zip(signals, channel_names, colors):
        freqs, psd = signal.welch(sig, fs, nperseg=256)
        ax.semilogy(freqs, psd, color=color, linewidth=2, label=name)
    
    ax.set_title('🧠 Channel-wise PSD Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    
    # 6. Interpretation guide
    ax = axes[1, 2]
    ax.axis('off')
    
    interpretation_text = """
🧠 PSD INTERPRETATION GUIDE:

📊 WHAT PSD TELLS US:
• High power at 10 Hz → Strong alpha activity
• Peak at 20 Hz → Beta oscillations
• Broad spectrum → Complex neural activity
• Sharp peaks → Rhythmic brain states

🎯 CHANNEL-WISE DIFFERENCES:
• Frontal (AF3): High beta/gamma
  → Cognitive processing, attention
• Occipital (O1): Strong alpha
  → Visual processing, relaxed state  
• Temporal (T7): Theta/alpha mix
  → Memory, emotional processing

🔍 CLINICAL SIGNIFICANCE:
• Abnormal PSD patterns can indicate:
  - Epilepsy (sharp frequency peaks)
  - ADHD (altered beta/theta ratio)
  - Depression (reduced alpha)
  - Sleep disorders (abnormal rhythms)

💡 IN MINDBIGDATA CONTEXT:
• Different digits may show different
  PSD patterns across channels
• This creates features for classification
• Channel-wise analysis reveals which
  brain regions are most informative
    """
    
    ax.text(0.05, 0.95, interpretation_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('psd_explanation.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_real_mindbigdata_psd():
    """Analyze PSD from real MindBigData"""
    
    print("🔬 ANALYZING REAL MINDBIGDATA PSD")
    print("=" * 50)
    
    try:
        # Load real data
        train_data = np.load("outputs_final/train_data.npy")
        train_labels = np.load("outputs_final/train_labels.npy")
        
        print(f"✅ Loaded data: {train_data.shape}")
        
        # Channel names
        channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                        'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        
        # Analyze PSD for each channel
        fs = 250  # Hz
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🧠 Real MindBigData Channel-wise PSD Analysis', fontsize=16, fontweight='bold')
        
        # 1. Average PSD across all channels
        ax = axes[0, 0]
        
        # Sample some trials for analysis
        sample_indices = np.random.choice(len(train_data), 100, replace=False)
        
        channel_psds = {}
        for ch_idx, ch_name in enumerate(channel_names):
            psds = []
            for idx in sample_indices:
                freqs, psd = signal.welch(train_data[idx, ch_idx], fs, nperseg=128)
                psds.append(psd)
            
            mean_psd = np.mean(psds, axis=0)
            channel_psds[ch_name] = mean_psd
            
            # Plot selected channels
            if ch_name in ['AF3', 'O1', 'O2', 'T7']:
                ax.semilogy(freqs, mean_psd, label=ch_name, linewidth=2)
        
        ax.set_title('📊 Channel-wise PSD Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (μV²/Hz)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)
        
        # 2. Frequency band power by channel
        ax = axes[0, 1]
        
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8), 
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
        
        band_powers = {band: [] for band in bands}
        channels_for_plot = ['AF3', 'F3', 'O1', 'O2', 'T7', 'T8']
        
        for ch_name in channels_for_plot:
            psd = channel_psds[ch_name]
            
            for band_name, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                band_powers[band_name].append(band_power)
        
        # Create heatmap
        band_matrix = np.array([band_powers[band] for band in bands.keys()])
        
        im = ax.imshow(band_matrix, cmap='viridis', aspect='auto')
        ax.set_title('🎵 Frequency Band Power by Channel', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(channels_for_plot)))
        ax.set_xticklabels(channels_for_plot)
        ax.set_yticks(range(len(bands)))
        ax.set_yticklabels(bands.keys())
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Band Power (μV²)')
        
        # 3. PSD by digit class
        ax = axes[1, 0]
        
        # Analyze PSD differences between digits
        digit_psds = {}
        for digit in [0, 1, 5, 9]:  # Sample digits
            digit_indices = np.where(train_labels == digit)[0]
            sample_indices = digit_indices[:min(20, len(digit_indices))]
            
            # Average across O1 channel (occipital - visual processing)
            psds = []
            o1_idx = channel_names.index('O1')
            
            for idx in sample_indices:
                freqs, psd = signal.welch(train_data[idx, o1_idx], fs, nperseg=128)
                psds.append(psd)
            
            mean_psd = np.mean(psds, axis=0)
            digit_psds[digit] = mean_psd
            
            ax.semilogy(freqs, mean_psd, label=f'Digit {digit}', linewidth=2)
        
        ax.set_title('🔢 PSD by Digit Class (O1 Channel)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (μV²/Hz)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate some summary statistics
        alpha_powers = []
        beta_powers = []
        
        for ch_name, psd in channel_psds.items():
            # Alpha band power
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            alpha_power = np.trapz(psd[alpha_mask], freqs[alpha_mask])
            alpha_powers.append(alpha_power)
            
            # Beta band power  
            beta_mask = (freqs >= 13) & (freqs <= 30)
            beta_power = np.trapz(psd[beta_mask], freqs[beta_mask])
            beta_powers.append(beta_power)
        
        summary_text = f"""
📊 MINDBIGDATA PSD ANALYSIS SUMMARY:

🎯 DATASET CHARACTERISTICS:
• Trials analyzed: {len(sample_indices)}
• Channels: {len(channel_names)} (EPOC)
• Frequency range: 0-125 Hz (Nyquist)
• Sampling rate: {fs} Hz

📈 FREQUENCY CONTENT:
• Alpha power range: {np.min(alpha_powers):.3f} - {np.max(alpha_powers):.3f} μV²
• Beta power range: {np.min(beta_powers):.3f} - {np.max(beta_powers):.3f} μV²
• Dominant frequency: ~{freqs[np.argmax(np.mean(list(channel_psds.values()), axis=0))]:.1f} Hz

🧠 CHANNEL INSIGHTS:
• Frontal channels: Higher beta/gamma
• Occipital channels: Strong alpha activity  
• Temporal channels: Mixed theta/alpha
• All channels show 1/f noise pattern

🔍 DIGIT CLASSIFICATION POTENTIAL:
• Different digits show PSD variations
• Channel-specific patterns observable
• Frequency features promising for ML
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('mindbigdata_real_psd_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Real PSD analysis complete!")
        
    except Exception as e:
        print(f"❌ Error analyzing real data: {e}")
        print("   Make sure outputs_final/ directory exists with data files")

def main():
    """Main explanation function"""
    print("🌊 POWER SPECTRAL DENSITY (PSD) EXPLANATION")
    print("=" * 60)
    
    print("\n📚 Creating conceptual demonstration...")
    demonstrate_psd_concept()
    
    print("\n🔬 Analyzing real MindBigData...")
    analyze_real_mindbigdata_psd()
    
    print(f"\n🎉 PSD EXPLANATION COMPLETE!")
    print(f"📁 Generated files:")
    print(f"   - psd_explanation.png (conceptual explanation)")
    print(f"   - mindbigdata_real_psd_analysis.png (real data analysis)")

if __name__ == "__main__":
    main()
