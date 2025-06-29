import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_eeg_data(file_path):
    """
    Parse EEG data from the text file
    Returns a pandas DataFrame with columns: id, event, device, channel, code, size, data
    """
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Split by tab
            parts = line.split('\t')
            if len(parts) >= 6:
                try:
                    id_val = int(parts[0])
                    event = int(parts[1])
                    device = parts[2]
                    channel = parts[3]
                    code = int(parts[4])
                    size = int(parts[5])
                    
                    # Parse the data values (comma-separated)
                    if len(parts) > 6:
                        data_str = parts[6]
                        # Convert comma-separated values to list of floats
                        data_values = [float(x) for x in data_str.split(',') if x.strip()]
                    else:
                        data_values = []
                    
                    data.append({
                        'id': id_val,
                        'event': event,
                        'device': device,
                        'channel': channel,
                        'code': code,
                        'size': size,
                        'data': data_values
                    })
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line[:100]}... Error: {e}")
                    continue
    
    return pd.DataFrame(data)

def plot_individual_codes(df, channels=['O1', 'O2'], max_signals_per_code=50):
    """
    Create individual plots for each stimulation code
    """
    # Filter data for the specified channels
    df_filtered = df[df['channel'].isin(channels)]
    
    # Get unique codes (0-9)
    codes = [i for i in range(10) if i in df_filtered['code'].unique()]
    print(f"Creating individual plots for codes: {codes}")
    
    for code in codes:
        # Create a figure for this code
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'EEG Signals for Stimulation Code {code}', fontsize=16)
        
        # Get data for this code
        code_data = df_filtered[df_filtered['code'] == code]
        
        for i, channel in enumerate(channels):
            channel_data = code_data[code_data['channel'] == channel]
            
            if not channel_data.empty:
                # Sample a subset of signals for better visualization
                if len(channel_data) > max_signals_per_code:
                    channel_data = channel_data.sample(n=max_signals_per_code, random_state=42)
                
                # Plot sampled signals for this code and channel
                for idx, row_data in channel_data.iterrows():
                    signal = row_data['data']
                    if signal:  # Check if signal is not empty
                        time_axis = np.arange(len(signal))
                        axes[i].plot(time_axis, signal, alpha=0.6, linewidth=0.5)
            
            # Set titles and labels
            total_signals = len(code_data[code_data["channel"] == channel])
            shown_signals = min(total_signals, max_signals_per_code)
            
            axes[i].set_title(f'Channel {channel}\n({shown_signals}/{total_signals} signals)')
            axes[i].set_xlabel('Time (samples)')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'eeg_code_{code}_signals.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot for code {code}")

def create_summary_statistics(df, channels=['O1', 'O2']):
    """
    Create summary statistics for each code
    """
    df_filtered = df[df['channel'].isin(channels)]
    
    print("\n=== SUMMARY STATISTICS ===")
    print("Signal count and average amplitude by code:")
    print("-" * 60)
    
    for code in sorted(df_filtered['code'].unique()):
        if code >= 0:  # Only show codes 0-9
            code_data = df_filtered[df_filtered['code'] == code]
            
            for channel in channels:
                channel_data = code_data[code_data['channel'] == channel]
                
                if not channel_data.empty:
                    # Calculate statistics
                    signal_count = len(channel_data)
                    
                    # Calculate average amplitude across all signals
                    all_amplitudes = []
                    for _, row in channel_data.iterrows():
                        if row['data']:
                            all_amplitudes.extend(row['data'])
                    
                    if all_amplitudes:
                        avg_amplitude = np.mean(all_amplitudes)
                        std_amplitude = np.std(all_amplitudes)
                        min_amplitude = np.min(all_amplitudes)
                        max_amplitude = np.max(all_amplitudes)
                        
                        print(f"Code {code} - {channel}: {signal_count:4d} signals | "
                              f"Avg: {avg_amplitude:7.1f} | Std: {std_amplitude:6.1f} | "
                              f"Range: [{min_amplitude:7.1f}, {max_amplitude:7.1f}]")

def main():
    # Path to the dataset file
    data_file = Path('../dataset/datasets/EP1.01.txt')
    
    if not data_file.exists():
        print(f"Error: File {data_file} not found!")
        return
    
    print("Loading and parsing EEG data...")
    df = parse_eeg_data(data_file)
    
    print(f"Loaded {len(df)} records")
    
    # Filter for channels O1 and O2
    o1_o2_data = df[df['channel'].isin(['O1', 'O2'])]
    print(f"Data for O1 and O2 channels: {len(o1_o2_data)} records")
    
    # Create individual plots for each code
    print("\nCreating individual plots for each stimulation code...")
    plot_individual_codes(df, channels=['O1', 'O2'], max_signals_per_code=50)
    
    # Create summary statistics
    create_summary_statistics(df, channels=['O1', 'O2'])
    
    print("\nAll plots saved successfully!")
    print("Files created:")
    print("- eeg_code_0_signals.png through eeg_code_9_signals.png")

if __name__ == "__main__":
    main()
