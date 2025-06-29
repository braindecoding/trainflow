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

def plot_signals_by_code(df, channels=['O1', 'O2'], max_signals_per_code=50):
    """
    Plot EEG signals grouped by stimulation code for specified channels
    """
    # Filter data for the specified channels
    df_filtered = df[df['channel'].isin(channels)]

    # Get unique codes (should be 0-9 and possibly -1 for random signals)
    codes = sorted(df_filtered['code'].unique())
    print(f"Found stimulation codes: {codes}")

    # Create subplots - 2 rows (for O1 and O2), 5 columns (for codes 0-4 and 5-9)
    fig, axes = plt.subplots(2, 5, figsize=(20, 12))
    fig.suptitle('EEG Signals by Stimulation Code (Channels O1 and O2)', fontsize=16)

    # Plot for each code (0-9)
    for i, code in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        if code in codes:
            col = i % 5

            # Get data for this code and both channels
            code_data = df_filtered[df_filtered['code'] == code]

            for channel in channels:
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
                            if channel == 'O1':
                                axes[0, col].plot(time_axis, signal, alpha=0.6, linewidth=0.5)
                            else:  # O2
                                axes[1, col].plot(time_axis, signal, alpha=0.6, linewidth=0.5)

            # Set titles and labels
            total_o1 = len(code_data[code_data["channel"] == "O1"])
            total_o2 = len(code_data[code_data["channel"] == "O2"])
            shown_o1 = min(total_o1, max_signals_per_code)
            shown_o2 = min(total_o2, max_signals_per_code)

            axes[0, col].set_title(f'Code {code} - O1\n({shown_o1}/{total_o1} signals)')
            axes[1, col].set_title(f'Code {code} - O2\n({shown_o2}/{total_o2} signals)')

            axes[0, col].set_xlabel('Time (samples)')
            axes[1, col].set_xlabel('Time (samples)')
            axes[0, col].set_ylabel('Amplitude')
            axes[1, col].set_ylabel('Amplitude')
            axes[0, col].grid(True, alpha=0.3)
            axes[1, col].grid(True, alpha=0.3)
        else:
            # If code not found, hide the subplot
            col = i % 5
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)

    plt.tight_layout()
    plt.savefig('eeg_signals_by_code.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def main():
    # Path to the dataset file
    data_file = Path('../dataset/datasets/EP1.01.txt')
    
    if not data_file.exists():
        print(f"Error: File {data_file} not found!")
        return
    
    print("Loading and parsing EEG data...")
    df = parse_eeg_data(data_file)
    
    print(f"Loaded {len(df)} records")
    print(f"Devices found: {df['device'].unique()}")
    print(f"Channels found: {df['channel'].unique()}")
    print(f"Codes found: {sorted(df['code'].unique())}")
    
    # Filter for channels O1 and O2
    o1_o2_data = df[df['channel'].isin(['O1', 'O2'])]
    print(f"\nData for O1 and O2 channels: {len(o1_o2_data)} records")
    
    # Show count by code for O1 and O2
    print("\nSignal count by code:")
    for code in sorted(o1_o2_data['code'].unique()):
        o1_count = len(o1_o2_data[(o1_o2_data['code'] == code) & (o1_o2_data['channel'] == 'O1')])
        o2_count = len(o1_o2_data[(o1_o2_data['code'] == code) & (o1_o2_data['channel'] == 'O2')])
        print(f"Code {code}: O1={o1_count}, O2={o2_count}")
    
    # Create the plots
    print("\nCreating plots (sampling 50 signals per code for better visualization)...")
    plot_signals_by_code(df, channels=['O1', 'O2'], max_signals_per_code=50)

    print("Plots saved as 'eeg_signals_by_code.png'")

if __name__ == "__main__":
    main()
