#!/usr/bin/env python3
"""
Process MindBigData Dataset for EEG-to-Digit Reconstruction
Extract EEG signals aligned with digit presentation
Create train/validation/test splits with CORRECT preprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")

def step1_bandpass_filter(raw_signal, fs=128, lowcut=0.5, highcut=50, order=4):
    """
    STEP 1: Apply bandpass filter to RAW EEG signal
    This MUST be done FIRST on raw data before any other processing
    """
    # Design Butterworth bandpass filter
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    # Create second-order sections for stability
    sos = signal.butter(order, [low, high], btype='band', output='sos')

    # Apply zero-phase filtering (forward and backward)
    filtered_signal = signal.sosfiltfilt(sos, raw_signal)

    return filtered_signal

def step2_artifact_detection(filtered_signal, fs=128):
    """
    STEP 2: Detect artifacts in filtered signal
    Apply AFTER filtering but BEFORE epoching
    """
    artifacts = {
        'amplitude': False,
        'gradient': False,
        'flatline': False,
        'is_artifact': False,
        'artifact_samples': []
    }

    # Use adaptive thresholds based on signal statistics
    signal_std = np.std(filtered_signal)
    signal_mean = np.mean(np.abs(filtered_signal))

    # 1. Amplitude artifacts (signal too large)
    # Use 6 standard deviations as threshold (very conservative)
    amplitude_threshold = signal_mean + 6 * signal_std
    amplitude_mask = np.abs(filtered_signal) > amplitude_threshold

    if np.any(amplitude_mask):
        artifacts['amplitude'] = True
        artifacts['artifact_samples'].extend(np.where(amplitude_mask)[0].tolist())

    # 2. Gradient artifacts (sudden jumps)
    gradient = np.diff(filtered_signal)
    gradient_std = np.std(gradient)
    gradient_threshold = 6 * gradient_std  # 6 standard deviations
    gradient_mask = np.abs(gradient) > gradient_threshold

    if np.any(gradient_mask):
        artifacts['gradient'] = True
        artifacts['artifact_samples'].extend((np.where(gradient_mask)[0] + 1).tolist())

    # 3. Flatline detection (signal too constant)
    if signal_std < 0.001:  # Completely flat signal
        artifacts['flatline'] = True

    # Overall artifact decision - be conservative
    # Only reject if multiple severe artifacts OR completely flat
    severe_amplitude = np.sum(amplitude_mask) > len(filtered_signal) * 0.1  # >10% of samples
    severe_gradient = np.sum(gradient_mask) > len(gradient) * 0.1  # >10% of samples

    artifacts['is_artifact'] = (severe_amplitude and severe_gradient) or artifacts['flatline']
    artifacts['artifact_samples'] = list(set(artifacts['artifact_samples']))

    return artifacts

def step4_baseline_correction(epoch, fs=128, baseline_duration=0.2):
    """
    STEP 4: Apply baseline correction to epoch
    Apply AFTER epoching but BEFORE normalization
    """
    baseline_samples = int(baseline_duration * fs)

    if len(epoch) < baseline_samples:
        # If epoch too short, use first 20% as baseline
        baseline_samples = max(1, len(epoch) // 5)

    # Calculate baseline (mean of pre-stimulus period)
    baseline = np.mean(epoch[:baseline_samples])

    # Subtract baseline from entire epoch
    corrected_epoch = epoch - baseline

    return corrected_epoch

def step5_normalization(baseline_corrected_signal, method='zscore'):
    """
    STEP 5: Apply normalization (FINAL STEP)
    Apply AFTER all other preprocessing steps
    """
    if method == 'zscore':
        # Standard z-score normalization
        normalized = zscore(baseline_corrected_signal)
    elif method == 'robust':
        # Robust normalization using median and MAD
        median = np.median(baseline_corrected_signal)
        mad = np.median(np.abs(baseline_corrected_signal - median))
        if mad > 0:
            normalized = (baseline_corrected_signal - median) / mad
        else:
            normalized = baseline_corrected_signal - median
    elif method == 'minmax':
        # Min-max normalization to [-1, 1]
        sig_min, sig_max = np.min(baseline_corrected_signal), np.max(baseline_corrected_signal)
        if sig_max > sig_min:
            normalized = 2 * (baseline_corrected_signal - sig_min) / (sig_max - sig_min) - 1
        else:
            normalized = np.zeros_like(baseline_corrected_signal)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized

def correct_preprocessing_pipeline(raw_signal, fs=128, target_length=128):
    """
    Complete CORRECT preprocessing pipeline for single EEG signal
    Apply steps in the CORRECT ORDER:
    1. Bandpass filtering (on raw data)
    2. Artifact detection
    3. Epoching (already done)
    4. Baseline correction
    5. Normalization
    """
    try:
        # STEP 1: Bandpass filtering (FIRST - on raw data)
        filtered_signal = step1_bandpass_filter(raw_signal, fs=fs)

        # STEP 2: Artifact detection (on filtered data)
        artifacts = step2_artifact_detection(filtered_signal, fs=fs)
        if artifacts['is_artifact']:
            return None

        # STEP 3: Epoching (already done in extract_eeg_epochs)
        # Just ensure correct length
        if len(filtered_signal) != target_length:
            if len(filtered_signal) > target_length:
                # Extract from center
                start_idx = (len(filtered_signal) - target_length) // 2
                epoched_signal = filtered_signal[start_idx:start_idx + target_length]
            else:
                # Pad if too short
                pad_width = target_length - len(filtered_signal)
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
                epoched_signal = np.pad(filtered_signal, (pad_left, pad_right), mode='constant')
        else:
            epoched_signal = filtered_signal

        # STEP 4: Baseline correction
        baseline_corrected = step4_baseline_correction(epoched_signal, fs=fs)

        # STEP 5: Normalization (FINAL STEP)
        normalized_signal = step5_normalization(baseline_corrected, method='zscore')

        return normalized_signal

    except Exception as e:
        return None

def load_mindbigdata_data():
    """Load and parse MindBigData dataset"""
    print("📂 Loading MindBigData dataset...")

    # Load EEG data from EP1.01.txt (EPOC device data)
    data_file = '../../dataset/datasets/EP1.01.txt'

    print(f"   Reading data from: {data_file}")
    print("   Expected format: [id][event][device][channel][code][size][data]")

    # Read the text file with error handling
    try:
        with open(data_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"   ❌ Error: File {data_file} not found!")
        print(f"   Please ensure the MindBigData EP1.01.txt file is in the correct location.")
        return None

    # Parse the data according to MindBigData format
    eeg_signals = []
    labels = []
    channels = []
    devices = []
    event_ids = []

    print(f"   Parsing {len(lines)} lines...")

    for i, line in enumerate(lines):
        line = line.strip()
        if line and not line.startswith('#'):  # Skip empty lines and comments
            try:
                # Split by tab (MindBigData format)
                parts = line.split('\t')

                if len(parts) >= 6:  # [id][event][device][channel][code][size][data]
                    signal_id = int(parts[0])
                    event_id = int(parts[1])
                    device = parts[2].strip()
                    channel = parts[3].strip()
                    code = int(parts[4])  # Digit label (0-9) or -1
                    size = int(parts[5])
                    data_str = parts[6]

                    # Skip random signals (code = -1) for now
                    if code == -1:
                        continue

                    # Parse signal data (comma-separated)
                    signal_values = [float(x.strip()) for x in data_str.split(',') if x.strip()]

                    if len(signal_values) > 0:  # Valid signal
                        eeg_signals.append(signal_values)
                        labels.append(code)
                        channels.append(channel)
                        devices.append(device)
                        event_ids.append(event_id)

                        if (i + 1) % 1000 == 0:
                            print(f"      Parsed {i + 1} lines, found {len(eeg_signals)} valid signals...")

            except (ValueError, IndexError) as e:
                print(f"   ⚠️ Warning: Skipping line {i+1} due to parsing error: {e}")
                continue

    if len(eeg_signals) == 0:
        print(f"   ❌ Error: No valid EEG signals found in the file!")
        return None

    print(f"   Found {len(eeg_signals)} valid signals")
    print(f"   Devices: {set(devices)}")
    print(f"   Channels: {set(channels)}")
    print(f"   Event IDs range: {min(event_ids)} to {max(event_ids)}")

    # Group signals by event_id to create multi-channel epochs
    print("   Grouping signals by event_id to create multi-channel epochs...")

    events_data = {}
    for i, (signal, label, channel, device, event_id) in enumerate(zip(eeg_signals, labels, channels, devices, event_ids)):
        if event_id not in events_data:
            events_data[event_id] = {
                'signals': {},
                'label': label,
                'device': device
            }
        events_data[event_id]['signals'][channel] = signal

    # Filter events that have multiple channels and same label
    print("   Filtering complete multi-channel events...")
    complete_events = []

    for event_id, event_data in events_data.items():
        signals = event_data['signals']
        if len(signals) > 1:  # Multi-channel event
            # Check if all signals have similar length
            signal_lengths = [len(sig) for sig in signals.values()]
            if len(set(signal_lengths)) <= 2:  # Allow small variation
                complete_events.append(event_data)

    print(f"   Found {len(complete_events)} complete multi-channel events")

    if len(complete_events) == 0:
        print(f"   ❌ Error: No complete multi-channel events found!")
        return None

    # Process complete events
    processed_epochs = []
    processed_labels = []

    # Standardize to EPOC 14 channels and 128 timepoints (1 second at 128 Hz)
    # EPOC channel order (standard 10-20 system)
    epoc_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    target_length = 128  # 1 second at 128 Hz
    target_channels = 14  # EPOC standard

    print(f"   Standardizing to: {target_channels} channels, {target_length} timepoints")
    print(f"   Target format: (n_trials, 14, 128)")
    print(f"   EPOC channel order: {epoc_channels}")

    for event in complete_events:
        signals = event['signals']
        label = event['label']

        # Create standardized 14-channel epoch
        epoch_channels = []
        for channel in epoc_channels:
            if channel in signals:
                signal = signals[channel]
                # Standardize length to 128 timepoints
                if len(signal) > target_length:
                    # Downsample or truncate to 128 points
                    start_idx = (len(signal) - target_length) // 2
                    signal = signal[start_idx:start_idx + target_length]
                elif len(signal) < target_length:
                    # Upsample or pad to 128 points
                    signal = signal + [0.0] * (target_length - len(signal))
                epoch_channels.append(signal)
            else:
                # Missing channel, fill with zeros
                epoch_channels.append([0.0] * target_length)

        # Ensure exactly 14 channels
        if len(epoch_channels) == target_channels:
            processed_epochs.append(epoch_channels)
            processed_labels.append(label)

    # Convert to numpy arrays
    epochs_array = np.array(processed_epochs)  # (n_epochs, n_channels, n_timepoints)
    labels_array = np.array(processed_labels)  # (n_epochs,)

    print(f"   Final epoch shape: {epochs_array.shape}")
    print(f"   Labels shape: {labels_array.shape}")
    print(f"   Label distribution: {dict(Counter(labels_array))}")

    return {
        'eeg_data': epochs_array,           # (n_trials, 14, 128)
        'labels': labels_array,             # (n_trials,)
        'n_samples': len(epochs_array),
        'n_timepoints': target_length,      # 128
        'n_channels': target_channels,      # 14
        'channel_order': epoc_channels      # EPOC channel names
    }

def load_stimuli():
    """Load digit stimulus images"""
    print(f"\n🖼️ Loading digit stimulus images...")

    stimuli_path = '../../dataset/datasets/MindbigdataStimuli'

    # Digit mapping (0-9)
    digit_mapping = {i: str(i) for i in range(10)}

    stimuli_data = {}

    for digit, digit_str in digit_mapping.items():
        filepath = os.path.join(stimuli_path, f'{digit}.jpg')

        # Load and process image
        img = Image.open(filepath)
        img_array = np.array(img)

        # Convert to grayscale if RGB
        if len(img_array.shape) == 3:
            img_gray = np.mean(img_array, axis=2)
        else:
            img_gray = img_array

        # Resize to 28x28 for consistency
        img_pil = Image.fromarray(img_gray.astype(np.uint8))
        img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        img_resized_array = np.array(img_resized)

        # Normalize to [0, 1]
        img_normalized = img_resized_array.astype(np.float32) / 255.0

        stimuli_data[digit] = {
            'digit': digit_str,
            'image': img_normalized,
            'original': img_array
        }

        print(f"   {digit_str} (digit {digit}): {img_normalized.shape}")

    return stimuli_data

def process_eeg_signals(mindbigdata_data, stimuli_data):
    """Process MindBigData EEG signals with CORRECT preprocessing to (n_trials, 14, 128) format"""
    print(f"\n🧠 Processing EEG signals with CORRECT preprocessing...")
    print("   Preprocessing order: Filter→Artifact→Baseline→Normalize")
    print("   Target format: (n_trials, 14, 128)")

    # Standardized parameters for UltraHighDimExtractor compatibility
    sampling_rate = 128  # EPOC sampling rate
    target_length = 128  # 1 second at 128 Hz
    target_channels = 14 # EPOC channels

    eeg_data = mindbigdata_data['eeg_data']  # (n_trials, 14, 128)
    labels = mindbigdata_data['labels']      # (n_trials,)
    n_channels = mindbigdata_data['n_channels']

    all_epochs = []
    all_labels = []
    all_images = []
    processed_count = 0
    rejected_count = 0

    print(f"   Processing {len(eeg_data)} standardized EEG trials...")
    print(f"   Format: {len(eeg_data)} trials × {n_channels} channels × {target_length} timepoints")

    for i, (eeg_epoch, label) in enumerate(zip(eeg_data, labels)):
        # Process each channel separately
        processed_channels = []
        channel_rejected = False

        for ch in range(n_channels):
            raw_signal = eeg_epoch[ch, :]  # (n_timepoints,)

            # Apply CORRECT preprocessing pipeline
            processed_signal = correct_preprocessing_pipeline(
                raw_signal, fs=sampling_rate, target_length=target_length
            )

            if processed_signal is not None:
                processed_channels.append(processed_signal)
            else:
                channel_rejected = True
                break

        # Only keep epoch if all channels processed successfully
        if not channel_rejected and len(processed_channels) == n_channels:
            # Stack channels: (n_channels, n_timepoints)
            epoch_processed = np.array(processed_channels)

            # Get corresponding stimulus image
            if label in stimuli_data:
                stimulus_image = stimuli_data[label]['image']

                all_epochs.append(epoch_processed)
                all_labels.append(label)
                all_images.append(stimulus_image)
                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"      Processed {processed_count} epochs...")
        else:
            rejected_count += 1

    # Convert to arrays
    epochs_array = np.array(all_epochs)  # (n_trials, 14, 128)
    labels_array = np.array(all_labels)  # (n_trials,)
    images_array = np.array(all_images)  # (n_trials, height, width)

    print(f"\n✅ CORRECT preprocessing completed:")
    print(f"   Total trials processed: {processed_count}")
    print(f"   Trials rejected: {rejected_count}")
    print(f"   Rejection rate: {rejected_count/(processed_count+rejected_count)*100:.1f}%")
    print(f"   Final format: {epochs_array.shape} (n_trials, 14, 128)")
    print(f"   Labels shape: {labels_array.shape}")
    print(f"   Images shape: {images_array.shape}")
    print(f"   Label distribution: {dict(Counter(labels_array))}")
    print(f"   ✅ Ready for UltraHighDimExtractor: (n_trials, 14*128) = (n_trials, 1792)")

    return epochs_array, labels_array, images_array

def create_digit_mapping(labels_array):
    """Create mapping from digit labels to indices"""
    unique_digits = sorted(np.unique(labels_array))

    digit_to_idx = {digit: idx for idx, digit in enumerate(unique_digits)}
    idx_to_digit = {idx: digit for idx, digit in enumerate(unique_digits)}

    # Digit names (0-9)
    digit_names = {i: str(i) for i in range(10)}

    idx_to_digit_name = {idx: digit_names[digit] for idx, digit in idx_to_digit.items()}

    print(f"\n📊 Digit mapping:")
    for idx, digit in idx_to_digit.items():
        digit_name = digit_names[digit]
        print(f"   {digit_name} (digit {digit}) -> index {idx}")

    return digit_to_idx, idx_to_digit, idx_to_digit_name

def create_data_splits(epochs_array, labels_array, images_array, test_size=0.2, val_size=0.2):
    """Create stratified train/validation/test splits"""
    print(f"\n📊 Creating data splits...")

    # Convert labels to indices for stratification
    digit_to_idx, idx_to_digit, idx_to_digit_name = create_digit_mapping(labels_array)
    labels_idx = np.array([digit_to_idx[digit] for digit in labels_array])

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test, img_temp, img_test = train_test_split(
        epochs_array, labels_idx, images_array,
        test_size=test_size, stratify=labels_idx, random_state=42
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    X_train, X_val, y_train, y_val, img_train, img_val = train_test_split(
        X_temp, y_temp, img_temp,
        test_size=val_size_adjusted, stratify=y_temp, random_state=42
    )

    print(f"   Training set: {X_train.shape[0]} trials")
    print(f"   Validation set: {X_val.shape[0]} trials")
    print(f"   Test set: {X_test.shape[0]} trials")

    # Check distribution
    for split_name, split_labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        dist = Counter(split_labels)
        print(f"   {split_name} distribution: {dict(dist)}")

    return {
        'training': {
            'eeg': X_train,
            'labels': y_train,
            'images': img_train
        },
        'validation': {
            'eeg': X_val,
            'labels': y_val,
            'images': img_val
        },
        'test': {
            'eeg': X_test,
            'labels': y_test,
            'images': img_test
        },
        'metadata': {
            'digit_to_idx': digit_to_idx,
            'idx_to_digit': idx_to_digit,
            'idx_to_digit_name': idx_to_digit_name,
            'n_channels': 14,                    # Standardized EPOC channels
            'n_timepoints': 128,                 # Standardized 1 second at 128 Hz
            'n_digits': len(digit_to_idx),
            'sampling_rate': 128,
            'trial_duration': 1.0,               # 1 second trials
            'baseline_duration': 0.2,
            'device': 'EPOC',
            'dataset': 'MindBigData',
            'format': '(n_trials, 14, 128)',     # Standard format for UltraHighDimExtractor
            'flattened_features': 14 * 128,     # 1792 features when flattened
            'ultrahighdim_ready': True
        }
    }

def visualize_sample_epochs(data_splits):
    """Visualize sample EEG epochs and corresponding digits"""
    print(f"\n🎨 Creating visualization...")

    # Get sample data
    sample_eeg = data_splits['training']['eeg'][:4]  # First 4 epochs
    sample_labels = data_splits['training']['labels'][:4]
    sample_images = data_splits['training']['images'][:4]

    idx_to_digit_name = data_splits['metadata']['idx_to_digit_name']
    n_channels = sample_eeg.shape[1]

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle('MindBigData Dataset: EEG Epochs and Digit Stimuli', fontsize=16, fontweight='bold')

    for i in range(4):
        # EEG epoch (show average across channels)
        eeg_avg = sample_eeg[i].mean(axis=0)  # Average across channels
        axes[i, 0].plot(eeg_avg)
        axes[i, 0].set_title(f'EEG Epoch {i+1} (Avg)\nDigit: {idx_to_digit_name[sample_labels[i]]}')
        axes[i, 0].set_xlabel('Time points')
        axes[i, 0].set_ylabel('Amplitude (μV)')
        axes[i, 0].grid(True, alpha=0.3)

        # EEG topography (show at peak time)
        peak_time = np.argmax(np.abs(eeg_avg))
        topo = sample_eeg[i][:, peak_time]

        # Create a simple grid layout for channels
        grid_size = int(np.ceil(np.sqrt(n_channels)))
        topo_grid = np.zeros((grid_size, grid_size))
        for ch in range(min(n_channels, grid_size * grid_size)):
            row, col = ch // grid_size, ch % grid_size
            topo_grid[row, col] = topo[ch] if ch < len(topo) else 0

        im = axes[i, 1].imshow(topo_grid, cmap='RdBu_r', aspect='auto')
        axes[i, 1].set_title(f'EEG Topography\n(t={peak_time})')
        axes[i, 1].axis('off')

        # Digit stimulus
        axes[i, 2].imshow(sample_images[i], cmap='gray')
        axes[i, 2].set_title(f'Digit Stimulus\n"{idx_to_digit_name[sample_labels[i]]}"')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('mindbigdata_sample_epochs.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Sample visualization saved as 'mindbigdata_sample_epochs.png'")

def main():
    """Main processing function"""
    print("🎯 MINDBIGDATA DATASET PROCESSING WITH CORRECT PREPROCESSING")
    print("=" * 90)
    print("📝 Processing EEG data for digit recognition task")
    print("🔧 CORRECT Preprocessing Order:")
    print("   1. Bandpass filtering (0.5-50 Hz) - on RAW data")
    print("   2. Artifact detection and removal")
    print("   3. Baseline correction")
    print("   4. Z-score normalization (FINAL step)")
    print("=" * 90)

    # Load raw data
    mindbigdata_data = load_mindbigdata_data()
    if mindbigdata_data is None:
        print("❌ Failed to load MindBigData. Exiting...")
        return

    stimuli_data = load_stimuli()

    # Process EEG signals
    epochs_array, labels_array, images_array = process_eeg_signals(mindbigdata_data, stimuli_data)

    # Create data splits
    data_splits = create_data_splits(epochs_array, labels_array, images_array)

    # Add preprocessing metadata
    data_splits['preprocessing_metadata'] = {
        'preprocessing_order': [
            '1. Bandpass filtering (0.5-50 Hz) - applied to RAW data',
            '2. Artifact detection and removal (6 std threshold)',
            '3. Baseline correction (subtract first 20% mean)',
            '4. Z-score normalization (final step)'
        ],
        'filter_range': '0.5-50 Hz',
        'filter_order': 4,
        'filter_type': 'Butterworth bandpass with zero-phase',
        'artifact_detection': 'Adaptive thresholds (6 std)',
        'baseline_correction': 'First 20% of signal',
        'normalization': 'Z-score per signal (mean=0, std=1)',
        'processing_order': 'CORRECT: Filter→Artifact→Baseline→Normalize'
    }

    # Save processed data (standard format)
    with open('mindbigdata_processed_data_correct.pkl', 'wb') as f:
        pickle.dump(data_splits, f)

    # Prepare and save UltraHighDimExtractor format
    ultrahighdim_data = prepare_for_ultrahighdim(data_splits)
    with open('mindbigdata_ultrahighdim_ready.pkl', 'wb') as f:
        pickle.dump(ultrahighdim_data, f)

    print(f"\n✅ CORRECT preprocessing completed!")
    print(f"   Standard format saved to: 'mindbigdata_processed_data_correct.pkl'")
    print(f"   UltraHighDimExtractor format saved to: 'mindbigdata_ultrahighdim_ready.pkl'")

    # Create visualization
    visualize_sample_epochs(data_splits)

    # Summary
    print(f"\n🎯 MINDBIGDATA DATASET SUMMARY:")
    print(f"   Task: EEG-to-Digit reconstruction")
    print(f"   Digits: {data_splits['metadata']['n_digits']} digits (0-9)")
    print(f"   Format: {data_splits['metadata']['format']}")
    print(f"   EEG channels: {data_splits['metadata']['n_channels']} (EPOC)")
    print(f"   Time points per trial: {data_splits['metadata']['n_timepoints']}")
    print(f"   Trial duration: {data_splits['metadata']['trial_duration']} second")
    print(f"   Sampling rate: {data_splits['metadata']['sampling_rate']} Hz")
    print(f"   Total trials: {sum(len(split['eeg']) for split in [data_splits['training'], data_splits['validation'], data_splits['test']])}")
    print(f"   Flattened features: {data_splits['metadata']['flattened_features']} (14×128)")
    print(f"   UltraHighDimExtractor ready: {data_splits['metadata']['ultrahighdim_ready']}")

    print(f"\n📁 Generated files:")
    print(f"   - mindbigdata_processed_data_correct.pkl (standard format: n_trials, 14, 128)")
    print(f"   - mindbigdata_ultrahighdim_ready.pkl (flattened format: n_trials, 1792)")
    print(f"   - mindbigdata_sample_epochs.png")

    print(f"\n🔧 CORRECT Preprocessing Applied:")
    print(f"   ✅ Step 1: Bandpass filtering (0.5-50 Hz) on RAW data")
    print(f"   ✅ Step 2: Artifact detection and removal")
    print(f"   ✅ Step 3: Baseline correction")
    print(f"   ✅ Step 4: Z-score normalization (FINAL step)")

    print(f"\n🚀 Ready for EEG-to-Digit modeling with CORRECT preprocessing!")

def prepare_for_ultrahighdim(data_splits):
    """
    Prepare processed data for UltraHighDimExtractor
    Convert from (n_trials, 14, 128) to (n_trials, 1792) format
    """
    print(f"\n🔧 Preparing data for UltraHighDimExtractor...")

    prepared_splits = {}

    for split_name, split_data in data_splits.items():
        if split_name == 'metadata' or split_name == 'preprocessing_metadata':
            prepared_splits[split_name] = split_data
            continue

        eeg_data = split_data['eeg']  # (n_trials, 14, 128)
        labels = split_data['labels']
        images = split_data['images']

        # Flatten EEG data: (n_trials, 14, 128) -> (n_trials, 1792)
        flattened_eeg = eeg_data.reshape(eeg_data.shape[0], -1)

        prepared_splits[split_name] = {
            'eeg': flattened_eeg,      # (n_trials, 1792)
            'labels': labels,          # (n_trials,)
            'images': images           # (n_trials, 28, 28)
        }

        print(f"   {split_name.capitalize()}: {eeg_data.shape} -> {flattened_eeg.shape}")

    # Update metadata
    prepared_splits['metadata']['flattened_shape'] = f"(n_trials, {14*128})"
    prepared_splits['metadata']['original_shape'] = "(n_trials, 14, 128)"

    print(f"   ✅ Ready for UltraHighDimExtractor with {14*128} features per trial")

    return prepared_splits


if __name__ == "__main__":
    main()
