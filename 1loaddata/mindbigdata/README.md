# 🧠 MindBigData EEG Dataset Processing

This directory contains the processing pipeline for the MindBigData EEG dataset, adapted from the Crell dataset preprocessing with **CORRECT preprocessing order**.

## 📊 Dataset Overview

### **MindBigData EEG Dataset**
- **File**: `EP1.01.txt` (tab-separated text format)
- **Stimuli**: `MindbigdataStimuli/` folder (digits 0-9)
- **Type**: Standardized multi-channel EEG trials for digit recognition
- **Content**: EEG recordings while viewing digits 0-9
- **Total Signals**: 1.2+ million signals from EPOC device
- **Standardized Format**: **(n_trials, 14, 128)** - Ready for UltraHighDimExtractor
- **Trial Duration**: 1 second at 128 Hz
- **Channels**: 14 EPOC channels (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
- **Target**: 28x28 images of digits 0-9
- **Raw Format**: `[id][event][device][channel][code][size][data]`

## 🔧 CORRECT Preprocessing Pipeline

### **Processing Order** (CRITICAL!)
1. **Bandpass filtering (0.5-50 Hz)** - Applied to RAW data FIRST
2. **Artifact detection and removal** - Conservative thresholds (6 std)
3. **Baseline correction** - Subtract first 20% mean
4. **Z-score normalization** - FINAL step (mean=0, std=1)

### **Key Differences from Crell**
- **14 EPOC channels** instead of 64 channels
- **128 timepoints (1 second)** instead of 500 timepoints
- **128 Hz sampling rate** instead of 500 Hz
- **Digit labels (0-9)** instead of letter codes
- **Tab-separated text format** instead of .mat file
- **Event-based grouping** for multi-channel trials
- **Standardized format (n_trials, 14, 128)** for UltraHighDimExtractor

## 📁 Files

### **Main Processing Script**
- `1process_mindbigdata_data.py` - Complete preprocessing pipeline

### **Functions**
- `load_mindbigdata_data()` - Load and parse EP1.01.txt with multi-channel grouping
- `load_stimuli()` - Load digit images (0-9)
- `process_eeg_signals()` - Apply CORRECT preprocessing to multi-channel trials
- `create_data_splits()` - Train/validation/test splits
- `visualize_sample_epochs()` - Create sample visualizations with topography
- `prepare_for_ultrahighdim()` - Convert to UltraHighDimExtractor format

## 🚀 Usage

```bash
cd 1loaddata/mindbigdata
python 1process_mindbigdata_data.py
```

## 📈 Output Files

- `mindbigdata_processed_data_correct.pkl` - Standard format (n_trials, 14, 128)
- `mindbigdata_ultrahighdim_ready.pkl` - UltraHighDimExtractor format (n_trials, 1792)
- `mindbigdata_sample_epochs.png` - Sample EEG epochs and digit stimuli with topography

## 📋 Technical Specifications

- **EEG Shape**: **(n_trials, 14, 128)** - Standardized format
- **Image Shape**: (n_trials, 28, 28)
- **Sampling Rate**: 128 Hz
- **Trial Duration**: 1.0 second (128 timepoints)
- **Channels**: 14 EPOC channels (standardized order)
- **Preprocessing**: CORRECT pipeline applied
- **Data Quality**: Conservative artifact rejection
- **Event Grouping**: Multi-channel trials from same event_id
- **UltraHighDimExtractor**: Ready (flattened: 14×128 = 1792 features)
- **Format Consistency**: Compatible with `np.array([100, 14, 128])` example

## 🚀 UltraHighDimExtractor Integration

### **Automatic Format Conversion**
The preprocessing pipeline automatically generates two formats:

1. **Standard Format**: `(n_trials, 14, 128)`
   - Preserves spatial-temporal structure
   - Suitable for CNN/RNN models
   - File: `mindbigdata_processed_data_correct.pkl`

2. **UltraHighDimExtractor Format**: `(n_trials, 1792)`
   - Flattened: 14 channels × 128 timepoints = 1792 features
   - Ready for sliding window processing (128 features per window)
   - File: `mindbigdata_ultrahighdim_ready.pkl`

### **Usage Example**
```python
# Load UltraHighDimExtractor ready data
with open('mindbigdata_ultrahighdim_ready.pkl', 'rb') as f:
    data = pickle.load(f)

# Data format: (n_trials, 1792)
X_train = data['training']['eeg']  # Shape: (n_trials, 1792)
y_train = data['training']['labels']

# Ready for UltraHighDimExtractor with 128-feature windows
# Windows per trial: 1792 // 128 = 14 windows
```

## 🔗 Related Files

- **Crell Processing**: `../crell/` (letter recognition)
- **Original Preprocessing**: Based on Crell's correct pipeline
- **Dataset Source**: `../../dataset/datasets/EP1.01.txt`

---

**Status**: ✅ Ready for EEG-to-Digit modeling with CORRECT preprocessing and UltraHighDimExtractor compatibility
