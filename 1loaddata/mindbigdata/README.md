# 🧠 MindBigData EEG Dataset Processing

This directory contains the processing pipeline for the MindBigData EEG dataset, adapted from the Crell dataset preprocessing with **CORRECT preprocessing order**.

## 📊 Dataset Overview

### **MindBigData EEG Dataset**
- **File**: `EP1.01.txt` (tab-separated text format)
- **Stimuli**: `MindbigdataStimuli/` folder (digits 0-9)
- **Type**: Multi-channel EEG signals for digit recognition
- **Content**: EEG recordings while viewing digits 0-9
- **Total Signals**: 1.2+ million signals from EPOC device
- **Signal Length**: ~260 timepoints (2 seconds at 128 Hz)
- **Channels**: 14 channels (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
- **Target**: 28x28 images of digits 0-9
- **Format**: `[id][event][device][channel][code][size][data]`

## 🔧 CORRECT Preprocessing Pipeline

### **Processing Order** (CRITICAL!)
1. **Bandpass filtering (0.5-50 Hz)** - Applied to RAW data FIRST
2. **Artifact detection and removal** - Conservative thresholds (6 std)
3. **Baseline correction** - Subtract first 20% mean
4. **Z-score normalization** - FINAL step (mean=0, std=1)

### **Key Differences from Crell**
- **14 EPOC channels** instead of 64 channels
- **~260 timepoints** instead of 500
- **128 Hz sampling rate** instead of 500 Hz
- **Digit labels (0-9)** instead of letter codes
- **Tab-separated text format** instead of .mat file
- **Event-based grouping** for multi-channel epochs

## 📁 Files

### **Main Processing Script**
- `1process_mindbigdata_data.py` - Complete preprocessing pipeline

### **Functions**
- `load_mindbigdata_data()` - Load and parse EP1.01.txt with multi-channel grouping
- `load_stimuli()` - Load digit images (0-9)
- `process_eeg_signals()` - Apply CORRECT preprocessing to multi-channel epochs
- `create_data_splits()` - Train/validation/test splits
- `visualize_sample_epochs()` - Create sample visualizations with topography

## 🚀 Usage

```bash
cd 1loaddata/mindbigdata
python 1process_mindbigdata_data.py
```

## 📈 Output Files

- `mindbigdata_processed_data_correct.pkl` - Processed dataset with CORRECT preprocessing
- `mindbigdata_sample_epochs.png` - Sample EEG epochs and digit stimuli with topography

## 📋 Technical Specifications

- **EEG Shape**: (n_epochs, 14, ~260)
- **Image Shape**: (n_epochs, 28, 28)
- **Sampling Rate**: 128 Hz
- **Signal Length**: ~260 timepoints (2 seconds)
- **Channels**: 14 EPOC channels
- **Preprocessing**: CORRECT pipeline applied
- **Data Quality**: Conservative artifact rejection
- **Event Grouping**: Multi-channel epochs from same event_id

## 🔗 Related Files

- **Crell Processing**: `../crell/` (letter recognition)
- **Original Preprocessing**: Based on Crell's correct pipeline
- **Dataset Source**: `../../dataset/datasets/EP1.01.txt`

---

**Status**: ✅ Ready for EEG-to-Digit modeling with CORRECT preprocessing applied
