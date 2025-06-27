# 🧠 EPOC Channel Configuration for MindBigData

## 📊 **EPOC 14-Channel Layout**

### **✅ Correct Channel Order:**
```python
# MindBigData uses Emotiv EPOC headset with 14 channels:
channels = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]
```

### **🗺️ Spatial Layout (10-20 System):**
```
        AF3     AF4
    F7   F3       F4   F8
FC5                     FC6
T7                       T8
P7                       P8
        O1       O2
```

### **🧠 Brain Region Coverage:**

#### **Frontal Regions:**
- **AF3, AF4**: Anterior frontal (attention, working memory)
- **F3, F4**: Frontal (executive functions, motor planning)
- **F7, F8**: Lateral frontal (language, spatial processing)

#### **Central Regions:**
- **FC5, FC6**: Frontal-central (motor control, sensorimotor)

#### **Temporal Regions:**
- **T7, T8**: Temporal (auditory processing, language)

#### **Parietal Regions:**
- **P7, P8**: Parietal (spatial attention, integration)

#### **Occipital Regions:**
- **O1, O2**: Occipital (visual processing) ⭐ **KEY for digit recognition**

## 🎯 **Relevance for Digit Recognition:**

### **Primary Visual Areas:**
```python
# Most important for digit/image processing:
visual_channels = ["O1", "O2"]  # Primary visual cortex
parietal_channels = ["P7", "P8"]  # Visual-spatial processing
```

### **Secondary Processing Areas:**
```python
# Supporting visual-cognitive processing:
attention_channels = ["AF3", "AF4"]  # Visual attention
integration_channels = ["FC5", "FC6"]  # Sensorimotor integration
```

## 🔧 **Technical Specifications:**

### **EPOC Hardware:**
```python
# Emotiv EPOC specifications:
- Channels: 14 + 2 reference (CMS/DRL)
- Sampling rate: 128 Hz (2048 Hz internal, downsampled)
- Resolution: 14-bit ADC (16384 levels)
- Bandwidth: 0.2-45 Hz
- Electrode type: Saline-based wet electrodes
```

### **Data Format:**
```python
# MindBigData format:
- Shape: (n_trials, 14_channels, n_timepoints)
- Channel order: Fixed as listed above
- Units: Microvolts (µV)
- Reference: Common average reference
```

## 📈 **Preprocessing Considerations:**

### **Channel-Specific Processing:**
```python
# Different regions may need different treatment:
visual_regions = ["O1", "O2", "P7", "P8"]  # Higher frequencies important
frontal_regions = ["AF3", "AF4", "F3", "F4", "F7", "F8"]  # Lower frequencies
temporal_regions = ["T7", "T8"]  # Mid-range frequencies
central_regions = ["FC5", "FC6"]  # Motor-related artifacts
```

### **Artifact Considerations:**
```python
# Common artifacts by region:
frontal_artifacts = ["eye_blinks", "eye_movements"]  # AF3, AF4, F7, F8
temporal_artifacts = ["muscle_tension", "jaw_clenching"]  # T7, T8
occipital_artifacts = ["alpha_rhythm", "visual_artifacts"]  # O1, O2
```

## 🎯 **Optimization for Digit Recognition:**

### **Visual Processing Pipeline:**
```python
# Expected signal flow for digit recognition:
1. Visual stimulus → O1, O2 (primary visual cortex)
2. Pattern recognition → P7, P8 (visual-spatial processing)
3. Attention/focus → AF3, AF4 (frontal attention)
4. Integration → FC5, FC6 (sensorimotor integration)
```

### **Feature Extraction Strategy:**
```python
# Channel grouping for feature extraction:
visual_group = ["O1", "O2"]  # Primary visual features
parietal_group = ["P7", "P8"]  # Spatial processing features
frontal_group = ["AF3", "AF4", "F3", "F4"]  # Attention features
lateral_group = ["F7", "F8", "T7", "T8"]  # Hemispheric processing
central_group = ["FC5", "FC6"]  # Motor/integration features
```

## 🔬 **Research Insights:**

### **Visual ERP Components:**
```python
# Expected ERP components for visual digit recognition:
P1 (80-120ms): Early visual processing (O1, O2)
N170 (150-200ms): Object recognition (P7, P8)
P300 (250-500ms): Attention/recognition (frontal-parietal)
```

### **Frequency Bands of Interest:**
```python
# Relevant frequency bands by region:
visual_bands = {
    "alpha": (8-13),  # Visual attention (O1, O2)
    "gamma": (30-50)  # Visual binding (O1, O2, P7, P8)
}

attention_bands = {
    "theta": (4-8),   # Attention (AF3, AF4)
    "beta": (13-30)   # Cognitive processing (frontal)
}
```

## ✅ **Validation:**

### **Channel Verification:**
```python
# Verify correct channel order in data:
expected_channels = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

# Check data shape:
assert data.shape[1] == 14, "Should have 14 channels"
assert len(expected_channels) == 14, "Channel list should have 14 entries"
```

### **Spatial Consistency:**
```python
# Verify left-right symmetry:
left_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1"]
right_channels = ["AF4", "F8", "F4", "FC6", "T8", "P8", "O2"]

assert len(left_channels) == len(right_channels), "Should have symmetric layout"
```

---

**📝 Note**: This channel configuration is specific to MindBigData's use of the Emotiv EPOC headset. Other EEG systems may use different channel layouts and naming conventions.

**🎯 Key Takeaway**: The occipital channels (O1, O2) and parietal channels (P7, P8) are most critical for visual digit recognition tasks.
