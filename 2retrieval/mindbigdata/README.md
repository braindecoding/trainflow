# 🧠 MindBigData Retrieval Pipeline

EEG-to-image retrieval system specifically designed for MindBigData digit recognition dataset.

## 📊 Input Data Specifications

### **Source Data:**
```python
# 🎯 FROM: 1loaddata/preprocessed_full_production/
Train EEG: (51,900, 14, 256) - 65K trials
Test EEG: (12,975, 14, 256) - Perfect balance
Channels: 14 EPOC channels (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
Duration: 2 seconds (256 timepoints at 128Hz)
Labels: Digit codes 0-9
Quality: Z-score normalized, balanced distribution
```

### **Target Images:**
```python
# 🎯 MNIST DIGITS (28x28):
Format: Grayscale images
Classes: 10 digits (0-9)
Size: 28x28 pixels
Normalization: 0-1 range
Augmentation: Optional rotations, scaling
```

## 🔬 Retrieval Methodology

### **EEG Feature Extraction:**
```python
# 🧠 Multi-domain features:
1. Temporal features:
   - Signal amplitude patterns
   - Peak detection
   - Temporal dynamics
   
2. Spectral features:
   - Power spectral density
   - Frequency band analysis (alpha, beta, gamma)
   - Spectral entropy
   
3. Spatial features:
   - Cross-channel correlations
   - Spatial patterns
   - Channel-wise statistics
   
4. Time-frequency features:
   - Wavelet coefficients
   - Short-time Fourier transform
   - Spectrograms
```

### **Image Feature Extraction:**
```python
# 🖼️ Visual features:
1. Pixel-level features:
   - Raw pixel values
   - Gradient information
   - Edge detection
   
2. Shape features:
   - Contour analysis
   - Geometric properties
   - Structural patterns
   
3. Texture features:
   - Local binary patterns
   - Gabor filters
   - Statistical textures
```

### **Similarity Matching:**
```python
# 🔗 EEG-Image pairing:
1. Direct matching:
   - Use digit labels for exact pairing
   - One-to-one correspondence
   - Label-guided retrieval
   
2. Feature similarity:
   - Cosine similarity
   - Euclidean distance
   - Learned metrics
   
3. Multi-stage retrieval:
   - Coarse-to-fine matching
   - Quality filtering
   - Confidence scoring
```

## 🎯 Implementation Strategy

### **Phase 1: Label-based Pairing**
```python
# 🎯 Exact digit matching:
For each EEG trial with digit label d:
1. Load corresponding MNIST digit d
2. Create EEG-image pair
3. Ensure balanced representation
4. Quality validation

Expected output:
- 51,900 train pairs
- 12,975 test pairs
- Perfect label correspondence
```

### **Phase 2: Feature-based Retrieval**
```python
# ⚡ Similarity-based matching:
1. Extract EEG features (multi-domain)
2. Extract image features (visual)
3. Learn similarity mapping
4. Retrieve best matching images
5. Quality assessment

Expected improvement:
- Better semantic alignment
- Robust to label noise
- Enhanced generalization
```

### **Phase 3: Learned Retrieval**
```python
# 🏆 Neural retrieval system:
1. Train EEG encoder
2. Train image encoder
3. Learn shared embedding space
4. Optimize retrieval metrics
5. End-to-end optimization

Expected performance:
- 90%+ retrieval accuracy
- Semantic consistency
- Robust representations
```

## 📁 Output Structure

### **Generated Files:**
```python
# 📦 Output directory: outputs/mindbigdata_pairs/
├── train_eeg_features.npy     # (51,900, feature_dim)
├── train_images.npy           # (51,900, 28, 28)
├── train_labels.npy           # (51,900,) - digit codes
├── test_eeg_features.npy      # (12,975, feature_dim)
├── test_images.npy            # (12,975, 28, 28)
├── test_labels.npy            # (12,975,) - digit codes
├── retrieval_metadata.json    # Parameters and metrics
└── quality_report.html        # Retrieval quality analysis
```

### **Quality Metrics:**
```python
# 📊 Evaluation metrics:
- Retrieval accuracy (top-1, top-5, top-10)
- Feature quality scores
- EEG-image similarity distributions
- Cross-validation performance
- Processing time and efficiency
```

## 🚀 Expected Performance

### **Retrieval Accuracy:**
```python
# 🎯 Performance targets:
Phase 1 (Label-based): 100% accuracy (exact matching)
Phase 2 (Feature-based): 85-90% top-1 accuracy
Phase 3 (Learned): 90-95% top-1 accuracy

# 📈 Impact on downstream tasks:
Current CLIP R@1: 15.69%
With Phase 1: 25-35% R@1
With Phase 2: 35-50% R@1
With Phase 3: 50-70% R@1
```

### **Quality Assessment:**
```python
# 🔬 Quality criteria:
- Semantic consistency between EEG and images
- Feature representation richness
- Balanced class distribution
- Noise robustness
- Scalability to full dataset
```

## 🛠️ Technical Implementation

### **Core Components:**
```python
# 📦 Main modules:
1. eeg_feature_extractor.py    # EEG signal processing
2. image_loader.py             # MNIST data handling
3. retrieval_engine.py         # Similarity matching
4. quality_assessor.py         # Performance evaluation
5. data_generator.py           # Paired dataset creation
```

### **Processing Pipeline:**
```python
# 🔄 Workflow:
1. Load preprocessed EEG data
2. Extract multi-domain features
3. Load MNIST digit images
4. Perform retrieval/matching
5. Generate paired dataset
6. Quality assessment
7. Export for CLIP training
```

## 📈 Success Metrics

### **Technical Success:**
```python
# ✅ Completion criteria:
- High retrieval accuracy (>85%)
- Balanced paired dataset
- Efficient processing (<1 hour for 65K)
- Quality validation passed
- Ready for CLIP training
```

### **Scientific Impact:**
```python
# 🔬 Research contributions:
- Novel EEG-image retrieval for digits
- Scalable preprocessing methodology
- Benchmark performance metrics
- Foundation for superior CLIP training
```

## 🎯 Next Steps

### **Immediate Implementation:**
```python
# 🚀 Priority tasks:
1. Implement basic EEG feature extraction
2. Load and prepare MNIST images
3. Create label-based pairing system
4. Generate initial paired dataset
5. Quality validation and metrics
```

### **Future Enhancements:**
```python
# 🔮 Advanced features:
- Multi-modal feature fusion
- Learned similarity metrics
- Real-time retrieval
- Interactive visualization
- Cross-dataset validation
```

---

**Status**: 🎯 Ready for implementation

**Input**: 65K preprocessed EEG trials + MNIST digits

**Output**: High-quality EEG-image pairs for CLIP training

**Goal**: Enable superior EEG-to-image reconstruction performance
