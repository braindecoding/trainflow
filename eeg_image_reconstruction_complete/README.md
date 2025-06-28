# 🧠🎨 EEG-to-Image Reconstruction dengan VAE-CLIP

## 📋 Overview

Sistem lengkap untuk rekonstruksi image dari sinyal EEG menggunakan:
- **Variational Autoencoder (VAE)** untuk EEG embedding
- **CLIP** untuk cross-modal alignment
- **TRUE subject-stimulus correspondence** (Subject 0 → 0.jpg, Subject 1 → 1.jpg, dst.)
- **Full MindbigData EP1.01 dataset** (908k+ samples)

## 🎯 Output Format

Sistem menghasilkan visualisasi dengan format:
- **Column 1**: EEG Signal (Original + VAE Reconstruction)
- **Column 2**: TRUE Ground Truth Image (Green border)
- **Column 3**: Top-1 Prediction (Gold border, ✅ if correct)
- **Column 4**: Top-2 Prediction (Silver border, ✅ if correct)
- **Column 5**: Top-3 Prediction (Bronze border, ✅ if correct)

## 📁 Struktur Folder

```
eeg_image_reconstruction_complete/
├── README.md                          # Panduan ini
├── requirements.txt                   # Dependencies
├── run_complete_pipeline.py           # Script utama - JALANKAN INI
├── dataset/
│   └── datasets/
│       ├── EP1.01.txt                 # Dataset asli MindbigData
│       └── *.jpg                      # 10 stimulus images (0.jpg - 9.jpg)
├── src/
│   ├── data_processing.py             # Processing dataset full
│   ├── model_architecture.py          # VAE-CLIP model
│   ├── training.py                    # Training pipeline
│   ├── evaluation.py                  # Evaluation & visualization
│   └── utils.py                       # Utility functions
└── results/                           # Output folder (auto-created)
    ├── models/                        # Trained models
    ├── visualizations/                # Result images
    └── logs/                          # Training logs
```

## 🚀 Quick Start (Tinggal Run!)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Complete Pipeline
```bash
python run_complete_pipeline.py
```

**That's it!** Script akan otomatis:
1. ✅ Process full dataset EP1.01.txt (908k+ samples)
2. ✅ Split train/test dengan stratified sampling
3. ✅ Train VAE-CLIP model dengan TRUE labels
4. ✅ Evaluate performance
5. ✅ Generate visualizations
6. ✅ Save semua results

## ⚙️ Configuration Options

Edit `run_complete_pipeline.py` untuk mengubah:

```python
# Dataset Configuration
MAX_TRAIN_SAMPLES = 50000      # Limit untuk memory management
MAX_TEST_SAMPLES = 10000       # Limit test samples
CHUNK_SIZE = 50000             # Processing chunk size

# Training Configuration
BATCH_SIZE = 32                # Batch size
NUM_EPOCHS = 15                # Training epochs
LEARNING_RATE = 1e-4           # Learning rate
VAE_LATENT_DIM = 128           # VAE latent dimension

# Hardware Configuration
USE_GPU = True                 # Set False untuk CPU only
NUM_WORKERS = 4                # DataLoader workers
```

## 📊 Expected Results

### Performance Metrics:
- **Top-1 Accuracy**: 15-25%
- **Top-3 Accuracy**: 40-60%
- **Top-5 Accuracy**: 60-80%

### Training Time:
- **50k samples**: ~30-60 minutes (GPU) / 2-4 hours (CPU)
- **100k samples**: ~1-2 hours (GPU) / 4-8 hours (CPU)

### Generated Files:
```
results/
├── models/
│   ├── vae_clip_model_final.pth       # Trained model
│   └── training_config.json           # Training configuration
├── visualizations/
│   ├── reconstruction_results.png     # Main results (format yang diminta)
│   ├── training_curves.png            # Loss progression
│   ├── performance_analysis.png       # Performance metrics
│   └── dataset_analysis.png           # Dataset statistics
└── logs/
    ├── training_log.txt               # Training progress
    ├── evaluation_results.json        # Detailed metrics
    └── dataset_summary.json           # Dataset statistics
```

## 🔧 Troubleshooting

### Memory Issues:
```python
# Reduce these values in run_complete_pipeline.py
MAX_TRAIN_SAMPLES = 20000
BATCH_SIZE = 16
```

### GPU Issues:
```python
# Set to CPU mode
USE_GPU = False
```

### Dataset Issues:
- Pastikan `dataset/datasets/EP1.01.txt` ada
- Pastikan `dataset/datasets/*.jpg` (10 files) ada

## 📈 Monitoring Progress

Script akan menampilkan progress real-time:

```
Phase 1: Processing full dataset...
  ✅ Found 908,476 total lines
  ✅ Extracted 45,234 valid samples (subjects 0-9)
  ✅ Train: 36,187 samples, Test: 9,047 samples

Phase 2: Training VAE-CLIP model...
  Epoch 1/15: Train Loss 6.234, Val Loss 5.987, Time 234s
  Epoch 2/15: Train Loss 5.876, Val Loss 5.654, Time 468s
  ...

Phase 3: Evaluation...
  ✅ Top-1 Accuracy: 18.5%
  ✅ Top-3 Accuracy: 52.3%
  ✅ Top-5 Accuracy: 71.2%

Phase 4: Generating visualizations...
  ✅ Reconstruction results saved
  ✅ Training curves saved
  ✅ Performance analysis saved
```

## 🎯 Key Features

### ✅ Complete Automation:
- Single command execution
- Automatic error handling
- Progress monitoring
- Result organization

### ✅ TRUE Label Correspondence:
- Subject 0 → 0.jpg
- Subject 1 → 1.jpg
- ... dan seterusnya

### ✅ Memory Management:
- Streaming data processing
- Chunked loading
- Automatic cleanup

### ✅ Comprehensive Output:
- Model weights
- Training logs
- Visualizations
- Performance metrics

## 📞 Support

Jika ada error:
1. Check `results/logs/training_log.txt`
2. Verify dataset files exist
3. Check memory/GPU availability
4. Reduce dataset size if needed

## 🏆 Expected Final Output

File utama yang dihasilkan:
- **`results/visualizations/reconstruction_results.png`** - Format yang Anda minta
- **`results/models/vae_clip_model_final.pth`** - Model terlatih
- **`results/logs/evaluation_results.json`** - Metrics lengkap

**Ready to run!** 🚀
