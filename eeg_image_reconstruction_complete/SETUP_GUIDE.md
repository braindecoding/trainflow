# 🚀 Setup Guide - EEG-to-Image Reconstruction

## 📋 Quick Setup (3 Steps)

### Step 1: Install Dependencies
```bash
python install_dependencies.py
```

### Step 2: Verify Dataset Files
Pastikan file-file ini ada:
- `dataset/datasets/EP1.01.txt` (Dataset MindbigData)
- `dataset/datasets/0.jpg` sampai `dataset/datasets/9.jpg` (10 stimulus images)

### Step 3: Run Pipeline
```bash
# Full pipeline (50k samples, ~2-4 hours)
python run_complete_pipeline.py

# OR Quick start (5k samples, ~30-60 minutes)
python quick_start.py
```

## 📁 Project Structure

```
eeg_image_reconstruction_complete/
├── README.md                          # Main documentation
├── SETUP_GUIDE.md                     # This setup guide
├── requirements.txt                   # Python dependencies
├── install_dependencies.py            # Auto-installer
├── run_complete_pipeline.py           # 🎯 MAIN SCRIPT
├── quick_start.py                     # Quick test version
├── dataset/
│   └── datasets/
│       ├── EP1.01.txt                 # MindbigData file (2.8GB)
│       ├── 0.jpg                      # Stimulus image 0
│       ├── 1.jpg                      # Stimulus image 1
│       └── ... (sampai 9.jpg)         # Total 10 stimulus images
├── src/                               # Source code modules
│   ├── __init__.py
│   ├── data_processing.py             # Dataset processing
│   ├── model_architecture.py          # VAE-CLIP model
│   ├── training.py                    # Training pipeline
│   ├── evaluation.py                  # Evaluation & visualization
│   └── utils.py                       # Utility functions
└── results/                           # Output folder (auto-created)
    ├── models/                        # Trained models
    ├── visualizations/                # Result images
    └── logs/                          # Training logs
```

## ⚙️ System Requirements

### Minimum Requirements:
- **Python**: 3.7+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space
- **CPU**: Multi-core processor

### Recommended for Faster Training:
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: 16GB+
- **Storage**: SSD with 10GB+ free space

## 🔧 Configuration Options

Edit `run_complete_pipeline.py` untuk mengubah:

```python
config = {
    # Dataset Size (adjust based on your RAM)
    'max_train_samples': 50000,    # Reduce if out of memory
    'max_test_samples': 10000,     # Reduce if out of memory
    
    # Training Speed vs Quality
    'batch_size': 32,              # Reduce if out of memory
    'num_epochs': 15,              # Increase for better quality
    
    # Hardware
    'use_gpu': True,               # Set False for CPU only
    'num_workers': 0,              # Keep 0 for Windows
}
```

## 🚨 Troubleshooting

### Memory Issues:
```python
# Reduce these values in config:
'max_train_samples': 20000,
'max_test_samples': 5000,
'batch_size': 16,
```

### GPU Issues:
```python
# Force CPU mode:
'use_gpu': False,
```

### Missing Files:
```bash
# Check if files exist:
ls dataset/datasets/EP1.01.txt
ls dataset/datasets/*.jpg
```

### Installation Issues:
```bash
# Manual installation:
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib seaborn Pillow tqdm opencv-python
pip install git+https://github.com/openai/CLIP.git
```

## 📊 Expected Output

### Console Output:
```
🧠🎨 EEG-TO-IMAGE RECONSTRUCTION COMPLETE PIPELINE
======================================================================
Phase 1: Dataset Processing...
  ✅ Found 908,476 total lines
  ✅ Extracted 45,234 valid samples (subjects 0-9)

Phase 2: Model Training...
  Epoch 1/15: Train Loss 6.234, Val Loss 5.987
  Epoch 15/15: Train Loss 4.123, Val Loss 4.456

Phase 3: Evaluation...
  ✅ Top-1 Accuracy: 18.5%
  ✅ Top-3 Accuracy: 52.3%
  ✅ Top-5 Accuracy: 71.2%

🎉 PIPELINE COMPLETED SUCCESSFULLY!
```

### Generated Files:
```
results/
├── models/
│   └── vae_clip_model_final.pth       # Trained model
├── visualizations/
│   ├── reconstruction_results.png     # 🎯 Main results
│   └── training_curves.png            # Training progress
└── logs/
    ├── evaluation_results.json        # Detailed metrics
    ├── final_summary.json             # Complete summary
    └── training_log.txt               # Full training log
```

## 🎯 Main Result Format

File `results/visualizations/reconstruction_results.png` akan menampilkan:

```
[EEG Signal] [TRUE Ground Truth] [Prediction 1] [Prediction 2] [Prediction 3]
     │              │                  │             │             │
     │              │                  │             │             │
  Original +     Green border      Gold border   Silver border Bronze border
  VAE recon    (TRUE stimulus)    (Top-1 pred)  (Top-2 pred)  (Top-3 pred)
                                      ✅ if correct  ✅ if correct ✅ if correct
```

## 🔄 Running Multiple Experiments

### Different Dataset Sizes:
```python
# Experiment 1: Small (fast)
'max_train_samples': 10000,
'max_test_samples': 2000,

# Experiment 2: Medium 
'max_train_samples': 30000,
'max_test_samples': 6000,

# Experiment 3: Large (best quality)
'max_train_samples': 100000,
'max_test_samples': 20000,
```

### Different Model Configurations:
```python
# Faster training:
'vae_latent_dim': 64,
'batch_size': 16,
'num_epochs': 10,

# Better quality:
'vae_latent_dim': 256,
'batch_size': 64,
'num_epochs': 25,
```

## 📞 Support

### Check Logs:
1. `results/logs/training_log.txt` - Full training progress
2. `results/logs/error_log.json` - Error details if failed

### Common Solutions:
- **Out of Memory**: Reduce `max_train_samples` and `batch_size`
- **Slow Training**: Set `use_gpu: True` if you have NVIDIA GPU
- **Missing Files**: Ensure EP1.01.txt and stimulus images are in correct folders

## ✅ Success Indicators

Pipeline berhasil jika:
1. ✅ All phases complete without errors
2. ✅ `reconstruction_results.png` shows clear visualizations
3. ✅ Top-3 accuracy > 30%
4. ✅ Model file `vae_clip_model_final.pth` created

## 🎉 Ready to Run!

Setelah setup selesai, tinggal jalankan:
```bash
python run_complete_pipeline.py
```

Dan tunggu hasil rekonstruksi EEG-to-Image dengan format yang Anda minta! 🧠✨
