# 🧠🎨 Pendekatan 1: CLIP sebagai Semantic Loss Function

## 📋 Overview

Pendekatan ini mengintegrasikan CLIP sebagai **fungsi kerugian semantik** dalam arsitektur VAE-GAN untuk rekonstruksi EEG-to-Image. Alih-alih hanya mengandalkan pixel-level reconstruction loss, model juga belajar menghasilkan gambar yang **secara semantik benar** menurut CLIP.

### 🏗️ Arsitektur

```
EEG Signal (264-dim)
         ↓
    EEG Encoder (VAE)
    ├── μ, σ (latent parameters)
    └── z ~ N(μ, σ²) (reparameterization)
         ↓
    Image Generator/Decoder
         ↓
    Generated Image (224×224×3)
         ↓
    [CLIP Frozen Encoder] ← Target Image
         ↓                      ↓
    Generated Embedding    Target Embedding
         ↓                      ↓
         └─── Semantic Loss ────┘
```

### 🎯 Loss Function

**Total Loss = λ_adv × L_GAN + λ_rec × L_REC + λ_kl × L_KL + λ_clip × L_CLIP**

Dimana:
- **L_GAN**: Adversarial loss (Generator vs Discriminator)
- **L_REC**: Pixel-level reconstruction loss (L1)
- **L_KL**: VAE KL divergence loss
- **L_CLIP**: **Semantic loss** = 1 - cosine_similarity(generated_embedding, target_embedding)

### ✨ Keunggulan Pendekatan 1

1. **Semantic Correctness**: Model belajar menghasilkan gambar yang **konseptually correct**
2. **Mengatasi Blur**: Semantic loss tidak terlalu "menghukum" detail kecil yang salah
3. **Stable Training**: Arsitektur VAE-GAN yang proven dengan tambahan semantic guidance
4. **TRUE Labels**: Menggunakan subject-stimulus correspondence yang benar

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Training
```bash
python run_approach1_pipeline.py
```

## 📊 Expected Results

### Performance Metrics:
- **Semantic Accuracy**: 60-80% (berdasarkan CLIP similarity)
- **Top-3 Stimulus Match**: 50-70%
- **Visual Quality**: High semantic correctness, good detail preservation

### Generated Output:
- **Format**: [EEG Signal] [TRUE Ground Truth] [Generated Image] [Top-3 Predictions]
- **Borders**: Green (TRUE), Gold/Silver/Bronze (predictions), ✅ (correct)

## 🔧 Configuration

Edit `run_approach1_pipeline.py`:

```python
config = {
    # Loss weights
    'lambda_adv': 1.0,      # GAN adversarial loss
    'lambda_rec': 10.0,     # Reconstruction loss
    'lambda_kl': 1.0,       # VAE KL loss
    'lambda_clip': 5.0,     # CLIP semantic loss (KEY!)
    
    # Training
    'batch_size': 16,       # Smaller for memory
    'num_epochs': 20,       # More epochs for convergence
    'learning_rate': 2e-4,  # Standard GAN learning rate
}
```

## 📁 File Structure

```
approach1_clip_semantic_loss/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── run_approach1_pipeline.py          # 🎯 MAIN SCRIPT
├── dataset/
│   └── datasets/
│       ├── EP1.01.txt                 # MindbigData
│       └── *.jpg                      # Stimulus images
├── src/
│   ├── semantic_loss_model.py         # VAE-GAN + CLIP model
│   ├── data_processing.py             # Dataset processing
│   ├── training.py                    # Training pipeline
│   ├── evaluation.py                  # Evaluation & visualization
│   └── utils.py                       # Utilities
└── results/                           # Output folder
    ├── models/                        # Trained models
    ├── visualizations/                # Generated images
    └── logs/                          # Training logs
```

## 🎨 Key Innovation

**Semantic Loss Function**:
```python
def semantic_loss_function(generated_images, target_images, clip_model):
    # Get CLIP embeddings
    gen_embeddings = clip_model.encode_image(generated_images)
    target_embeddings = clip_model.encode_image(target_images)
    
    # Cosine similarity loss
    cosine_sim = F.cosine_similarity(gen_embeddings, target_embeddings)
    semantic_loss = 1 - cosine_sim.mean()
    
    return semantic_loss
```

Ini memaksa generator menghasilkan gambar yang **semantically similar** dengan target, bukan hanya pixel-perfect.

## 🔬 Technical Details

### Model Components:
1. **EEG Encoder**: Transforms EEG → latent space (VAE)
2. **Image Generator**: Transforms latent → RGB image (ConvTranspose2d)
3. **Discriminator**: Real/fake image classification
4. **CLIP Encoder**: Frozen pre-trained, untuk semantic loss

### Training Process:
1. **Phase 1**: Train Discriminator (real vs fake)
2. **Phase 2**: Train Generator with combined loss:
   - Fool discriminator (adversarial)
   - Reconstruct pixels (L1)
   - Maintain latent distribution (KL)
   - **Match semantic content (CLIP)**

## 🎯 Expected Advantages

1. **Better Semantic Quality**: Gambar yang dihasilkan lebih "meaningful"
2. **Reduced Blur**: Semantic loss membantu detail yang sharp
3. **Robust to Pixel Variations**: Focus pada konsep, bukan pixel exact
4. **Improved Generalization**: CLIP knowledge membantu unseen variations

## 🚀 Ready to Run!

Folder ini siap dijalankan dengan:
```bash
python run_approach1_pipeline.py
```

Akan menghasilkan rekonstruksi EEG-to-Image dengan **semantic guidance dari CLIP**! 🧠✨🎨
