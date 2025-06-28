# 🧠🎨 Pendekatan 2: EEG-to-CLIP Direct Mapping

## 📋 Overview

Pendekatan ini mengubah masalah EEG-to-Image menjadi **dua tahap terpisah**:
1. **Tahap 1**: Melatih Encoder EEG untuk memprediksi CLIP embedding secara langsung
2. **Tahap 2**: Menggunakan pre-trained image generator untuk mengubah CLIP embedding menjadi gambar

Ini adalah pendekatan yang lebih **radikal dan ambisius** yang memisahkan "pemahaman konsep" dari "generasi visual".

### 🏗️ Arsitektur

```
TAHAP 1: EEG → CLIP Embedding
EEG Signal (264-dim)
         ↓
    EEG-to-CLIP Encoder
         ↓
    Predicted CLIP Embedding (512-dim)
         ↓
    [Loss: Distance to TRUE CLIP Embedding]

TAHAP 2: CLIP Embedding → Image  
Predicted CLIP Embedding (512-dim)
         ↓
    Pre-trained Image Decoder
    (VQGAN/StyleGAN/Diffusion)
         ↓
    Generated Image (224×224×3)
```

### 🎯 Training Strategy

**Tahap 1 - EEG-to-CLIP Mapping:**
```python
# Input: EEG signal dari subjek yang melihat angka '7'
# Target: CLIP embedding dari gambar angka '7' yang asli
# Loss: L2 atau cosine distance antara predicted vs target embedding

predicted_clip_embedding = eeg_encoder(eeg_signal)
target_clip_embedding = clip_model.encode_image(target_image)
loss = 1 - cosine_similarity(predicted_clip_embedding, target_clip_embedding)
```

**Tahap 2 - CLIP-to-Image Generation:**
```python
# Input: Predicted CLIP embedding dari Tahap 1
# Output: Generated image menggunakan pre-trained decoder
# Tidak perlu training tambahan jika menggunakan pre-trained model

generated_image = pretrained_decoder(predicted_clip_embedding)
```

### ✨ Keunggulan Pendekatan 2

1. **Conceptual Clarity**: Memisahkan "understanding" dari "generation"
2. **Leveraging Pre-trained Models**: Menggunakan decoder yang sudah powerful
3. **Interpretable Intermediate**: CLIP embedding bisa dianalisis secara semantik
4. **Modular Design**: Bisa mengganti decoder tanpa retrain encoder
5. **TRUE Labels**: Menggunakan subject-stimulus correspondence yang benar

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Two-Stage Training
```bash
python run_approach2_pipeline.py
```

## 📊 Expected Results

### Stage 1 Performance:
- **CLIP Embedding Similarity**: 0.7-0.9 cosine similarity
- **Semantic Accuracy**: 70-90% (concept-level correctness)

### Stage 2 Performance:
- **Image Quality**: High (leveraging pre-trained decoder)
- **Semantic Consistency**: Very high (CLIP-guided)

### Overall Performance:
- **Top-3 Stimulus Match**: 60-80%
- **Visual Quality**: Excellent semantic correctness

## 🔧 Configuration

Edit `run_approach2_pipeline.py`:

```python
config = {
    # Stage 1: EEG-to-CLIP
    'stage1_epochs': 30,        # More epochs for embedding learning
    'stage1_lr': 1e-3,          # Higher learning rate
    'embedding_dim': 512,       # CLIP embedding dimension
    'encoder_hidden_dims': [512, 256, 128],  # Encoder architecture
    
    # Stage 2: CLIP-to-Image
    'decoder_type': 'vqgan',    # 'vqgan', 'stylegan', or 'diffusion'
    'use_pretrained': True,     # Use pre-trained decoder
    
    # Loss configuration
    'similarity_loss': 'cosine', # 'cosine' or 'l2'
}
```

## 📁 File Structure

```
approach2_eeg_to_clip_direct/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── run_approach2_pipeline.py          # 🎯 MAIN SCRIPT
├── dataset/
│   └── datasets/
│       ├── EP1.01.txt                 # MindbigData
│       └── *.jpg                      # Stimulus images
├── src/
│   ├── eeg_to_clip_model.py           # Stage 1: EEG→CLIP encoder
│   ├── clip_to_image_model.py         # Stage 2: CLIP→Image decoder
│   ├── two_stage_training.py          # Two-stage training pipeline
│   ├── data_processing.py             # Dataset processing
│   ├── evaluation.py                  # Evaluation & visualization
│   └── utils.py                       # Utilities
└── results/                           # Output folder
    ├── stage1_models/                 # EEG-to-CLIP models
    ├── stage2_models/                 # CLIP-to-Image models
    ├── visualizations/                # Generated images
    └── logs/                          # Training logs
```

## 🎨 Key Innovation

**Direct CLIP Embedding Prediction**:
```python
class EEGToCLIPEncoder(nn.Module):
    def __init__(self, eeg_input_dim, clip_embedding_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(eeg_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, clip_embedding_dim)
        )
    
    def forward(self, eeg_signals):
        # Direct prediction of CLIP embedding
        predicted_embedding = self.encoder(eeg_signals)
        # Normalize to unit sphere (like CLIP)
        return F.normalize(predicted_embedding, dim=1)
```

## 🔬 Technical Details

### Stage 1 - EEG-to-CLIP Training:
1. **Input**: EEG signal (264-dim)
2. **Target**: CLIP embedding dari stimulus image yang benar
3. **Architecture**: Multi-layer perceptron dengan dropout
4. **Loss**: Cosine similarity loss
5. **Optimization**: Adam dengan learning rate scheduling

### Stage 2 - CLIP-to-Image Generation:
1. **Input**: Predicted CLIP embedding dari Stage 1
2. **Decoder Options**:
   - **VQGAN**: Vector Quantized GAN
   - **StyleGAN**: Style-based GAN
   - **Diffusion**: Stable Diffusion dengan CLIP conditioning
3. **Pre-trained**: Menggunakan model yang sudah dilatih

### Evaluation Pipeline:
1. **Stage 1 Evaluation**: Cosine similarity dengan true CLIP embeddings
2. **Stage 2 Evaluation**: Image quality dan semantic correctness
3. **End-to-End**: EEG → CLIP → Image → stimulus matching

## 🎯 Expected Advantages

1. **Higher Semantic Accuracy**: Direct CLIP prediction lebih focused
2. **Better Image Quality**: Pre-trained decoder sudah optimal
3. **Interpretable**: CLIP embedding bisa dianalisis
4. **Modular**: Bisa upgrade decoder tanpa retrain encoder
5. **Efficient**: Stage 1 training lebih cepat dari full GAN

## 🚀 Ready to Run!

Folder ini siap dijalankan dengan:
```bash
python run_approach2_pipeline.py
```

Akan menghasilkan rekonstruksi EEG-to-Image dengan **two-stage CLIP mapping**! 🧠✨🎨
