#!/usr/bin/env python3
"""
EEG Dataset MindBigData Adapter
==============================

Adapter to make MindBigData compatible with original EEGDataset interface.
This allows ATMS_retrieval.py to work with MindBigData without major changes.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import clip
from torch.nn import functional as F
import json

# Setup device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model_type = 'ViT-B/32'
vlmodel, preprocess_train = clip.load(model_type, device=device)

# Load configuration
config_path = "data_config.json"
with open(config_path, "r") as config_file:
    config = json.load(config_file)

data_path = config["data_path"]
img_directory_training = config["img_directory_training"]
img_directory_test = config["img_directory_test"]

class EEGDataset():
    """
    MindBigData adapter that mimics the original EEGDataset interface
    
    This class provides compatibility with ATMS_retrieval.py while using
    MindBigData preprocessed data instead of Things-EEG2 subject-based data.
    """
    
    def __init__(self, data_path, exclude_subject=None, subjects=None, train=True, 
                 time_window=[0, 1.8], classes=None, pictures=None, val_size=None):
        
        # Store parameters (for compatibility)
        self.data_path = data_path
        self.train = train
        self.exclude_subject = exclude_subject
        self.subjects = subjects or ['mindbigdata']  # Single "subject" for MindBigData
        self.n_sub = 1  # Single dataset
        self.time_window = time_window
        self.n_cls = 10  # 10 digits
        self.classes = classes
        self.pictures = pictures
        self.val_size = val_size
        
        print(f"🧠 Loading MindBigData via EEGDataset adapter...")
        print(f"   Mode: {'Train' if train else 'Test'}")
        print(f"   Data path: {data_path}")
        print(f"   Subjects: {self.subjects}")
        
        # Load data using MindBigData format
        self.data, self.labels, self.text, self.img = self.load_data()
        
        # Extract EEG in time window
        self.data = self.extract_eeg(self.data, time_window)
        
        # Generate CLIP features (cached)
        self.text_features = self.Textencoder(self.text)
        self.img_features = self.ImageEncoder(self.img)
        
        print(f"✅ Dataset loaded:")
        print(f"   EEG data: {self.data.shape}")
        print(f"   Labels: {len(self.labels)}")
        print(f"   Text features: {self.text_features.shape}")
        print(f"   Image features: {self.img_features.shape}")
    
    def load_data(self):
        """Load MindBigData in format compatible with original EEGDataset"""
        
        # Determine mode
        mode = "train" if self.train else "test"
        
        # Load preprocessed MindBigData
        eeg_file = os.path.join(data_path, f"{mode}_eeg_data.npy")
        labels_file = os.path.join(data_path, f"{mode}_labels.npy")
        
        # Load data
        eeg_data = np.load(eeg_file)  # (N, 14, 230)
        labels = np.load(labels_file)  # (N,)
        
        # Convert to torch tensors
        data = torch.from_numpy(eeg_data).float()
        labels = torch.from_numpy(labels).long()
        
        # Generate texts and images for each trial
        texts = []
        images = []
        
        for label in labels:
            digit = label.item()
            texts.append(f"This picture is digit {digit}")
            img_path = os.path.join(img_directory_training, f"{digit}.jpg")
            images.append(img_path)
        
        print(f"   Loaded {len(data)} trials")
        print(f"   Unique labels: {torch.unique(labels).tolist()}")
        
        return data, labels, texts, images
    
    def extract_eeg(self, eeg_data, time_window):
        """Extract EEG data within time window"""
        
        # MindBigData is already windowed to [0, 1.8]s (230 timepoints at 250Hz)
        # Just return as-is since it matches the expected time window
        
        print(f"   Time window: {time_window}")
        print(f"   EEG shape after extraction: {eeg_data.shape}")
        
        return eeg_data
    
    def Textencoder(self, texts):
        """Encode texts using CLIP - compatible with original interface"""
        
        print(f"📝 Encoding {len(texts)} texts...")
        
        # Use cached features if available
        cache_file = f"ViT-B_32_features_mindbigdata_{'train' if self.train else 'test'}.pt"
        
        if os.path.exists(cache_file):
            print(f"   Loading cached features from {cache_file}")
            cached = torch.load(cache_file, map_location='cpu')
            
            # Expand cached features to match number of trials
            unique_features = cached['text_features']  # (10, 512)
            expanded_features = torch.zeros(len(texts), unique_features.shape[1])
            
            for i, label in enumerate(self.labels):
                expanded_features[i] = unique_features[label.item()]
            
            return expanded_features.to(device)
        
        # If no cache, encode normally
        unique_texts = [f"This picture is digit {i}" for i in range(10)]
        text_inputs = torch.cat([clip.tokenize(t) for t in unique_texts]).to(device)
        
        with torch.no_grad():
            text_features = vlmodel.encode_text(text_inputs)
        
        text_features = F.normalize(text_features, dim=-1).detach()
        
        # Expand to match trials
        expanded_features = torch.zeros(len(texts), text_features.shape[1])
        for i, label in enumerate(self.labels):
            expanded_features[i] = text_features[label.item()]
        
        return expanded_features.to(device)
    
    def ImageEncoder(self, image_paths):
        """Encode images using CLIP - compatible with original interface"""
        
        print(f"🖼️ Encoding {len(image_paths)} images...")
        
        # Use cached features if available
        cache_file = f"ViT-B_32_features_mindbigdata_{'train' if self.train else 'test'}.pt"
        
        if os.path.exists(cache_file):
            print(f"   Loading cached features from {cache_file}")
            cached = torch.load(cache_file, map_location='cpu')
            
            # Expand cached features to match number of trials
            unique_features = cached['img_features']  # (10, 512)
            expanded_features = torch.zeros(len(image_paths), unique_features.shape[1])
            
            for i, label in enumerate(self.labels):
                expanded_features[i] = unique_features[label.item()]
            
            return expanded_features.to(device)
        
        # If no cache, encode normally (this would be slow for many images)
        from PIL import Image
        
        unique_images = []
        for digit in range(10):
            img_path = os.path.join(img_directory_training, f"{digit}.jpg")
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                unique_images.append(preprocess_train(image))
        
        if unique_images:
            image_batch = torch.stack(unique_images).to(device)
            
            with torch.no_grad():
                img_features = vlmodel.encode_image(image_batch)
            
            img_features = F.normalize(img_features, dim=-1).detach()
            
            # Expand to match trials
            expanded_features = torch.zeros(len(image_paths), img_features.shape[1])
            for i, label in enumerate(self.labels):
                expanded_features[i] = img_features[label.item()]
            
            return expanded_features.to(device)
        
        # Fallback
        return torch.zeros(len(image_paths), 512).to(device)

# For backward compatibility, export the class
__all__ = ['EEGDataset']
