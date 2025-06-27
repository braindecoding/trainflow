#!/usr/bin/env python3
"""
MindBigData Retrieval System
===========================

Adapted from eegdatasets_leaveone.py for MindBigData digit recognition.
Creates EEG-image pairs using CLIP embeddings for contrastive learning.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm

# Setup device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP model (using standard clip library)
model_type = 'ViT-B/32'
vlmodel, preprocess_train = clip.load(model_type, device=device)

class MindBigDataRetrieval:
    """
    MindBigData EEG-to-Image Retrieval System
    
    Adapts the methodology from eegdatasets_leaveone.py for MindBigData:
    - Loads preprocessed EEG data (65K trials)
    - Loads digit stimuli images (0-9)
    - Creates EEG-image pairs using CLIP embeddings
    - Generates paired dataset for CLIP training
    """
    
    def __init__(self,
                 eeg_data_path=r"d:\trainflow\1loaddata\preprocessed_full_production",
                 stimuli_path=r"d:\trainflow\dataset\datasets\MindbigdataStimuli",
                 output_path=r"d:\trainflow\2retrieval\outputs\mindbigdata_pairs",
                 train=True,
                 time_window=[0, 2.0],
                 use_cached_features=True):
        
        self.eeg_data_path = eeg_data_path
        self.stimuli_path = stimuli_path
        self.output_path = output_path
        self.train = train
        self.time_window = time_window
        self.use_cached_features = use_cached_features
        
        print(f"🧠 Initializing MindBigData Retrieval System...")
        print(f"   EEG data: {eeg_data_path}")
        print(f"   Stimuli: {stimuli_path}")
        print(f"   Output: {output_path}")
        print(f"   Mode: {'Train' if train else 'Test'}")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Load data
        self.eeg_data, self.labels = self.load_eeg_data()
        self.images, self.texts = self.load_stimuli_data()
        
        # Extract EEG features in time window
        self.eeg_data = self.extract_eeg(self.eeg_data, time_window)
        
        # Generate CLIP features
        self.text_features, self.img_features = self.generate_clip_features()
        
        print(f"✅ Initialization complete!")
        print(f"   EEG data: {self.eeg_data.shape}")
        print(f"   Labels: {self.labels.shape}")
        print(f"   Images: {len(self.images)}")
        print(f"   Texts: {len(self.texts)}")
    
    def load_eeg_data(self):
        """Load preprocessed EEG data from 1loaddata"""
        
        print(f"📂 Loading EEG data...")
        
        if self.train:
            data_file = os.path.join(self.eeg_data_path, "train_data.npy")
            labels_file = os.path.join(self.eeg_data_path, "train_labels.npy")
        else:
            data_file = os.path.join(self.eeg_data_path, "test_data.npy")
            labels_file = os.path.join(self.eeg_data_path, "test_labels.npy")
        
        # Load data
        eeg_data = np.load(data_file)  # (N, 14, 256)
        labels = np.load(labels_file)  # (N,)
        
        # Convert to torch tensors
        eeg_data = torch.from_numpy(eeg_data).float()
        labels = torch.from_numpy(labels).long()
        
        print(f"   EEG data shape: {eeg_data.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Unique labels: {torch.unique(labels).tolist()}")
        
        return eeg_data, labels
    
    def load_stimuli_data(self):
        """Load MindBigData stimuli images and generate text descriptions"""
        
        print(f"🖼️ Loading stimuli data...")
        
        images = []
        texts = []
        
        # Load digit images (0-9)
        for digit in range(10):
            img_path = os.path.join(self.stimuli_path, f"{digit}.jpg")
            if os.path.exists(img_path):
                images.append(img_path)
                texts.append(f"This picture is digit {digit}")
            else:
                print(f"⚠️ Warning: Missing stimuli for digit {digit}")
        
        print(f"   Loaded {len(images)} stimuli images")
        print(f"   Generated {len(texts)} text descriptions")
        
        return images, texts
    
    def extract_eeg(self, eeg_data, time_window):
        """Extract EEG data within specified time window"""
        
        print(f"⏱️ Extracting EEG time window: {time_window}")
        
        # Assuming 128Hz sampling rate, 2 seconds = 256 timepoints
        sampling_rate = 128
        start_time, end_time = time_window
        
        start_idx = int(start_time * sampling_rate)
        end_idx = int(end_time * sampling_rate)
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(eeg_data.shape[-1], end_idx)
        
        extracted_data = eeg_data[..., start_idx:end_idx]
        
        print(f"   Original shape: {eeg_data.shape}")
        print(f"   Extracted shape: {extracted_data.shape}")
        print(f"   Time indices: [{start_idx}, {end_idx}]")
        
        return extracted_data
    
    def generate_clip_features(self):
        """Generate CLIP features for texts and images"""
        
        print(f"🔮 Generating CLIP features...")
        
        # Check for cached features
        model_name = model_type.replace('/', '_')
        features_filename = f'{model_name}_features_mindbigdata_{"train" if self.train else "test"}.pt'
        
        if self.use_cached_features and os.path.exists(features_filename):
            print(f"   Loading cached features from {features_filename}")
            saved_features = torch.load(features_filename)
            text_features = saved_features['text_features']
            img_features = saved_features['img_features']
        else:
            print(f"   Computing new features...")
            text_features = self.encode_texts(self.texts)
            img_features = self.encode_images(self.images)
            
            # Cache features
            torch.save({
                'text_features': text_features.cpu(),
                'img_features': img_features.cpu(),
            }, features_filename)
            print(f"   Cached features to {features_filename}")
        
        print(f"   Text features: {text_features.shape}")
        print(f"   Image features: {img_features.shape}")
        
        return text_features, img_features
    
    def encode_texts(self, texts):
        """Encode text descriptions using CLIP text encoder"""
        
        print(f"📝 Encoding {len(texts)} text descriptions...")
        
        text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(device)
        
        with torch.no_grad():
            text_features = vlmodel.encode_text(text_inputs)
        
        text_features = F.normalize(text_features, dim=-1).detach()
        
        return text_features
    
    def encode_images(self, image_paths):
        """Encode images using CLIP image encoder"""
        
        print(f"🖼️ Encoding {len(image_paths)} images...")
        
        batch_size = 10  # Small batch for digit images
        image_features_list = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load and preprocess images
            batch_images = []
            for img_path in batch_paths:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(preprocess_train(img))
            
            image_inputs = torch.stack(batch_images).to(device)
            
            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)
                batch_image_features = F.normalize(batch_image_features, dim=-1)
            
            image_features_list.append(batch_image_features)
        
        image_features = torch.cat(image_features_list, dim=0)
        
        return image_features
    
    def create_eeg_image_pairs(self):
        """Create EEG-image pairs using label-based matching"""
        
        print(f"🔗 Creating EEG-image pairs...")
        
        n_trials = len(self.labels)
        paired_eeg = []
        paired_images = []
        paired_labels = []
        paired_text_features = []
        paired_img_features = []
        
        for i in tqdm(range(n_trials), desc="Creating pairs"):
            label = self.labels[i].item()
            
            # Get EEG trial
            eeg_trial = self.eeg_data[i]
            
            # Get corresponding image and features
            if label < len(self.images):
                img_path = self.images[label]
                text_feat = self.text_features[label]
                img_feat = self.img_features[label]
                
                paired_eeg.append(eeg_trial)
                paired_images.append(img_path)
                paired_labels.append(label)
                paired_text_features.append(text_feat)
                paired_img_features.append(img_feat)
        
        # Convert to tensors
        paired_eeg = torch.stack(paired_eeg)
        paired_labels = torch.tensor(paired_labels)
        paired_text_features = torch.stack(paired_text_features)
        paired_img_features = torch.stack(paired_img_features)
        
        print(f"✅ Created {len(paired_eeg)} EEG-image pairs")
        print(f"   EEG shape: {paired_eeg.shape}")
        print(f"   Labels shape: {paired_labels.shape}")
        print(f"   Text features: {paired_text_features.shape}")
        print(f"   Image features: {paired_img_features.shape}")
        
        return {
            'eeg_data': paired_eeg,
            'image_paths': paired_images,
            'labels': paired_labels,
            'text_features': paired_text_features,
            'img_features': paired_img_features
        }
    
    def save_paired_dataset(self, paired_data):
        """Save paired dataset for CLIP training"""
        
        print(f"💾 Saving paired dataset...")
        
        mode = "train" if self.train else "test"
        
        # Save EEG data
        eeg_file = os.path.join(self.output_path, f"{mode}_eeg_data.npy")
        np.save(eeg_file, paired_data['eeg_data'].cpu().numpy())

        # Save labels
        labels_file = os.path.join(self.output_path, f"{mode}_labels.npy")
        np.save(labels_file, paired_data['labels'].cpu().numpy())

        # Save CLIP features
        text_feat_file = os.path.join(self.output_path, f"{mode}_text_features.npy")
        np.save(text_feat_file, paired_data['text_features'].cpu().numpy())

        img_feat_file = os.path.join(self.output_path, f"{mode}_img_features.npy")
        np.save(img_feat_file, paired_data['img_features'].cpu().numpy())
        
        # Save image paths
        img_paths_file = os.path.join(self.output_path, f"{mode}_image_paths.json")
        with open(img_paths_file, 'w') as f:
            json.dump(paired_data['image_paths'], f)
        
        # Save metadata
        metadata = {
            'mode': mode,
            'n_trials': len(paired_data['labels']),
            'n_classes': 10,
            'eeg_shape': list(paired_data['eeg_data'].shape),
            'time_window': self.time_window,
            'model_type': model_type.replace('/', '_'),
            'stimuli_path': self.stimuli_path
        }
        
        metadata_file = os.path.join(self.output_path, f"{mode}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved paired dataset to {self.output_path}")
        print(f"   Files: {mode}_eeg_data.npy, {mode}_labels.npy")
        print(f"   Features: {mode}_text_features.npy, {mode}_img_features.npy")
        print(f"   Metadata: {mode}_metadata.json")
        
        return self.output_path


def main():
    """Main function to create MindBigData retrieval dataset"""
    
    print("🚀 MINDBIGDATA RETRIEVAL PIPELINE")
    print("=" * 60)
    
    # Create train dataset
    print("\n📊 PROCESSING TRAIN DATASET:")
    train_retrieval = MindBigDataRetrieval(train=True)
    train_pairs = train_retrieval.create_eeg_image_pairs()
    train_output = train_retrieval.save_paired_dataset(train_pairs)
    
    # Create test dataset
    print("\n📊 PROCESSING TEST DATASET:")
    test_retrieval = MindBigDataRetrieval(train=False)
    test_pairs = test_retrieval.create_eeg_image_pairs()
    test_output = test_retrieval.save_paired_dataset(test_pairs)
    
    print(f"\n🎉 RETRIEVAL PIPELINE COMPLETE!")
    print(f"   Train pairs: {len(train_pairs['labels']):,}")
    print(f"   Test pairs: {len(test_pairs['labels']):,}")
    print(f"   Output directory: {train_output}")
    print(f"   Ready for CLIP training!")


if __name__ == "__main__":
    main()
