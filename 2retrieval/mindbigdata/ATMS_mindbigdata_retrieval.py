#!/usr/bin/env python3
"""
ATMS MindBigData Retrieval
=========================

Adapted ATMS (Attention-based Transformer Multi-Subject) model for MindBigData digit recognition.
Original: THINGS dataset with complex natural images
Adapted: MindBigData with digit stimuli (0-9)
"""

import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
import random
import csv
from torch import Tensor
import itertools
import math
import re
import datetime
import argparse
import json

# Import our MindBigData components
from mindbigdata_retrieval import MindBigDataRetrieval

# Setup device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class Config:
    def __init__(self):
        self.task_name = 'classification'
        self.seq_len = 256                 # MindBigData: 256 timepoints (2 seconds at 128Hz)
        self.pred_len = 256
        self.output_attention = False
        self.d_model = 256                 # Adjusted for MindBigData
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.25
        self.factor = 1
        self.n_heads = 4
        self.e_layers = 1
        self.d_ff = 256
        self.activation = 'gelu'
        self.enc_in = 14                   # MindBigData: 14 EPOC channels

class MindBigDataDataset(Dataset):
    """
    MindBigData Dataset for ATMS training
    
    Loads preprocessed EEG-image pairs with CLIP features
    """
    
    def __init__(self, data_path=r"d:\trainflow\2retrieval\outputs\mindbigdata_pairs", 
                 train=True, time_window=[0, 2.0]):
        
        self.data_path = data_path
        self.train = train
        self.time_window = time_window
        
        print(f"🧠 Loading MindBigData Dataset...")
        print(f"   Mode: {'Train' if train else 'Test'}")
        print(f"   Path: {data_path}")
        
        # Load data
        self.load_data()
        
        print(f"✅ Dataset loaded:")
        print(f"   EEG data: {self.eeg_data.shape}")
        print(f"   Labels: {self.labels.shape}")
        print(f"   Text features: {self.text_features.shape}")
        print(f"   Image features: {self.img_features.shape}")
    
    def load_data(self):
        """Load preprocessed MindBigData"""
        
        mode = "train" if self.train else "test"
        
        # Load EEG data
        eeg_file = os.path.join(self.data_path, f"{mode}_eeg_data.npy")
        self.eeg_data = torch.from_numpy(np.load(eeg_file)).float()
        
        # Load labels
        labels_file = os.path.join(self.data_path, f"{mode}_labels.npy")
        self.labels = torch.from_numpy(np.load(labels_file)).long()
        
        # Load CLIP features
        text_feat_file = os.path.join(self.data_path, f"{mode}_text_features.npy")
        self.text_features = torch.from_numpy(np.load(text_feat_file)).float()
        
        img_feat_file = os.path.join(self.data_path, f"{mode}_img_features.npy")
        self.img_features = torch.from_numpy(np.load(img_feat_file)).float()
        
        # Load metadata
        metadata_file = os.path.join(self.data_path, f"{mode}_metadata.json")
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Create unique CLIP features for each digit (10 classes)
        unique_labels = torch.unique(self.labels)
        self.text_features_all = torch.zeros(len(unique_labels), self.text_features.shape[1])
        self.img_features_all = torch.zeros(len(unique_labels), self.img_features.shape[1])
        
        for i, label in enumerate(unique_labels):
            # Get first occurrence of each digit for unique features
            idx = (self.labels == label).nonzero(as_tuple=True)[0][0]
            self.text_features_all[i] = self.text_features[idx]
            self.img_features_all[i] = self.img_features[idx]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        eeg_data = self.eeg_data[idx]  # (14, 256)
        label = self.labels[idx]
        text_feature = self.text_features[idx]  # (512,)
        img_feature = self.img_features[idx]    # (512,)
        
        # For compatibility with original ATMS format
        text = f"This picture is digit {label.item()}"
        img = f"digit_{label.item()}.jpg"
        
        return eeg_data, label, text, text_feature, img, img_feature

class PatchEmbedding(nn.Module):
    """Adapted for MindBigData EEG format"""
    def __init__(self, emb_size=40):
        super().__init__()
        # Adapted for MindBigData: 14 channels, 256 timepoints
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (14, 1), stride=(1, 1)),  # 14 channels for EPOC
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)     
        x = self.tsconv(x)
        x = self.projection(x)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1480, proj_dim=512, drop_proj=0.5):  # proj_dim=512 for CLIP
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class ATMS_MindBigData(nn.Module):    
    """ATMS model adapted for MindBigData"""
    def __init__(self, num_channels=14, sequence_length=256, num_subjects=1, num_features=64, num_latents=512, num_blocks=1):
        super(ATMS_MindBigData, self).__init__()
        
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg(embedding_dim=1480, proj_dim=512)  # Project to CLIP dimension
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # Simple loss function for MindBigData
        self.loss_func = nn.CrossEntropyLoss()
         
    def forward(self, x, subject_ids=None):
        # x shape: (batch_size, 14, 256)
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out  # (batch_size, 512)
    
    def clip_loss(self, eeg_features, target_features, logit_scale):
        """CLIP-style contrastive loss"""
        # Normalize features
        eeg_features = F.normalize(eeg_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)
        
        # Compute similarity
        logits = logit_scale * eeg_features @ target_features.T
        
        # Create labels (diagonal should be positive)
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        # Symmetric loss
        loss_eeg = F.cross_entropy(logits, labels)
        loss_target = F.cross_entropy(logits.T, labels)
        
        return (loss_eeg + loss_target) / 2

def train_model(eeg_model, dataloader, optimizer, device, text_features_all, img_features_all, config):
    eeg_model.train()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99  # Weight for image vs text loss
    
    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        eeg_features = eeg_model(eeg_data).float()
        logit_scale = eeg_model.logit_scale
        
        # Compute losses
        img_loss = eeg_model.clip_loss(eeg_features, img_features, logit_scale)
        text_loss = eeg_model.clip_loss(eeg_features, text_features, logit_scale)
        loss = alpha * img_loss + (1 - alpha) * text_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Compute accuracy
        logits_img = logit_scale * eeg_features @ img_features_all.T
        predicted = torch.argmax(logits_img, dim=1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    average_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return average_loss, accuracy

def evaluate_model(eeg_model, dataloader, device, text_features_all, img_features_all, k=10, config=None):
    eeg_model.eval()
    
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    
    total_loss = 0
    correct = 0
    total = 0
    top5_correct = 0
    alpha = 0.99
    
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            img_features = img_features.to(device).float()
            labels = labels.to(device)
            
            eeg_features = eeg_model(eeg_data)
            logit_scale = eeg_model.logit_scale
            
            # Compute loss
            img_loss = eeg_model.clip_loss(eeg_features, img_features, logit_scale)
            text_loss = eeg_model.clip_loss(eeg_features, text_features, logit_scale)
            loss = alpha * img_loss + (1 - alpha) * text_loss
            total_loss += loss.item()
            
            # Compute accuracy
            logits_img = logit_scale * eeg_features @ img_features_all.T
            predicted = torch.argmax(logits_img, dim=1)
            
            correct += (predicted == labels).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = torch.topk(logits_img, 5, dim=1)
            top5_correct += sum([labels[i] in top5_pred[i] for i in range(len(labels))])
            
            total += labels.size(0)
    
    average_loss = total_loss / len(dataloader)
    accuracy = correct / total
    top5_acc = top5_correct / total
    
    return average_loss, accuracy, top5_acc

def main():
    parser = argparse.ArgumentParser(description='ATMS MindBigData Training Script')
    parser.add_argument('--data_path', type=str, 
                       default=r"d:\trainflow\2retrieval\outputs\mindbigdata_pairs", 
                       help='Path to MindBigData paired dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/atms_mindbigdata', 
                       help='Directory to save output results')    
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU device to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', 
                       help='Device to run on (cpu or gpu)')    
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.gpu)
    else:
        device = torch.device('cpu')
    
    print(f"🚀 ATMS MINDBIGDATA TRAINING")
    print("=" * 60)
    print(f"   Device: {device}")
    print(f"   Data path: {args.data_path}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    
    # Initialize model
    eeg_model = ATMS_MindBigData()
    eeg_model.to(device)
    
    optimizer = AdamW(eeg_model.parameters(), lr=args.lr)
    
    # Load datasets
    train_dataset = MindBigDataDataset(args.data_path, train=True)
    test_dataset = MindBigDataDataset(args.data_path, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=0, drop_last=False)
    
    # Get all features for evaluation
    text_features_all = train_dataset.text_features_all
    img_features_all = train_dataset.img_features_all
    
    # Training loop
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    best_accuracy = 0.0
    
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_accuracy = train_model(eeg_model, train_loader, optimizer, device, 
                                                text_features_all, img_features_all, args)
        
        # Evaluate
        test_loss, test_accuracy, top5_acc = evaluate_model(eeg_model, test_loader, device, 
                                                          text_features_all, img_features_all, 
                                                          k=10, config=args)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(eeg_model.state_dict(), 
                      os.path.join(args.output_dir, f'best_model_{current_time}.pth'))
        
        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, "
              f"Top5 Acc: {top5_acc:.4f}")
    
    print(f"\n🎉 Training completed!")
    print(f"   Best accuracy: {best_accuracy:.4f}")
    print(f"   Model saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
