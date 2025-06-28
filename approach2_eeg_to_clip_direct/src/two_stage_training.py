import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import time
from tqdm import tqdm
import json
from .eeg_to_clip_model import EEGToCLIPModel, combined_embedding_loss, EmbeddingQualityMetrics
from .clip_to_image_model import CLIPToImageModel, TwoStageEEGToImageModel

class TwoStageEEGImageDataset(Dataset):
    """Dataset for two-stage training with TRUE subject-stimulus correspondence"""
    
    def __init__(self, eeg_data, metadata, stimuli_dir, transform=None, stage='stage1'):
        self.eeg_data = eeg_data
        self.metadata = metadata
        self.stimuli_dir = stimuli_dir
        self.transform = transform
        self.stage = stage
        
        # Get available stimuli images
        self.stimuli_images = []
        for i in range(10):
            img_path = os.path.join(stimuli_dir, f"{i}.jpg")
            if os.path.exists(img_path):
                self.stimuli_images.append(img_path)
        
        print(f"Two-Stage Dataset ({stage}): {len(self.eeg_data)} samples, {len(self.stimuli_images)} stimuli")
        
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        # Get EEG signal
        eeg_signal = torch.FloatTensor(self.eeg_data[idx])
        
        # Get TRUE ground truth based on subject
        subject = self.metadata[idx]['subject']
        true_stimulus_idx = subject % len(self.stimuli_images)
        image_path = self.stimuli_images[true_stimulus_idx]
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return eeg_signal, image, true_stimulus_idx, image_path

def train_stage1_eeg_to_clip(model, train_loader, val_loader, config, device, log_file=None):
    """Train Stage 1: EEG to CLIP embedding prediction"""
    
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.eeg_encoder.parameters(), 
        lr=config['stage1_lr'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['stage1_epochs'])
    
    # Training history
    history = {
        'train_losses': [], 'val_losses': [], 'cosine_similarities': [],
        'l2_distances': [], 'embedding_qualities': []
    }
    
    # Metrics tracker
    metrics_tracker = EmbeddingQualityMetrics()
    
    def log_message(message):
        print(message)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
    
    log_message(f"Training Stage 1 (EEG-to-CLIP) on device: {device}")
    log_message(f"Configuration: {config}")
    
    for epoch in range(config['stage1_epochs']):
        model.train()
        
        epoch_train_loss = 0.0
        metrics_tracker.reset()
        
        pbar = tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{config['stage1_epochs']}")
        
        for batch_idx, (eeg_signals, target_images, stimulus_indices, image_paths) in enumerate(pbar):
            eeg_signals = eeg_signals.to(device)
            target_images = target_images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_clip_embeddings = model.encode_eeg_to_clip(eeg_signals)
            target_clip_embeddings = model.encode_image_to_clip(target_images)
            
            # Compute loss
            loss, loss_components = combined_embedding_loss(
                predicted_clip_embeddings, 
                target_clip_embeddings,
                cosine_weight=config.get('cosine_weight', 0.7),
                l2_weight=config.get('l2_weight', 0.3)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_train_loss += loss.item()
            metrics_tracker.update(predicted_clip_embeddings, target_clip_embeddings)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Cosine': f'{loss_components["cosine"]:.4f}',
                'L2': f'{loss_components["l2"]:.4f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics_tracker = EmbeddingQualityMetrics()
        
        with torch.no_grad():
            for eeg_signals, target_images, stimulus_indices, image_paths in val_loader:
                eeg_signals = eeg_signals.to(device)
                target_images = target_images.to(device)
                
                predicted_clip_embeddings = model.encode_eeg_to_clip(eeg_signals)
                target_clip_embeddings = model.encode_image_to_clip(target_images)
                
                loss, _ = combined_embedding_loss(
                    predicted_clip_embeddings, target_clip_embeddings
                )
                
                val_loss += loss.item()
                val_metrics_tracker.update(predicted_clip_embeddings, target_clip_embeddings)
        
        # Calculate epoch averages
        epoch_train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Get metrics
        train_metrics = metrics_tracker.get_average_metrics()
        val_metrics = val_metrics_tracker.get_average_metrics()
        
        # Store history
        history['train_losses'].append(epoch_train_loss)
        history['val_losses'].append(val_loss)
        history['cosine_similarities'].append(train_metrics['avg_cosine_similarity'])
        history['l2_distances'].append(train_metrics['avg_l2_distance'])
        history['embedding_qualities'].append(train_metrics)
        
        # Update learning rate
        scheduler.step()
        
        log_message(f"\nStage 1 Epoch {epoch+1}/{config['stage1_epochs']} Summary:")
        log_message(f"  Train Loss: {epoch_train_loss:.4f}")
        log_message(f"  Val Loss: {val_loss:.4f}")
        log_message(f"  Train Cosine Similarity: {train_metrics['avg_cosine_similarity']:.4f}")
        log_message(f"  Val Cosine Similarity: {val_metrics['avg_cosine_similarity']:.4f}")
        log_message(f"  Train L2 Distance: {train_metrics['avg_l2_distance']:.4f}")
        log_message(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        log_message("-" * 70)
    
    return history

def train_stage2_clip_to_image(stage1_model, stage2_model, train_loader, val_loader, 
                              config, device, log_file=None):
    """Train Stage 2: CLIP embedding to image generation"""
    
    stage1_model.eval()  # Freeze stage 1
    stage2_model.to(device)
    
    # Optimizer for stage 2 only
    optimizer = optim.Adam(
        stage2_model.parameters(), 
        lr=config['stage2_lr'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Loss function
    reconstruction_loss = nn.L1Loss()
    
    # Training history
    history = {
        'train_losses': [], 'val_losses': [], 'reconstruction_losses': []
    }
    
    def log_message(message):
        print(message)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
    
    log_message(f"Training Stage 2 (CLIP-to-Image) on device: {device}")
    
    for epoch in range(config['stage2_epochs']):
        stage2_model.train()
        
        epoch_train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Stage 2 Epoch {epoch+1}/{config['stage2_epochs']}")
        
        for batch_idx, (eeg_signals, target_images, stimulus_indices, image_paths) in enumerate(pbar):
            eeg_signals = eeg_signals.to(device)
            target_images = target_images.to(device)
            
            # Normalize target images to [-1, 1] for decoder
            target_images = (target_images - 0.5) * 2
            
            optimizer.zero_grad()
            
            # Get CLIP embeddings from stage 1 (frozen)
            with torch.no_grad():
                clip_embeddings = stage1_model.encode_eeg_to_clip(eeg_signals)
            
            # Generate images from CLIP embeddings
            generated_images = stage2_model(clip_embeddings)
            
            # Compute reconstruction loss
            loss = reconstruction_loss(generated_images, target_images)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Recon Loss': f'{loss.item():.4f}'
            })
        
        # Validation phase
        stage2_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for eeg_signals, target_images, stimulus_indices, image_paths in val_loader:
                eeg_signals = eeg_signals.to(device)
                target_images = target_images.to(device)
                target_images = (target_images - 0.5) * 2
                
                clip_embeddings = stage1_model.encode_eeg_to_clip(eeg_signals)
                generated_images = stage2_model(clip_embeddings)
                
                loss = reconstruction_loss(generated_images, target_images)
                val_loss += loss.item()
        
        # Calculate epoch averages
        epoch_train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Store history
        history['train_losses'].append(epoch_train_loss)
        history['val_losses'].append(val_loss)
        history['reconstruction_losses'].append(epoch_train_loss)
        
        log_message(f"\nStage 2 Epoch {epoch+1}/{config['stage2_epochs']} Summary:")
        log_message(f"  Train Reconstruction Loss: {epoch_train_loss:.4f}")
        log_message(f"  Val Reconstruction Loss: {val_loss:.4f}")
        log_message("-" * 70)
    
    return history

def create_two_stage_data_loaders(train_signals, train_metadata, test_signals, test_metadata, 
                                 stimuli_dir, config):
    """Create data loaders for two-stage training"""
    
    # Create transform for images
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    train_dataset = TwoStageEEGImageDataset(
        train_signals, train_metadata, stimuli_dir, transform, 'training'
    )
    test_dataset = TwoStageEEGImageDataset(
        test_signals, test_metadata, stimuli_dir, transform, 'testing'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    val_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    return train_loader, val_loader
