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
from .model_architecture import EEGImageCLIP, vae_loss_function, contrastive_loss

class EEGImageDataset(Dataset):
    """Dataset for EEG-Image pairs with TRUE subject-stimulus correspondence"""
    
    def __init__(self, eeg_data, metadata, stimuli_dir, transform=None):
        self.eeg_data = eeg_data
        self.metadata = metadata
        self.stimuli_dir = stimuli_dir
        self.transform = transform
        
        # Get available stimuli images
        self.stimuli_images = []
        for i in range(10):
            img_path = os.path.join(stimuli_dir, f"{i}.jpg")
            if os.path.exists(img_path):
                self.stimuli_images.append(img_path)
        
        print(f"Dataset: {len(self.eeg_data)} samples, {len(self.stimuli_images)} stimuli")
        
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
        
        return eeg_signal, image, true_stimulus_idx, image_path

def train_model(model, train_loader, val_loader, config, device, log_file=None):
    """Train EEG-Image CLIP model"""
    
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    train_losses = []
    val_losses = []
    vae_losses = []
    clip_losses = []
    
    def log_message(message):
        print(message)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
    
    log_message(f"Training on device: {device}")
    log_message(f"Training samples: {len(train_loader.dataset):,}")
    log_message(f"Validation samples: {len(val_loader.dataset):,}")
    log_message(f"Configuration: {config}")
    
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_vae_loss = 0.0
        train_clip_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        for batch_idx, (eeg_signals, images, stimulus_indices, image_paths) in enumerate(train_pbar):
            eeg_signals = eeg_signals.to(device)
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            eeg_clip_emb, image_emb, eeg_recon, mu, logvar, vae_emb = model(eeg_signals, images)
            
            # VAE loss
            vae_loss, recon_loss, kl_loss = vae_loss_function(
                eeg_recon, eeg_signals, mu, logvar, beta=config.get('vae_beta', 0.1)
            )
            
            # CLIP contrastive loss
            clip_loss = contrastive_loss(eeg_clip_emb, image_emb, model.temperature)
            
            # Combined loss
            total_loss = vae_loss + clip_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_vae_loss += vae_loss.item()
            train_clip_loss += clip_loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{total_loss.item():.3f}',
                'VAE': f'{vae_loss.item():.3f}',
                'CLIP': f'{clip_loss.item():.3f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
            for eeg_signals, images, stimulus_indices, image_paths in val_pbar:
                eeg_signals = eeg_signals.to(device)
                images = images.to(device)
                
                eeg_clip_emb, image_emb, eeg_recon, mu, logvar, vae_emb = model(eeg_signals, images)
                
                vae_loss, _, _ = vae_loss_function(
                    eeg_recon, eeg_signals, mu, logvar, beta=config.get('vae_beta', 0.1)
                )
                clip_loss = contrastive_loss(eeg_clip_emb, image_emb, model.temperature)
                
                total_loss = vae_loss + clip_loss
                val_loss += total_loss.item()
                
                val_pbar.set_postfix({'Val Loss': f'{total_loss.item():.3f}'})
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_vae_loss /= len(train_loader)
        train_clip_loss /= len(train_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        vae_losses.append(train_vae_loss)
        clip_losses.append(train_clip_loss)
        
        scheduler.step()
        
        elapsed = time.time() - start_time
        log_message(f"\nEpoch {epoch+1}/{config['num_epochs']} Summary:")
        log_message(f"  Total Loss - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        log_message(f"  VAE Loss: {train_vae_loss:.4f}")
        log_message(f"  CLIP Loss: {train_clip_loss:.4f}")
        log_message(f"  Temperature: {model.temperature.exp().item():.4f}")
        log_message(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        log_message(f"  Elapsed Time: {elapsed:.1f}s")
        log_message("-" * 70)
    
    total_time = time.time() - start_time
    log_message(f"Training completed in {total_time:.1f}s")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'vae_losses': vae_losses,
        'clip_losses': clip_losses,
        'training_time': total_time,
        'final_temperature': model.temperature.exp().item()
    }

def create_data_loaders(train_signals, train_metadata, test_signals, test_metadata, 
                       stimuli_dir, clip_preprocess, config):
    """Create data loaders for training"""
    
    # Create datasets
    train_dataset = EEGImageDataset(
        train_signals, train_metadata, stimuli_dir, clip_preprocess
    )
    test_dataset = EEGImageDataset(
        test_signals, test_metadata, stimuli_dir, clip_preprocess
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

def save_model(model, scaler, config, training_results, save_path):
    """Save trained model and metadata"""
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'config': config,
        'training_results': training_results,
        'model_architecture': {
            'eeg_input_dim': config['eeg_input_dim'],
            'vae_latent_dim': config['vae_latent_dim'],
            'clip_model': config.get('clip_model', 'ViT-B/32')
        }
    }, save_path)
    
    print(f"Model saved to: {save_path}")

def load_model(model_path, device='cpu'):
    """Load trained model"""
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Initialize model
    config = checkpoint['config']
    model = EEGImageCLIP(
        eeg_input_dim=config['eeg_input_dim'],
        vae_latent_dim=config['vae_latent_dim']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint['scaler'], checkpoint
