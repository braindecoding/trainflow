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
from .semantic_loss_model import SemanticLossEEGImageModel, combined_loss_function, adversarial_loss_function

class SemanticLossEEGImageDataset(Dataset):
    """Dataset for Semantic Loss approach with TRUE subject-stimulus correspondence"""
    
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
        
        print(f"Semantic Loss Dataset: {len(self.eeg_data)} samples, {len(self.stimuli_images)} stimuli")
        
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
        
        # Convert to tensor and normalize to [-1, 1] for generator
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image = (image - 0.5) * 2  # [0, 1] -> [-1, 1]
        
        return eeg_signal, image, true_stimulus_idx, image_path

def train_semantic_loss_model(model, train_loader, val_loader, config, device, log_file=None):
    """Train model with CLIP semantic loss approach"""
    
    model.to(device)
    
    # Separate optimizers for generator and discriminator
    optimizer_G = optim.Adam(
        list(model.eeg_encoder.parameters()) + list(model.image_generator.parameters()),
        lr=config['learning_rate'], betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        model.discriminator.parameters(),
        lr=config['learning_rate'], betas=(0.5, 0.999)
    )
    
    # Training history
    history = {
        'g_losses': [], 'd_losses': [], 'semantic_losses': [], 
        'reconstruction_losses': [], 'kl_losses': [], 'adversarial_losses': []
    }
    
    def log_message(message):
        print(message)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
    
    log_message(f"Training Semantic Loss Model on device: {device}")
    log_message(f"Configuration: {config}")
    
    for epoch in range(config['num_epochs']):
        model.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_semantic_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_adv_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, (eeg_signals, target_images, stimulus_indices, image_paths) in enumerate(pbar):
            eeg_signals = eeg_signals.to(device)
            target_images = target_images.to(device)
            batch_size = eeg_signals.size(0)
            
            # Real and fake labels
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)
            
            # =====================================
            # Train Discriminator
            # =====================================
            optimizer_D.zero_grad()
            
            # Real images
            real_output = model.discriminate(target_images)
            d_loss_real = adversarial_loss_function(real_output, real_labels)
            
            # Generated images
            generated_images, mu, logvar, z = model(eeg_signals)
            fake_output = model.discriminate(generated_images.detach())
            d_loss_fake = adversarial_loss_function(fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # =====================================
            # Train Generator
            # =====================================
            optimizer_G.zero_grad()
            
            # Get discriminator output for generated images
            fake_output = model.discriminate(generated_images)
            
            # Combined generator loss
            g_loss, loss_components = combined_loss_function(
                generated_images, target_images, mu, logvar,
                fake_output, real_labels, model.clip_model, config
            )
            
            g_loss.backward()
            optimizer_G.step()
            
            # Update metrics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_semantic_loss += loss_components['semantic']
            epoch_recon_loss += loss_components['reconstruction']
            epoch_kl_loss += loss_components['kl_divergence']
            epoch_adv_loss += loss_components['adversarial']
            
            # Update progress bar
            pbar.set_postfix({
                'G_Loss': f'{g_loss.item():.3f}',
                'D_Loss': f'{d_loss.item():.3f}',
                'Semantic': f'{loss_components["semantic"]:.3f}',
                'Recon': f'{loss_components["reconstruction"]:.3f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for eeg_signals, target_images, stimulus_indices, image_paths in val_loader:
                eeg_signals = eeg_signals.to(device)
                target_images = target_images.to(device)
                
                generated_images, mu, logvar, z = model(eeg_signals)
                fake_output = model.discriminate(generated_images)
                
                g_loss, _ = combined_loss_function(
                    generated_images, target_images, mu, logvar,
                    fake_output, torch.ones(eeg_signals.size(0)).to(device), 
                    model.clip_model, config
                )
                
                val_loss += g_loss.item()
        
        # Calculate epoch averages
        num_batches = len(train_loader)
        epoch_g_loss /= num_batches
        epoch_d_loss /= num_batches
        epoch_semantic_loss /= num_batches
        epoch_recon_loss /= num_batches
        epoch_kl_loss /= num_batches
        epoch_adv_loss /= num_batches
        val_loss /= len(val_loader)
        
        # Store history
        history['g_losses'].append(epoch_g_loss)
        history['d_losses'].append(epoch_d_loss)
        history['semantic_losses'].append(epoch_semantic_loss)
        history['reconstruction_losses'].append(epoch_recon_loss)
        history['kl_losses'].append(epoch_kl_loss)
        history['adversarial_losses'].append(epoch_adv_loss)
        
        log_message(f"\nEpoch {epoch+1}/{config['num_epochs']} Summary:")
        log_message(f"  Generator Loss: {epoch_g_loss:.4f}")
        log_message(f"  Discriminator Loss: {epoch_d_loss:.4f}")
        log_message(f"  Semantic Loss: {epoch_semantic_loss:.4f}")
        log_message(f"  Reconstruction Loss: {epoch_recon_loss:.4f}")
        log_message(f"  KL Loss: {epoch_kl_loss:.4f}")
        log_message(f"  Adversarial Loss: {epoch_adv_loss:.4f}")
        log_message(f"  Validation Loss: {val_loss:.4f}")
        log_message("-" * 70)
    
    return history

def create_semantic_data_loaders(train_signals, train_metadata, test_signals, test_metadata, 
                                stimuli_dir, config):
    """Create data loaders for semantic loss training"""
    
    # Create transform for images
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    train_dataset = SemanticLossEEGImageDataset(
        train_signals, train_metadata, stimuli_dir, transform
    )
    test_dataset = SemanticLossEEGImageDataset(
        test_signals, test_metadata, stimuli_dir, transform
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

def save_semantic_model(model, scaler, config, training_history, save_path):
    """Save trained semantic loss model"""
    
    torch.save({
        'generator_state_dict': model.image_generator.state_dict(),
        'encoder_state_dict': model.eeg_encoder.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'scaler': scaler,
        'config': config,
        'training_history': training_history,
        'model_type': 'semantic_loss_approach'
    }, save_path)
    
    print(f"Semantic Loss Model saved to: {save_path}")

def load_semantic_model(model_path, eeg_input_dim, device='cpu'):
    """Load trained semantic loss model"""
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Initialize model
    config = checkpoint['config']
    model = SemanticLossEEGImageModel(
        eeg_input_dim=eeg_input_dim,
        latent_dim=config.get('latent_dim', 128)
    )
    
    # Load state dicts
    model.eeg_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.image_generator.load_state_dict(checkpoint['generator_state_dict'])
    model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model, checkpoint['scaler'], checkpoint
