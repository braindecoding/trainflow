import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip

class EEG_VAE_Encoder(nn.Module):
    """VAE Encoder for EEG signals"""
    
    def __init__(self, eeg_input_dim, latent_dim=128):
        super(EEG_VAE_Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(eeg_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # VAE latent parameters
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class ImageGenerator(nn.Module):
    """Generator that creates images from latent vectors"""
    
    def __init__(self, latent_dim=128, image_size=224):
        super(ImageGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Calculate initial feature map size
        self.init_size = image_size // 32  # 224 // 32 = 7
        
        # Linear layer to expand latent vector
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size * self.init_size),
            nn.ReLU()
        )
        
        # Convolutional layers for upsampling
        self.conv_blocks = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 112x112 -> 224x224
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, z):
        # Expand latent vector
        out = self.fc(z)
        out = out.view(out.size(0), 512, self.init_size, self.init_size)
        
        # Generate image
        img = self.conv_blocks(out)
        return img

class ImageDiscriminator(nn.Module):
    """Discriminator for adversarial training"""
    
    def __init__(self, image_size=224):
        super(ImageDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            # 112x112 -> 56x56
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 56x56 -> 28x28
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 28x28 -> 14x14
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 14x14 -> 7x7
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 7x7 -> 1x1
            nn.Conv2d(512, 1, 7, 1, 0),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1)

class SemanticLossEEGImageModel(nn.Module):
    """Complete model with CLIP semantic loss"""
    
    def __init__(self, eeg_input_dim, latent_dim=128, clip_model_name="ViT-B/32"):
        super(SemanticLossEEGImageModel, self).__init__()
        
        # Load pre-trained CLIP model (frozen)
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Model components
        self.eeg_encoder = EEG_VAE_Encoder(eeg_input_dim, latent_dim)
        self.image_generator = ImageGenerator(latent_dim)
        self.discriminator = ImageDiscriminator()
        
    def encode_eeg(self, eeg_signals):
        """Encode EEG to latent space"""
        mu, logvar = self.eeg_encoder(eeg_signals)
        z = self.eeg_encoder.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def generate_image(self, z):
        """Generate image from latent vector"""
        return self.image_generator(z)
    
    def discriminate(self, images):
        """Discriminate real vs fake images"""
        return self.discriminator(images)
    
    def encode_image_with_clip(self, images):
        """Encode images using frozen CLIP"""
        with torch.no_grad():
            # Normalize images from [-1, 1] to [0, 1] for CLIP
            normalized_images = (images + 1) / 2
            clip_embeddings = self.clip_model.encode_image(normalized_images)
            return clip_embeddings.float()
    
    def forward(self, eeg_signals):
        """Forward pass: EEG -> latent -> image"""
        z, mu, logvar = self.encode_eeg(eeg_signals)
        generated_image = self.generate_image(z)
        return generated_image, mu, logvar, z

# Loss Functions
def semantic_loss_function(generated_images, target_images, clip_model):
    """CLIP-based semantic loss"""
    with torch.no_grad():
        # Normalize to [0, 1] for CLIP
        gen_normalized = (generated_images + 1) / 2
        target_normalized = (target_images + 1) / 2
        
        # Get CLIP embeddings
        gen_embeddings = clip_model.encode_image(gen_normalized).float()
        target_embeddings = clip_model.encode_image(target_normalized).float()
    
    # Cosine similarity loss
    cosine_sim = F.cosine_similarity(gen_embeddings, target_embeddings, dim=1)
    semantic_loss = 1 - cosine_sim.mean()
    
    return semantic_loss

def vae_loss_function(mu, logvar, beta=0.1):
    """VAE KL divergence loss"""
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / mu.size(0)  # Average over batch
    return beta * kl_loss

def adversarial_loss_function(output, target):
    """Binary cross entropy loss for GAN"""
    return F.binary_cross_entropy(output, target)

def reconstruction_loss_function(generated_images, target_images):
    """L1 reconstruction loss"""
    return F.l1_loss(generated_images, target_images)

def combined_loss_function(generated_images, target_images, mu, logvar, 
                          discriminator_output, real_labels, clip_model, config):
    """Combined loss function for generator"""
    
    # Adversarial loss
    adv_loss = adversarial_loss_function(discriminator_output, real_labels)
    
    # Reconstruction loss
    recon_loss = reconstruction_loss_function(generated_images, target_images)
    
    # VAE KL loss
    kl_loss = vae_loss_function(mu, logvar, config.get('vae_beta', 0.1))
    
    # CLIP Semantic loss
    semantic_loss = semantic_loss_function(generated_images, target_images, clip_model)
    
    # Combined loss
    total_loss = (config.get('lambda_adv', 1.0) * adv_loss + 
                  config.get('lambda_rec', 10.0) * recon_loss + 
                  config.get('lambda_kl', 1.0) * kl_loss + 
                  config.get('lambda_clip', 5.0) * semantic_loss)
    
    return total_loss, {
        'adversarial': adv_loss.item(),
        'reconstruction': recon_loss.item(),
        'kl_divergence': kl_loss.item(),
        'semantic': semantic_loss.item(),
        'total': total_loss.item()
    }
