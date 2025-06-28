import torch
import torch.nn as nn
import numpy as np
import clip

class EEG_VAE(nn.Module):
    """Variational Autoencoder for EEG signals"""
    
    def __init__(self, input_dim, latent_dim=128, hidden_dims=[512, 256]):
        super(EEG_VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
    
    def get_latent_embedding(self, x):
        """Get latent embedding for downstream tasks"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

class EEGImageCLIP(nn.Module):
    """Complete EEG-VAE-CLIP model for image reconstruction"""
    
    def __init__(self, eeg_input_dim, vae_latent_dim=128, clip_model_name="ViT-B/32"):
        super(EEGImageCLIP, self).__init__()
        
        # Load pre-trained CLIP model
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # EEG VAE
        self.eeg_vae = EEG_VAE(eeg_input_dim, vae_latent_dim)
        
        # Projection from VAE latent to CLIP embedding space
        self.vae_to_clip = nn.Sequential(
            nn.Linear(vae_latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512)  # CLIP embedding dimension
        )
        
        # Image projection
        self.image_projection = nn.Linear(512, 512)
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_eeg(self, eeg_signals):
        """Encode EEG signals through VAE to CLIP embedding space"""
        # Get VAE latent embedding
        vae_embedding = self.eeg_vae.get_latent_embedding(eeg_signals)
        
        # Project to CLIP space
        clip_embedding = self.vae_to_clip(vae_embedding)
        
        # Normalize
        clip_embedding = clip_embedding / clip_embedding.norm(dim=-1, keepdim=True)
        
        return clip_embedding, vae_embedding
    
    def encode_image(self, images):
        """Encode images using CLIP"""
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
        
        image_embeddings = self.image_projection(image_features.float())
        # Normalize embeddings
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        return image_embeddings
    
    def forward(self, eeg_signals, images=None):
        # EEG encoding through VAE
        eeg_recon, mu, logvar, vae_z = self.eeg_vae(eeg_signals)
        
        # Get CLIP embeddings
        eeg_clip_embedding, vae_embedding = self.encode_eeg(eeg_signals)
        
        if images is not None:
            image_embeddings = self.encode_image(images)
            return eeg_clip_embedding, image_embeddings, eeg_recon, mu, logvar, vae_embedding
        else:
            return eeg_clip_embedding, vae_embedding, eeg_recon, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function with reconstruction and KL divergence"""
    # Reconstruction loss
    recon_loss = nn.MSELoss()(recon_x, x)
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.size(0)  # Average over batch
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def contrastive_loss(eeg_embeddings, image_embeddings, temperature):
    """Contrastive loss for CLIP-style training"""
    logits = torch.matmul(eeg_embeddings, image_embeddings.T) * temperature.exp()
    batch_size = eeg_embeddings.shape[0]
    labels = torch.arange(batch_size).to(eeg_embeddings.device)
    
    loss_eeg_to_image = nn.CrossEntropyLoss()(logits, labels)
    loss_image_to_eeg = nn.CrossEntropyLoss()(logits.T, labels)
    
    return (loss_eeg_to_image + loss_image_to_eeg) / 2
