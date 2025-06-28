import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class SimpleCLIPToImageDecoder(nn.Module):
    """Simple decoder from CLIP embedding to image"""
    
    def __init__(self, clip_embedding_dim=512, image_size=224):
        super(SimpleCLIPToImageDecoder, self).__init__()
        
        self.clip_embedding_dim = clip_embedding_dim
        self.image_size = image_size
        
        # Calculate initial feature map size
        self.init_size = image_size // 32  # 224 // 32 = 7
        
        # Linear layer to expand CLIP embedding
        self.fc = nn.Sequential(
            nn.Linear(clip_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512 * self.init_size * self.init_size),
            nn.ReLU()
        )
        
        # Convolutional decoder
        self.decoder = nn.Sequential(
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
        
    def forward(self, clip_embeddings):
        """Generate image from CLIP embedding"""
        # Expand embedding
        x = self.fc(clip_embeddings)
        x = x.view(x.size(0), 512, self.init_size, self.init_size)
        
        # Decode to image
        image = self.decoder(x)
        return image

class CLIPToImageModel(nn.Module):
    """Complete model for Stage 2: CLIP embedding to image generation"""
    
    def __init__(self, clip_embedding_dim=512, decoder_type='simple', image_size=224):
        super(CLIPToImageModel, self).__init__()
        
        self.clip_embedding_dim = clip_embedding_dim
        self.decoder_type = decoder_type
        self.image_size = image_size
        
        if decoder_type == 'simple':
            self.decoder = SimpleCLIPToImageDecoder(clip_embedding_dim, image_size)
        else:
            raise ValueError(f"Decoder type '{decoder_type}' not implemented")
    
    def generate_image(self, clip_embeddings):
        """Generate image from CLIP embeddings"""
        return self.decoder(clip_embeddings)
    
    def forward(self, clip_embeddings):
        """Forward pass for training"""
        return self.generate_image(clip_embeddings)

class TwoStageEEGToImageModel(nn.Module):
    """Complete two-stage model: EEG → CLIP → Image"""
    
    def __init__(self, stage1_model, stage2_model):
        super(TwoStageEEGToImageModel, self).__init__()
        
        self.stage1_model = stage1_model  # EEG-to-CLIP
        self.stage2_model = stage2_model  # CLIP-to-Image
        
        # Freeze stage 1 if needed
        self.freeze_stage1()
    
    def freeze_stage1(self):
        """Freeze stage 1 parameters"""
        for param in self.stage1_model.parameters():
            param.requires_grad = False
    
    def unfreeze_stage1(self):
        """Unfreeze stage 1 parameters for joint training"""
        for param in self.stage1_model.parameters():
            param.requires_grad = True
    
    def forward(self, eeg_signals):
        """Complete forward pass: EEG → CLIP → Image"""
        # Stage 1: EEG to CLIP embedding
        clip_embeddings = self.stage1_model.encode_eeg_to_clip(eeg_signals)
        
        # Stage 2: CLIP embedding to image
        generated_images = self.stage2_model.generate_image(clip_embeddings)
        
        return generated_images, clip_embeddings

# Training functions for Stage 2
def train_stage2_decoder(stage2_model, clip_embeddings, target_images, config, device):
    """Train Stage 2 decoder (CLIP → Image)"""
    
    stage2_model.to(device)
    optimizer = torch.optim.Adam(stage2_model.parameters(), lr=config['stage2_lr'])
    
    # Loss function
    reconstruction_loss = nn.L1Loss()
    
    stage2_model.train()
    
    # Forward pass
    generated_images = stage2_model(clip_embeddings)
    
    # Compute loss
    loss = reconstruction_loss(generated_images, target_images)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate_stage2_quality(stage2_model, clip_embeddings, target_images, device):
    """Evaluate Stage 2 image generation quality"""
    
    stage2_model.eval()
    
    with torch.no_grad():
        generated_images = stage2_model(clip_embeddings)
        
        # Compute metrics
        l1_loss = F.l1_loss(generated_images, target_images).item()
        mse_loss = F.mse_loss(generated_images, target_images).item()
        
        # PSNR
        psnr = 20 * torch.log10(2.0 / torch.sqrt(F.mse_loss(generated_images, target_images)))
        
        return {
            'l1_loss': l1_loss,
            'mse_loss': mse_loss,
            'psnr': psnr.item()
        }

# Utility functions for image generation
def clip_embedding_to_image(clip_embedding, stage2_model, device):
    """Convert single CLIP embedding to image"""
    
    stage2_model.eval()
    
    with torch.no_grad():
        if len(clip_embedding.shape) == 1:
            clip_embedding = clip_embedding.unsqueeze(0)
        
        clip_embedding = clip_embedding.to(device)
        generated_image = stage2_model(clip_embedding)
        
        # Convert from [-1, 1] to [0, 1]
        generated_image = (generated_image + 1) / 2
        generated_image = torch.clamp(generated_image, 0, 1)
        
        return generated_image.squeeze(0)

def save_generated_image(image_tensor, save_path):
    """Save generated image tensor to file"""
    
    # Convert tensor to PIL Image
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.permute(1, 2, 0)  # CHW -> HWC
    
    image_array = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save(save_path)

def create_stage2_model(clip_embedding_dim, config):
    """Factory function to create Stage 2 model"""
    
    model = CLIPToImageModel(
        clip_embedding_dim=clip_embedding_dim,
        decoder_type=config.get('decoder_type', 'simple'),
        image_size=config.get('image_size', 224)
    )
    
    return model

def save_stage2_model(model, config, training_history, save_path):
    """Save Stage 2 model (CLIP-to-Image)"""
    
    torch.save({
        'decoder_state_dict': model.decoder.state_dict(),
        'config': config,
        'training_history': training_history,
        'model_type': 'clip_to_image_stage2',
        'clip_embedding_dim': model.clip_embedding_dim,
        'decoder_type': model.decoder_type
    }, save_path)
    
    print(f"Stage 2 Model (CLIP-to-Image) saved to: {save_path}")

def load_stage2_model(model_path, device='cpu'):
    """Load Stage 2 model (CLIP-to-Image)"""
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Initialize model
    config = checkpoint['config']
    model = CLIPToImageModel(
        clip_embedding_dim=checkpoint['clip_embedding_dim'],
        decoder_type=checkpoint['decoder_type'],
        image_size=config.get('image_size', 224)
    )
    
    # Load decoder state
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model, checkpoint

def create_two_stage_model(stage1_model, stage2_model):
    """Create complete two-stage model"""
    
    return TwoStageEEGToImageModel(stage1_model, stage2_model)

# Advanced decoder options (for future implementation)
class VQGANCLIPDecoder(nn.Module):
    """VQGAN-based decoder (placeholder for future implementation)"""
    
    def __init__(self, clip_embedding_dim=512):
        super(VQGANCLIPDecoder, self).__init__()
        # TODO: Implement VQGAN decoder
        raise NotImplementedError("VQGAN decoder not yet implemented")

class StyleGANCLIPDecoder(nn.Module):
    """StyleGAN-based decoder (placeholder for future implementation)"""
    
    def __init__(self, clip_embedding_dim=512):
        super(StyleGANCLIPDecoder, self).__init__()
        # TODO: Implement StyleGAN decoder
        raise NotImplementedError("StyleGAN decoder not yet implemented")
