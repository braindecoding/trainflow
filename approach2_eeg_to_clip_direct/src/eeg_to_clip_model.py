import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip

class EEGToCLIPEncoder(nn.Module):
    """Direct EEG to CLIP embedding encoder"""
    
    def __init__(self, eeg_input_dim, clip_embedding_dim=512, hidden_dims=[512, 256, 128]):
        super(EEGToCLIPEncoder, self).__init__()
        
        self.eeg_input_dim = eeg_input_dim
        self.clip_embedding_dim = clip_embedding_dim
        
        # Build encoder layers
        layers = []
        prev_dim = eeg_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Final projection to CLIP embedding space
        layers.append(nn.Linear(prev_dim, clip_embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, eeg_signals):
        """Forward pass: EEG → CLIP embedding"""
        # Encode EEG to embedding space
        embedding = self.encoder(eeg_signals)
        
        # Normalize to unit sphere (like CLIP embeddings)
        normalized_embedding = F.normalize(embedding, dim=1)
        
        return normalized_embedding

class EEGToCLIPModel(nn.Module):
    """Complete model for Stage 1: EEG to CLIP embedding prediction"""
    
    def __init__(self, eeg_input_dim, clip_model_name="ViT-B/32", hidden_dims=[512, 256, 128]):
        super(EEGToCLIPModel, self).__init__()
        
        # Load pre-trained CLIP model (frozen)
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Get CLIP embedding dimension
        with torch.no_grad():
            dummy_image = torch.randn(1, 3, 224, 224)
            clip_embedding_dim = self.clip_model.encode_image(dummy_image).shape[1]
        
        # EEG to CLIP encoder
        self.eeg_encoder = EEGToCLIPEncoder(
            eeg_input_dim, clip_embedding_dim, hidden_dims
        )
        
        self.clip_embedding_dim = clip_embedding_dim
        
    def encode_eeg_to_clip(self, eeg_signals):
        """Encode EEG signals to CLIP embedding space"""
        return self.eeg_encoder(eeg_signals)
    
    def encode_image_to_clip(self, images):
        """Encode images to CLIP embedding space (for target)"""
        with torch.no_grad():
            clip_embeddings = self.clip_model.encode_image(images)
            return F.normalize(clip_embeddings.float(), dim=1)
    
    def forward(self, eeg_signals):
        """Forward pass for training"""
        predicted_clip_embedding = self.encode_eeg_to_clip(eeg_signals)
        return predicted_clip_embedding

# Loss Functions for Stage 1
def cosine_similarity_loss(predicted_embeddings, target_embeddings):
    """Cosine similarity loss for CLIP embedding prediction"""
    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(predicted_embeddings, target_embeddings, dim=1)
    
    # Convert to loss (1 - similarity)
    loss = 1 - cosine_sim.mean()
    
    return loss

def l2_distance_loss(predicted_embeddings, target_embeddings):
    """L2 distance loss for CLIP embedding prediction"""
    return F.mse_loss(predicted_embeddings, target_embeddings)

def combined_embedding_loss(predicted_embeddings, target_embeddings, 
                           cosine_weight=0.7, l2_weight=0.3):
    """Combined cosine + L2 loss for better embedding learning"""
    cosine_loss = cosine_similarity_loss(predicted_embeddings, target_embeddings)
    l2_loss = l2_distance_loss(predicted_embeddings, target_embeddings)
    
    total_loss = cosine_weight * cosine_loss + l2_weight * l2_loss
    
    return total_loss, {
        'cosine': cosine_loss.item(),
        'l2': l2_loss.item(),
        'total': total_loss.item()
    }

def evaluate_embedding_similarity(predicted_embeddings, target_embeddings):
    """Evaluate the quality of predicted embeddings"""
    with torch.no_grad():
        # Cosine similarity
        cosine_sim = F.cosine_similarity(predicted_embeddings, target_embeddings, dim=1)
        mean_cosine_sim = cosine_sim.mean().item()
        
        # L2 distance
        l2_distance = torch.norm(predicted_embeddings - target_embeddings, dim=1)
        mean_l2_distance = l2_distance.mean().item()
        
        # Embedding magnitude (should be close to 1 after normalization)
        pred_magnitude = torch.norm(predicted_embeddings, dim=1).mean().item()
        target_magnitude = torch.norm(target_embeddings, dim=1).mean().item()
        
        return {
            'cosine_similarity': mean_cosine_sim,
            'l2_distance': mean_l2_distance,
            'pred_magnitude': pred_magnitude,
            'target_magnitude': target_magnitude
        }

class EmbeddingQualityMetrics:
    """Track embedding quality metrics during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.cosine_similarities = []
        self.l2_distances = []
        self.pred_magnitudes = []
        self.target_magnitudes = []
    
    def update(self, predicted_embeddings, target_embeddings):
        """Update metrics with new batch"""
        metrics = evaluate_embedding_similarity(predicted_embeddings, target_embeddings)
        
        self.cosine_similarities.append(metrics['cosine_similarity'])
        self.l2_distances.append(metrics['l2_distance'])
        self.pred_magnitudes.append(metrics['pred_magnitude'])
        self.target_magnitudes.append(metrics['target_magnitude'])
    
    def get_average_metrics(self):
        """Get average metrics"""
        return {
            'avg_cosine_similarity': np.mean(self.cosine_similarities),
            'avg_l2_distance': np.mean(self.l2_distances),
            'avg_pred_magnitude': np.mean(self.pred_magnitudes),
            'avg_target_magnitude': np.mean(self.target_magnitudes),
            'std_cosine_similarity': np.std(self.cosine_similarities),
            'std_l2_distance': np.std(self.l2_distances)
        }

def create_eeg_to_clip_model(eeg_input_dim, config):
    """Factory function to create EEG-to-CLIP model"""
    
    model = EEGToCLIPModel(
        eeg_input_dim=eeg_input_dim,
        clip_model_name=config.get('clip_model', 'ViT-B/32'),
        hidden_dims=config.get('encoder_hidden_dims', [512, 256, 128])
    )
    
    return model

def save_stage1_model(model, scaler, config, training_history, save_path):
    """Save Stage 1 model (EEG-to-CLIP)"""
    
    torch.save({
        'encoder_state_dict': model.eeg_encoder.state_dict(),
        'scaler': scaler,
        'config': config,
        'training_history': training_history,
        'model_type': 'eeg_to_clip_stage1',
        'clip_embedding_dim': model.clip_embedding_dim,
        'eeg_input_dim': model.eeg_encoder.eeg_input_dim
    }, save_path)
    
    print(f"Stage 1 Model (EEG-to-CLIP) saved to: {save_path}")

def load_stage1_model(model_path, device='cpu'):
    """Load Stage 1 model (EEG-to-CLIP)"""
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Initialize model
    config = checkpoint['config']
    model = EEGToCLIPModel(
        eeg_input_dim=checkpoint['eeg_input_dim'],
        clip_model_name=config.get('clip_model', 'ViT-B/32'),
        hidden_dims=config.get('encoder_hidden_dims', [512, 256, 128])
    )
    
    # Load encoder state
    model.eeg_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model, checkpoint['scaler'], checkpoint
