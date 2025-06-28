#!/usr/bin/env python3
"""
Pendekatan 2: EEG-to-CLIP Direct Mapping
=======================================

Pipeline lengkap untuk EEG-to-Image reconstruction menggunakan two-stage approach:
Stage 1: EEG → CLIP embedding prediction
Stage 2: CLIP embedding → Image generation

Usage: python run_approach2_pipeline.py
"""

import os
import sys
import json
from datetime import datetime
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import process_full_dataset, prepare_training_data
from src.eeg_to_clip_model import EEGToCLIPModel, save_stage1_model
from src.clip_to_image_model import CLIPToImageModel, save_stage2_model, create_two_stage_model
from src.two_stage_training import create_two_stage_data_loaders, train_stage1_eeg_to_clip, train_stage2_clip_to_image

def main():
    """Main pipeline for Approach 2: Two-Stage Direct Mapping"""
    
    print("="*70)
    print("🧠🎨 PENDEKATAN 2: EEG-TO-CLIP DIRECT MAPPING")
    print("="*70)
    print("Two-Stage Approach dengan TRUE Subject-Stimulus Correspondence")
    print("Stage 1: EEG → CLIP Embedding Prediction")
    print("Stage 2: CLIP Embedding → Image Generation")
    print("="*70)
    
    # Configuration for Approach 2
    config = {
        # Dataset Configuration
        'max_train_samples': 25000,      # Larger for embedding learning
        'max_test_samples': 5000,        # Larger test set
        'chunk_size': 40000,             # Processing chunk size
        
        # Model Configuration
        'encoder_hidden_dims': [512, 256, 128],  # EEG encoder architecture
        'clip_model': 'ViT-B/32',        # CLIP model variant
        'decoder_type': 'simple',        # Image decoder type
        'image_size': 224,               # Generated image size
        
        # Stage 1 Training (EEG → CLIP)
        'stage1_epochs': 30,             # More epochs for embedding learning
        'stage1_lr': 1e-3,               # Higher learning rate for stage 1
        'cosine_weight': 0.7,            # Weight for cosine similarity loss
        'l2_weight': 0.3,                # Weight for L2 distance loss
        
        # Stage 2 Training (CLIP → Image)
        'stage2_epochs': 20,             # Epochs for image generation
        'stage2_lr': 2e-4,               # Learning rate for stage 2
        
        # Training Configuration
        'batch_size': 32,                # Batch size
        'weight_decay': 1e-5,            # Weight decay
        
        # Hardware Configuration
        'use_gpu': True,                 # Use GPU if available
        'num_workers': 0,                # DataLoader workers
        
        # Evaluation Configuration
        'eval_samples': 1000,            # Samples for evaluation
        'demo_samples': 5,               # Samples for demonstration
    }
    
    print(f"Configuration:")
    print(f"  Approach: Two-Stage Direct Mapping")
    print(f"  Stage 1: {config['stage1_epochs']} epochs, LR={config['stage1_lr']}")
    print(f"  Stage 2: {config['stage2_epochs']} epochs, LR={config['stage2_lr']}")
    print(f"  Training samples: {config['max_train_samples']:,}")
    print(f"  Loss weights: Cosine={config['cosine_weight']}, L2={config['l2_weight']}")
    
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(base_dir, 'dataset', 'datasets', 'EP1.01.txt')
    stimuli_dir = os.path.join(base_dir, 'dataset', 'datasets')
    processed_dir = os.path.join(base_dir, 'dataset', 'processed')
    results_dir = os.path.join(base_dir, 'results')
    
    # Create directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'stage1_models'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'stage2_models'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(results_dir, 'logs', 'approach2_training_log.txt')
    
    # Get device
    device = torch.device("cuda" if config['use_gpu'] and torch.cuda.is_available() else "cpu")
    config['device'] = device
    print(f"Using device: {device}")
    
    try:
        # Phase 1: Process Dataset
        print("\n" + "="*50)
        print("PHASE 1: DATASET PROCESSING")
        print("="*50)
        
        train_path, test_path, summary_path = process_full_dataset(
            dataset_file, 
            processed_dir, 
            test_size=0.2, 
            chunk_size=config['chunk_size'],
            max_samples=config['max_train_samples'] + config['max_test_samples']
        )
        
        # Phase 2: Prepare Training Data
        print("\n" + "="*50)
        print("PHASE 2: DATA PREPARATION")
        print("="*50)
        
        (train_signals, train_metadata, 
         test_signals, test_metadata, scaler) = prepare_training_data(
            train_path, test_path, 
            config['max_train_samples'], 
            config['max_test_samples']
        )
        
        config['eeg_input_dim'] = train_signals.shape[1]
        print(f"EEG input dimension: {config['eeg_input_dim']}")
        
        # Phase 3: Initialize Models
        print("\n" + "="*50)
        print("PHASE 3: MODEL INITIALIZATION")
        print("="*50)
        
        # Stage 1 Model: EEG → CLIP
        stage1_model = EEGToCLIPModel(
            eeg_input_dim=config['eeg_input_dim'],
            clip_model_name=config['clip_model'],
            hidden_dims=config['encoder_hidden_dims']
        )
        
        print(f"Stage 1 Model (EEG → CLIP):")
        print(f"  Input: EEG signals ({config['eeg_input_dim']} dim)")
        print(f"  Output: CLIP embeddings ({stage1_model.clip_embedding_dim} dim)")
        print(f"  Architecture: {config['encoder_hidden_dims']}")
        
        # Stage 2 Model: CLIP → Image
        stage2_model = CLIPToImageModel(
            clip_embedding_dim=stage1_model.clip_embedding_dim,
            decoder_type=config['decoder_type'],
            image_size=config['image_size']
        )
        
        print(f"Stage 2 Model (CLIP → Image):")
        print(f"  Input: CLIP embeddings ({stage1_model.clip_embedding_dim} dim)")
        print(f"  Output: RGB images ({config['image_size']}×{config['image_size']}×3)")
        print(f"  Decoder: {config['decoder_type']}")
        
        # Create data loaders
        train_loader, val_loader = create_two_stage_data_loaders(
            train_signals, train_metadata, 
            test_signals, test_metadata,
            stimuli_dir, config
        )
        
        # Phase 4: Stage 1 Training (EEG → CLIP)
        print("\n" + "="*50)
        print("PHASE 4: STAGE 1 TRAINING (EEG → CLIP)")
        print("="*50)
        
        stage1_history = train_stage1_eeg_to_clip(
            stage1_model, train_loader, val_loader, config, device, log_file
        )
        
        # Save Stage 1 Model
        stage1_model_path = os.path.join(results_dir, 'stage1_models', 'eeg_to_clip_model.pth')
        save_stage1_model(stage1_model, scaler, config, stage1_history, stage1_model_path)
        
        # Phase 5: Stage 2 Training (CLIP → Image)
        print("\n" + "="*50)
        print("PHASE 5: STAGE 2 TRAINING (CLIP → IMAGE)")
        print("="*50)
        
        stage2_history = train_stage2_clip_to_image(
            stage1_model, stage2_model, train_loader, val_loader, config, device, log_file
        )
        
        # Save Stage 2 Model
        stage2_model_path = os.path.join(results_dir, 'stage2_models', 'clip_to_image_model.pth')
        save_stage2_model(stage2_model, config, stage2_history, stage2_model_path)
        
        # Phase 6: Create Complete Two-Stage Model
        print("\n" + "="*50)
        print("PHASE 6: COMPLETE MODEL CREATION")
        print("="*50)
        
        complete_model = create_two_stage_model(stage1_model, stage2_model)
        
        # Save complete model
        complete_model_path = os.path.join(results_dir, 'two_stage_complete_model.pth')
        torch.save({
            'stage1_model_path': stage1_model_path,
            'stage2_model_path': stage2_model_path,
            'config': config,
            'stage1_history': stage1_history,
            'stage2_history': stage2_history,
            'model_type': 'two_stage_complete'
        }, complete_model_path)
        
        print(f"Complete two-stage model saved to: {complete_model_path}")
        
        # Phase 7: Generate Visualizations
        print("\n" + "="*50)
        print("PHASE 7: VISUALIZATION")
        print("="*50)
        
        import matplotlib.pyplot as plt
        
        # Plot Stage 1 training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs1 = range(1, len(stage1_history['train_losses']) + 1)
        
        # Stage 1 losses
        axes[0, 0].plot(epochs1, stage1_history['train_losses'], 'b-', label='Train')
        axes[0, 0].plot(epochs1, stage1_history['val_losses'], 'r-', label='Validation')
        axes[0, 0].set_title('Stage 1: EEG-to-CLIP Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Combined Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Cosine similarity
        axes[0, 1].plot(epochs1, stage1_history['cosine_similarities'], 'g-', label='Cosine Similarity')
        axes[0, 1].set_title('Stage 1: CLIP Embedding Similarity')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Cosine Similarity')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # L2 distances
        axes[1, 0].plot(epochs1, stage1_history['l2_distances'], 'm-', label='L2 Distance')
        axes[1, 0].set_title('Stage 1: CLIP Embedding L2 Distance')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('L2 Distance')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Stage 2 losses
        epochs2 = range(1, len(stage2_history['train_losses']) + 1)
        axes[1, 1].plot(epochs2, stage2_history['train_losses'], 'orange', label='Train')
        axes[1, 1].plot(epochs2, stage2_history['val_losses'], 'purple', label='Validation')
        axes[1, 1].set_title('Stage 2: CLIP-to-Image Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Reconstruction Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        curves_path = os.path.join(results_dir, 'visualizations', 'approach2_training_curves.png')
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save configuration and results
        config_path = os.path.join(results_dir, 'logs', 'approach2_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        results_summary = {
            'approach': 'Two-Stage Direct Mapping',
            'stage1_final_loss': stage1_history['train_losses'][-1],
            'stage1_final_cosine_similarity': stage1_history['cosine_similarities'][-1],
            'stage2_final_loss': stage2_history['train_losses'][-1],
            'stage1_epochs': len(stage1_history['train_losses']),
            'stage2_epochs': len(stage2_history['train_losses']),
            'stage1_model_path': stage1_model_path,
            'stage2_model_path': stage2_model_path,
            'complete_model_path': complete_model_path,
            'config': config
        }
        
        summary_path = os.path.join(results_dir, 'logs', 'approach2_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print("\n" + "="*70)
        print("🎉 PENDEKATAN 2 COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Approach: Two-Stage Direct Mapping")
        print(f"Final Results:")
        print(f"  Stage 1 Loss: {stage1_history['train_losses'][-1]:.4f}")
        print(f"  Stage 1 Cosine Similarity: {stage1_history['cosine_similarities'][-1]:.4f}")
        print(f"  Stage 2 Reconstruction Loss: {stage2_history['train_losses'][-1]:.4f}")
        print(f"\nGenerated Files:")
        print(f"  🤖 Stage 1 Model: {stage1_model_path}")
        print(f"  🎨 Stage 2 Model: {stage2_model_path}")
        print(f"  🔗 Complete Model: {complete_model_path}")
        print(f"  📈 Training curves: {curves_path}")
        print(f"  📋 Summary: {summary_path}")
        print("="*70)
        print("✅ Two-stage model trained successfully!")
        print("✅ EEG signals can now be mapped to CLIP embeddings")
        print("✅ CLIP embeddings can be converted to images")
        print("✅ Ready for end-to-end EEG-to-Image generation")
        
    except Exception as e:
        print(f"\n❌ Error in Approach 2 pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Save error log
        error_log = {
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'approach': 'two_stage_direct_mapping'
        }
        
        error_path = os.path.join(results_dir, 'logs', 'approach2_error.json')
        with open(error_path, 'w') as f:
            json.dump(error_log, f, indent=2, default=str)
        
        raise

if __name__ == "__main__":
    main()
