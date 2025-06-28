#!/usr/bin/env python3
"""
Pendekatan 1: CLIP sebagai Semantic Loss Function
================================================

Pipeline lengkap untuk EEG-to-Image reconstruction menggunakan CLIP sebagai semantic loss
dalam arsitektur VAE-GAN.

Usage: python run_approach1_pipeline.py
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import process_full_dataset, prepare_training_data
from src.semantic_loss_model import SemanticLossEEGImageModel
from src.training import create_semantic_data_loaders, train_semantic_loss_model, save_semantic_model

def main():
    """Main pipeline for Approach 1: Semantic Loss"""
    
    print("="*70)
    print("🧠🎨 PENDEKATAN 1: CLIP SEBAGAI SEMANTIC LOSS FUNCTION")
    print("="*70)
    print("VAE-GAN + CLIP Semantic Loss dengan TRUE Subject-Stimulus Correspondence")
    print("L_total = λ_adv×L_GAN + λ_rec×L_REC + λ_kl×L_KL + λ_clip×L_CLIP")
    print("="*70)
    
    # Configuration for Approach 1
    config = {
        # Dataset Configuration
        'max_train_samples': 20000,      # Smaller for GAN training
        'max_test_samples': 4000,        # Smaller test set
        'chunk_size': 30000,             # Processing chunk size
        
        # Model Configuration
        'latent_dim': 128,               # VAE latent dimension
        'image_size': 224,               # Generated image size
        
        # Training Configuration
        'batch_size': 16,                # Smaller batch for memory
        'num_epochs': 25,                # More epochs for GAN convergence
        'learning_rate': 2e-4,           # Standard GAN learning rate
        
        # Loss Weights (KEY for Approach 1)
        'lambda_adv': 1.0,               # GAN adversarial loss
        'lambda_rec': 10.0,              # Pixel reconstruction loss
        'lambda_kl': 1.0,                # VAE KL divergence loss
        'lambda_clip': 5.0,              # CLIP semantic loss (MAIN INNOVATION!)
        'vae_beta': 0.1,                 # VAE beta parameter
        
        # Hardware Configuration
        'use_gpu': True,                 # Use GPU if available
        'num_workers': 0,                # DataLoader workers
        
        # Evaluation Configuration
        'eval_samples': 500,             # Samples for evaluation
        'demo_samples': 5,               # Samples for demonstration
    }
    
    print(f"Configuration:")
    print(f"  Approach: Semantic Loss Function")
    print(f"  Loss weights: Adv={config['lambda_adv']}, Rec={config['lambda_rec']}, KL={config['lambda_kl']}, CLIP={config['lambda_clip']}")
    print(f"  Training samples: {config['max_train_samples']:,}")
    print(f"  Epochs: {config['num_epochs']}")
    
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(base_dir, 'dataset', 'datasets', 'EP1.01.txt')
    stimuli_dir = os.path.join(base_dir, 'dataset', 'datasets')
    processed_dir = os.path.join(base_dir, 'dataset', 'processed')
    results_dir = os.path.join(base_dir, 'results')
    
    # Create directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(results_dir, 'logs', 'approach1_training_log.txt')
    
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
        
        # Phase 3: Initialize Model
        print("\n" + "="*50)
        print("PHASE 3: MODEL INITIALIZATION")
        print("="*50)
        
        model = SemanticLossEEGImageModel(
            eeg_input_dim=config['eeg_input_dim'],
            latent_dim=config['latent_dim']
        )
        
        print(f"Model components:")
        print(f"  EEG Encoder: VAE with latent dim {config['latent_dim']}")
        print(f"  Image Generator: ConvTranspose2d layers")
        print(f"  Discriminator: Convolutional layers")
        print(f"  CLIP Model: Frozen ViT-B/32 for semantic loss")
        
        # Create data loaders
        train_loader, val_loader = create_semantic_data_loaders(
            train_signals, train_metadata, 
            test_signals, test_metadata,
            stimuli_dir, config
        )
        
        # Phase 4: Training
        print("\n" + "="*50)
        print("PHASE 4: SEMANTIC LOSS TRAINING")
        print("="*50)
        
        training_history = train_semantic_loss_model(
            model, train_loader, val_loader, config, device, log_file
        )
        
        # Phase 5: Save Model
        print("\n" + "="*50)
        print("PHASE 5: MODEL SAVING")
        print("="*50)
        
        model_path = os.path.join(results_dir, 'models', 'semantic_loss_model.pth')
        save_semantic_model(model, scaler, config, training_history, model_path)
        
        # Phase 6: Generate Training Curves
        print("\n" + "="*50)
        print("PHASE 6: VISUALIZATION")
        print("="*50)
        
        import matplotlib.pyplot as plt
        
        # Plot training curves
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        epochs = range(1, len(training_history['g_losses']) + 1)
        
        # Generator and Discriminator losses
        axes[0, 0].plot(epochs, training_history['g_losses'], 'b-', label='Generator')
        axes[0, 0].plot(epochs, training_history['d_losses'], 'r-', label='Discriminator')
        axes[0, 0].set_title('Generator vs Discriminator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Semantic loss (KEY METRIC!)
        axes[0, 1].plot(epochs, training_history['semantic_losses'], 'g-', label='CLIP Semantic Loss')
        axes[0, 1].set_title('CLIP Semantic Loss (Key Innovation)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Semantic Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Reconstruction loss
        axes[0, 2].plot(epochs, training_history['reconstruction_losses'], 'm-', label='L1 Reconstruction')
        axes[0, 2].set_title('Pixel Reconstruction Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('L1 Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # KL loss
        axes[1, 0].plot(epochs, training_history['kl_losses'], 'c-', label='KL Divergence')
        axes[1, 0].set_title('VAE KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Adversarial loss
        axes[1, 1].plot(epochs, training_history['adversarial_losses'], 'orange', label='Adversarial')
        axes[1, 1].set_title('Generator Adversarial Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Adversarial Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Combined view
        axes[1, 2].plot(epochs, training_history['semantic_losses'], 'g-', label=f'Semantic (λ={config["lambda_clip"]})')
        axes[1, 2].plot(epochs, training_history['reconstruction_losses'], 'm-', label=f'Reconstruction (λ={config["lambda_rec"]})')
        axes[1, 2].plot(epochs, training_history['adversarial_losses'], 'orange', label=f'Adversarial (λ={config["lambda_adv"]})')
        axes[1, 2].set_title('Combined Loss Components')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        curves_path = os.path.join(results_dir, 'visualizations', 'approach1_training_curves.png')
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save configuration and results
        config_path = os.path.join(results_dir, 'logs', 'approach1_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        results_summary = {
            'approach': 'Semantic Loss Function',
            'final_generator_loss': training_history['g_losses'][-1],
            'final_discriminator_loss': training_history['d_losses'][-1],
            'final_semantic_loss': training_history['semantic_losses'][-1],
            'final_reconstruction_loss': training_history['reconstruction_losses'][-1],
            'training_epochs': len(training_history['g_losses']),
            'model_path': model_path,
            'config': config
        }
        
        summary_path = os.path.join(results_dir, 'logs', 'approach1_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print("\n" + "="*70)
        print("🎉 PENDEKATAN 1 COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Approach: CLIP Semantic Loss Function")
        print(f"Final Results:")
        print(f"  Generator Loss: {training_history['g_losses'][-1]:.4f}")
        print(f"  Discriminator Loss: {training_history['d_losses'][-1]:.4f}")
        print(f"  Semantic Loss: {training_history['semantic_losses'][-1]:.4f}")
        print(f"  Reconstruction Loss: {training_history['reconstruction_losses'][-1]:.4f}")
        print(f"\nGenerated Files:")
        print(f"  🤖 Model: {model_path}")
        print(f"  📈 Training curves: {curves_path}")
        print(f"  📋 Summary: {summary_path}")
        print("="*70)
        print("✅ Model trained with CLIP semantic guidance!")
        print("✅ Generator learns to create semantically correct images")
        print("✅ Ready for image generation and evaluation")
        
    except Exception as e:
        print(f"\n❌ Error in Approach 1 pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Save error log
        error_log = {
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'approach': 'semantic_loss'
        }
        
        error_path = os.path.join(results_dir, 'logs', 'approach1_error.json')
        with open(error_path, 'w') as f:
            json.dump(error_log, f, indent=2, default=str)
        
        raise

if __name__ == "__main__":
    import torch
    main()
