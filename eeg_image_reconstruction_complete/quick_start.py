#!/usr/bin/env python3
"""
Quick Start Script for EEG-to-Image Reconstruction
=================================================

This script provides a quick start option with smaller dataset for testing.

Usage: python quick_start.py
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def quick_start():
    """Run quick start with smaller dataset"""
    
    print("🚀 EEG-to-Image Reconstruction - Quick Start")
    print("=" * 50)
    print("This will run with smaller dataset for quick testing:")
    print("  - Max 5,000 training samples")
    print("  - Max 1,000 test samples") 
    print("  - 5 epochs")
    print("  - CPU training")
    print("=" * 50)
    
    # Modify the main script configuration for quick start
    import run_complete_pipeline
    
    # Override configuration for quick start
    original_main = run_complete_pipeline.main
    
    def quick_main():
        # Patch the config in the main function
        import run_complete_pipeline
        
        # Store original config creation
        original_code = """
        config = {
            # Dataset Configuration
            'max_train_samples': 50000,      # Limit for memory management
            'max_test_samples': 10000,       # Limit test samples
            'chunk_size': 50000,             # Processing chunk size
            
            # Training Configuration
            'batch_size': 32,                # Batch size
            'num_epochs': 15,                # Training epochs
            'learning_rate': 1e-4,           # Learning rate
            'vae_latent_dim': 128,           # VAE latent dimension
            'vae_beta': 0.1,                 # VAE KL divergence weight
            'clip_model': 'ViT-B/32',        # CLIP model variant
            
            # Hardware Configuration
            'use_gpu': True,                 # Set False for CPU only
            'num_workers': 0,                # DataLoader workers (0 for Windows)
            
            # Evaluation Configuration
            'eval_samples': 1000,            # Samples for evaluation
            'demo_samples': 5,               # Samples for demonstration
        }
        """
        
        # Quick start config
        quick_config = {
            # Dataset Configuration
            'max_train_samples': 5000,       # Smaller for quick start
            'max_test_samples': 1000,        # Smaller for quick start
            'chunk_size': 10000,             # Smaller chunks
            
            # Training Configuration
            'batch_size': 16,                # Smaller batch size
            'num_epochs': 5,                 # Fewer epochs
            'learning_rate': 1e-4,           # Same learning rate
            'vae_latent_dim': 64,            # Smaller latent dimension
            'vae_beta': 0.1,                 # Same VAE beta
            'clip_model': 'ViT-B/32',        # Same CLIP model
            
            # Hardware Configuration
            'use_gpu': False,                # Force CPU for compatibility
            'num_workers': 0,                # No parallel workers
            
            # Evaluation Configuration
            'eval_samples': 500,             # Fewer evaluation samples
            'demo_samples': 3,               # Fewer demo samples
        }
        
        # Replace the config in the main function
        run_complete_pipeline.config = quick_config
        
        # Call the original main with modified config
        return original_main()
    
    # Replace main function temporarily
    run_complete_pipeline.main = quick_main
    
    try:
        # Run the modified main
        quick_main()
    finally:
        # Restore original main
        run_complete_pipeline.main = original_main

if __name__ == "__main__":
    quick_start()
