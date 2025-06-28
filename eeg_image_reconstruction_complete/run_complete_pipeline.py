#!/usr/bin/env python3
"""
Complete EEG-to-Image Reconstruction Pipeline
============================================

This script runs the complete pipeline for EEG-to-Image reconstruction:
1. Process full MindbigData EP1.01 dataset
2. Train VAE-CLIP model with TRUE subject-stimulus correspondence
3. Evaluate performance and generate visualizations

Usage: python run_complete_pipeline.py

Requirements: See requirements.txt
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import process_full_dataset, prepare_training_data
from src.model_architecture import EEGImageCLIP
from src.training import create_data_loaders, train_model, save_model
from src.evaluation import evaluate_model, demonstrate_reconstruction, visualize_reconstruction_results, create_training_curves, save_evaluation_results
from src.utils import setup_directories, get_device, save_config, log_system_info, check_dataset_files, print_progress_summary, Timer, validate_config, create_final_summary, estimate_memory_usage

def main():
    """Main pipeline execution"""
    
    print("="*70)
    print("🧠🎨 EEG-TO-IMAGE RECONSTRUCTION COMPLETE PIPELINE")
    print("="*70)
    print("VAE-CLIP with TRUE Subject-Stimulus Correspondence")
    print("Subject 0 → 0.jpg, Subject 1 → 1.jpg, etc.")
    print("="*70)
    
    # Configuration
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
    
    # Validate configuration
    validate_config(config)
    
    # Setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    setup_directories(base_dir)
    
    # Paths
    dataset_file = os.path.join(base_dir, 'dataset', 'datasets', 'EP1.01.txt')
    stimuli_dir = os.path.join(base_dir, 'dataset', 'datasets')
    processed_dir = os.path.join(base_dir, 'dataset', 'processed')
    results_dir = os.path.join(base_dir, 'results')
    
    # Check dataset files
    if not check_dataset_files(os.path.join(base_dir, 'dataset')):
        print("❌ Required dataset files missing. Please ensure:")
        print("  - dataset/datasets/EP1.01.txt exists")
        print("  - dataset/datasets/0.jpg through 9.jpg exist")
        return
    
    # Setup logging
    log_file = os.path.join(results_dir, 'logs', 'training_log.txt')
    log_system_info(log_file)
    
    # Get device
    device = get_device(config['use_gpu'])
    config['device'] = device
    
    # Estimate memory usage
    estimate_memory_usage(config['batch_size'], 264, config['vae_latent_dim'])
    
    # Save configuration
    config_path = os.path.join(results_dir, 'logs', 'training_config.json')
    save_config(config, config_path)
    
    # Start pipeline timer
    pipeline_timer = Timer()
    pipeline_timer.start()
    
    try:
        # Phase 1: Process Dataset
        print_progress_summary("Phase 1: Dataset Processing", "start")
        phase_timer = Timer()
        phase_timer.start()
        
        train_path, test_path, summary_path = process_full_dataset(
            dataset_file, 
            processed_dir, 
            test_size=0.2, 
            chunk_size=config['chunk_size'],
            max_samples=config['max_train_samples'] + config['max_test_samples']
        )
        
        phase_time = phase_timer.stop()
        print_progress_summary("Phase 1: Dataset Processing", "complete", [
            f"Processing time: {phase_timer.elapsed_str()}",
            f"Train data: {train_path}",
            f"Test data: {test_path}",
            f"Summary: {summary_path}"
        ])
        
        # Phase 2: Prepare Training Data
        print_progress_summary("Phase 2: Data Preparation", "start")
        phase_timer.start()
        
        (train_signals, train_metadata, 
         test_signals, test_metadata, scaler) = prepare_training_data(
            train_path, test_path, 
            config['max_train_samples'], 
            config['max_test_samples']
        )
        
        config['eeg_input_dim'] = train_signals.shape[1]
        
        phase_time = phase_timer.stop()
        print_progress_summary("Phase 2: Data Preparation", "complete", [
            f"Preparation time: {phase_timer.elapsed_str()}",
            f"Train samples: {len(train_signals):,}",
            f"Test samples: {len(test_signals):,}",
            f"EEG signal length: {config['eeg_input_dim']}"
        ])
        
        # Phase 3: Initialize Model
        print_progress_summary("Phase 3: Model Initialization", "start")
        
        model = EEGImageCLIP(
            eeg_input_dim=config['eeg_input_dim'],
            vae_latent_dim=config['vae_latent_dim'],
            clip_model_name=config['clip_model']
        )
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_signals, train_metadata, 
            test_signals, test_metadata,
            stimuli_dir, model.clip_preprocess, config
        )
        
        print_progress_summary("Phase 3: Model Initialization", "complete", [
            f"Model architecture: EEG-VAE-CLIP",
            f"EEG input dim: {config['eeg_input_dim']}",
            f"VAE latent dim: {config['vae_latent_dim']}",
            f"CLIP model: {config['clip_model']}"
        ])
        
        # Phase 4: Training
        print_progress_summary("Phase 4: Model Training", "start")
        phase_timer.start()
        
        training_results = train_model(
            model, train_loader, val_loader, config, device, log_file
        )
        
        phase_time = phase_timer.stop()
        print_progress_summary("Phase 4: Model Training", "complete", [
            f"Training time: {phase_timer.elapsed_str()}",
            f"Final train loss: {training_results['train_losses'][-1]:.4f}",
            f"Final val loss: {training_results['val_losses'][-1]:.4f}",
            f"Final temperature: {training_results['final_temperature']:.4f}"
        ])
        
        # Phase 5: Save Model
        print_progress_summary("Phase 5: Model Saving", "start")
        
        model_path = os.path.join(results_dir, 'models', 'vae_clip_model_final.pth')
        save_model(model, scaler, config, training_results, model_path)
        
        print_progress_summary("Phase 5: Model Saving", "complete", [
            f"Model saved to: {model_path}"
        ])
        
        # Phase 6: Evaluation
        print_progress_summary("Phase 6: Model Evaluation", "start")
        phase_timer.start()
        
        evaluation_results = evaluate_model(
            model, val_loader, stimuli_dir, device, config['eval_samples']
        )
        
        demo_results = demonstrate_reconstruction(
            model, test_signals, test_metadata, stimuli_dir, device, config['demo_samples']
        )
        
        phase_time = phase_timer.stop()
        print_progress_summary("Phase 6: Model Evaluation", "complete", [
            f"Evaluation time: {phase_timer.elapsed_str()}",
            f"Samples evaluated: {evaluation_results['total_samples']:,}",
            f"Top-1 accuracy: {evaluation_results['accuracies'][1]:.1%}",
            f"Top-3 accuracy: {evaluation_results['accuracies'][3]:.1%}",
            f"Top-5 accuracy: {evaluation_results['accuracies'][5]:.1%}"
        ])
        
        # Phase 7: Generate Visualizations
        print_progress_summary("Phase 7: Visualization Generation", "start")
        
        # Main reconstruction results (requested format)
        reconstruction_path = os.path.join(results_dir, 'visualizations', 'reconstruction_results.png')
        visualize_reconstruction_results(demo_results, reconstruction_path)
        
        # Training curves
        training_curves_path = os.path.join(results_dir, 'visualizations', 'training_curves.png')
        create_training_curves(training_results, training_curves_path)
        
        # Save evaluation results
        eval_results_path = os.path.join(results_dir, 'logs', 'evaluation_results.json')
        save_evaluation_results(evaluation_results, demo_results, eval_results_path)
        
        print_progress_summary("Phase 7: Visualization Generation", "complete", [
            f"Reconstruction results: {reconstruction_path}",
            f"Training curves: {training_curves_path}",
            f"Evaluation results: {eval_results_path}"
        ])
        
        # Final Summary
        total_time = pipeline_timer.stop()
        final_summary = create_final_summary(config, training_results, evaluation_results, total_time)
        
        summary_path = os.path.join(results_dir, 'logs', 'final_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)
        
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Total execution time: {pipeline_timer.elapsed_str()}")
        print(f"Final performance:")
        print(f"  Top-1 Accuracy: {evaluation_results['accuracies'][1]:.1%}")
        print(f"  Top-3 Accuracy: {evaluation_results['accuracies'][3]:.1%}")
        print(f"  Top-5 Accuracy: {evaluation_results['accuracies'][5]:.1%}")
        print(f"\nGenerated files:")
        print(f"  📊 Main results: {reconstruction_path}")
        print(f"  📈 Training curves: {training_curves_path}")
        print(f"  🤖 Trained model: {model_path}")
        print(f"  📋 Full summary: {summary_path}")
        print("="*70)
        
    except Exception as e:
        total_time = pipeline_timer.stop()
        print_progress_summary("Pipeline Execution", "error", [
            f"Error: {str(e)}",
            f"Execution time before error: {pipeline_timer.elapsed_str()}"
        ])
        
        # Save error log
        error_log = {
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'execution_time': total_time,
            'config': config
        }
        
        error_path = os.path.join(results_dir, 'logs', 'error_log.json')
        with open(error_path, 'w') as f:
            json.dump(error_log, f, indent=2, default=str)
        
        print(f"Error details saved to: {error_path}")
        raise

if __name__ == "__main__":
    main()
