import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import json

def evaluate_model(model, test_loader, stimuli_dir, device, max_samples=1000):
    """Evaluate model performance"""
    
    model.eval()
    
    # Load all stimuli embeddings
    stimuli_embeddings = []
    stimuli_paths = []
    
    for i in range(10):
        img_path = os.path.join(stimuli_dir, f"{i}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            img_tensor = model.clip_preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                img_embedding = model.encode_image(img_tensor)
                stimuli_embeddings.append(img_embedding)
                stimuli_paths.append(img_path)
    
    stimuli_embeddings = torch.cat(stimuli_embeddings, dim=0)
    
    # Evaluate samples
    correct_predictions = {1: 0, 3: 0, 5: 0}
    total_samples = 0
    all_similarities = []
    
    with torch.no_grad():
        eval_pbar = tqdm(test_loader, desc="Evaluating")
        for batch_idx, (eeg_signals, images, stimulus_indices, image_paths) in enumerate(eval_pbar):
            if total_samples >= max_samples:
                break
                
            eeg_signals = eeg_signals.to(device)
            batch_size = eeg_signals.size(0)
            
            # Get EEG embeddings
            eeg_clip_emb, vae_emb, eeg_recon, mu, logvar = model(eeg_signals)
            
            # Compute similarities
            similarities = torch.matmul(eeg_clip_emb, stimuli_embeddings.T)
            
            # Check accuracy for each sample in batch
            for i in range(batch_size):
                if total_samples >= max_samples:
                    break
                    
                sample_similarities = similarities[i]
                top_indices = torch.argsort(sample_similarities, descending=True)
                true_stimulus = stimulus_indices[i].item()
                
                # Store similarity scores
                all_similarities.append(sample_similarities.cpu().numpy())
                
                # Check top-k accuracy
                for k in [1, 3, 5]:
                    if true_stimulus in top_indices[:k]:
                        correct_predictions[k] += 1
                
                total_samples += 1
            
            # Update progress
            current_acc = correct_predictions[1] / total_samples if total_samples > 0 else 0
            eval_pbar.set_postfix({'Top-1 Acc': f'{current_acc:.3f}'})
    
    # Calculate final accuracies
    accuracies = {k: correct_predictions[k] / total_samples for k in correct_predictions}
    
    # Calculate similarity statistics
    all_similarities = np.array(all_similarities)
    similarity_stats = {
        'mean': float(np.mean(all_similarities)),
        'std': float(np.std(all_similarities)),
        'min': float(np.min(all_similarities)),
        'max': float(np.max(all_similarities))
    }
    
    results = {
        'accuracies': accuracies,
        'total_samples': total_samples,
        'correct_predictions': correct_predictions,
        'similarity_stats': similarity_stats
    }
    
    print(f"\nEvaluation Results ({total_samples} samples):")
    for k, acc in accuracies.items():
        print(f"  Top-{k} Accuracy: {acc:.3f} ({correct_predictions[k]}/{total_samples})")
    
    return results

def demonstrate_reconstruction(model, test_signals, test_metadata, stimuli_dir, device, num_demos=5):
    """Demonstrate reconstruction with specific samples"""
    
    model.eval()
    
    # Load all stimuli images and embeddings
    stimuli_images = []
    stimuli_embeddings = []
    stimuli_paths = []
    
    for i in range(10):
        img_path = os.path.join(stimuli_dir, f"{i}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            img_tensor = model.clip_preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                img_embedding = model.encode_image(img_tensor)
                stimuli_embeddings.append(img_embedding)
                stimuli_images.append(img)
                stimuli_paths.append(img_path)
    
    stimuli_embeddings = torch.cat(stimuli_embeddings, dim=0)
    
    # Select random test samples
    indices = np.random.choice(len(test_signals), min(num_demos, len(test_signals)), replace=False)
    
    results = []
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get original EEG signal
            original_eeg = test_signals[idx]
            
            # Encode EEG through VAE-CLIP
            query_eeg = torch.FloatTensor(original_eeg).unsqueeze(0).to(device)
            eeg_clip_emb, vae_emb, eeg_recon, mu, logvar = model(query_eeg)
            
            # Compute similarities with all stimuli
            similarities = torch.matmul(eeg_clip_emb, stimuli_embeddings.T).squeeze()
            
            # Get top 5 matches
            top_indices = torch.argsort(similarities, descending=True)[:5]
            
            # Get TRUE ground truth based on subject
            subject = test_metadata[idx]['subject']
            true_stimulus_idx = subject % len(stimuli_images)
            
            result = {
                'query_idx': idx,
                'electrode': test_metadata[idx]['electrode'],
                'subject': subject,
                'trial': test_metadata[idx]['trial'],
                'original_eeg': original_eeg,
                'reconstructed_eeg': eeg_recon.cpu().numpy().squeeze(),
                'vae_embedding': vae_emb.cpu().numpy().squeeze(),
                'true_stimulus_idx': true_stimulus_idx,
                'true_stimulus_image': stimuli_images[true_stimulus_idx],
                'true_stimulus_path': stimuli_paths[true_stimulus_idx],
                'top_predictions': [
                    {
                        'rank': j + 1,
                        'stimuli_idx': top_indices[j].item(),
                        'similarity': similarities[top_indices[j]].item(),
                        'image': stimuli_images[top_indices[j]],
                        'path': stimuli_paths[top_indices[j]],
                        'is_correct': top_indices[j].item() == true_stimulus_idx
                    }
                    for j in range(5)
                ]
            }
            
            results.append(result)
            
            print(f"\nDemo {i+1}: Electrode {test_metadata[idx]['electrode']}, Subject {subject}")
            print(f"  TRUE Ground Truth: Stimuli {true_stimulus_idx}")
            print(f"  Predictions:")
            for pred in result['top_predictions'][:3]:
                correct_mark = " ✅" if pred['is_correct'] else ""
                print(f"    Rank {pred['rank']}: Stimuli {pred['stimuli_idx']} (similarity: {pred['similarity']:.4f}){correct_mark}")
    
    return results

def visualize_reconstruction_results(results, save_path):
    """Create the requested visualization format"""
    
    num_queries = len(results)
    fig, axes = plt.subplots(num_queries, 5, figsize=(20, 4*num_queries))
    
    if num_queries == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Column 1: Original EEG signal + VAE reconstruction
        axes[i, 0].plot(result['original_eeg'], 'b-', linewidth=1, label='Original')
        axes[i, 0].plot(result['reconstructed_eeg'], 'r--', linewidth=1, alpha=0.7, label='VAE Recon')
        axes[i, 0].set_title(f"EEG Signal\n{result['electrode']} - Subject {result['subject']}")
        axes[i, 0].set_xlabel("Sample")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Column 2: TRUE Ground Truth Image
        axes[i, 1].imshow(result['true_stimulus_image'])
        axes[i, 1].set_title(f"TRUE Ground Truth\nStimuli {result['true_stimulus_idx']}\n(Subject {result['subject']})")
        axes[i, 1].axis('off')
        
        # Add GREEN border for true ground truth
        for spine in axes[i, 1].spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(4)
        
        # Columns 3-5: Top 3 Predictions
        for j in range(3):
            pred = result['top_predictions'][j]
            axes[i, j+2].imshow(pred['image'])
            
            # Title with correctness indicator
            title = f"Prediction {pred['rank']}\nStimuli {pred['stimuli_idx']}\nSim: {pred['similarity']:.3f}"
            if pred['is_correct']:
                title += "\n✅ CORRECT"
            
            axes[i, j+2].set_title(title)
            axes[i, j+2].axis('off')
            
            # Border color based on correctness and rank
            if pred['is_correct']:
                border_color = 'green'
                border_width = 4
            elif j == 0:  # Top prediction
                border_color = 'gold'
                border_width = 3
            elif j == 1:  # Second prediction
                border_color = 'silver'
                border_width = 2
            else:  # Third prediction
                border_color = '#CD7F32'  # Bronze
                border_width = 2
            
            for spine in axes[i, j+2].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Reconstruction results saved to: {save_path}")

def create_training_curves(training_results, save_path):
    """Create training curves visualization"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    epochs = range(1, len(training_results['train_losses']) + 1)
    
    # Total loss
    axes[0].plot(epochs, training_results['train_losses'], 'b-', label='Train', marker='o')
    axes[0].plot(epochs, training_results['val_losses'], 'r-', label='Validation', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # VAE loss
    axes[1].plot(epochs, training_results['vae_losses'], 'g-', label='VAE Loss', marker='^')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('VAE Loss')
    axes[1].set_title('VAE Loss (Reconstruction + KL)')
    axes[1].legend()
    axes[1].grid(True)
    
    # CLIP loss
    axes[2].plot(epochs, training_results['clip_losses'], 'm-', label='CLIP Loss', marker='d')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('CLIP Loss')
    axes[2].set_title('CLIP Contrastive Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training curves saved to: {save_path}")

def save_evaluation_results(evaluation_results, demo_results, save_path):
    """Save evaluation results to JSON"""
    
    # Convert demo results to serializable format
    serializable_demos = []
    for demo in demo_results:
        serializable_demo = {
            'query_idx': demo['query_idx'],
            'electrode': demo['electrode'],
            'subject': demo['subject'],
            'trial': demo['trial'],
            'true_stimulus_idx': demo['true_stimulus_idx'],
            'top_predictions': [
                {
                    'rank': pred['rank'],
                    'stimuli_idx': pred['stimuli_idx'],
                    'similarity': float(pred['similarity']),
                    'is_correct': pred['is_correct']
                }
                for pred in demo['top_predictions']
            ]
        }
        serializable_demos.append(serializable_demo)
    
    results = {
        'evaluation_metrics': evaluation_results,
        'demonstration_samples': serializable_demos,
        'summary': {
            'total_evaluated': evaluation_results['total_samples'],
            'top_1_accuracy': evaluation_results['accuracies'][1],
            'top_3_accuracy': evaluation_results['accuracies'][3],
            'top_5_accuracy': evaluation_results['accuracies'][5],
            'demo_samples': len(demo_results)
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to: {save_path}")
