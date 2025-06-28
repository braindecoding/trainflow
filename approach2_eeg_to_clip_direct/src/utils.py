import os
import json
import time
from datetime import datetime
import torch

def setup_directories(base_dir):
    """Setup required directories"""
    
    directories = [
        os.path.join(base_dir, 'results'),
        os.path.join(base_dir, 'results', 'models'),
        os.path.join(base_dir, 'results', 'visualizations'),
        os.path.join(base_dir, 'results', 'logs'),
        os.path.join(base_dir, 'dataset', 'processed')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directory structure created:")
    for directory in directories:
        print(f"  ✓ {directory}")

def get_device(use_gpu=True):
    """Get appropriate device for training"""
    
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device

def save_config(config, save_path):
    """Save configuration to JSON file"""
    
    # Convert any non-serializable objects
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, torch.device):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Configuration saved to: {save_path}")

def create_timestamp():
    """Create timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def log_system_info(log_file):
    """Log system information"""
    
    info = [
        f"Timestamp: {datetime.now()}",
        f"PyTorch version: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]
    
    if torch.cuda.is_available():
        info.extend([
            f"CUDA version: {torch.version.cuda}",
            f"GPU count: {torch.cuda.device_count()}",
            f"Current GPU: {torch.cuda.current_device()}",
            f"GPU name: {torch.cuda.get_device_name()}",
        ])
    
    with open(log_file, 'w') as f:
        f.write("EEG-to-Image Reconstruction Training Log\n")
        f.write("=" * 50 + "\n")
        for line in info:
            f.write(line + "\n")
        f.write("=" * 50 + "\n\n")
    
    print("System info logged:")
    for line in info:
        print(f"  {line}")

def estimate_memory_usage(batch_size, signal_length, vae_latent_dim=128):
    """Estimate memory usage for training"""
    
    # Rough estimates in MB
    eeg_batch = batch_size * signal_length * 4 / 1e6  # float32
    image_batch = batch_size * 3 * 224 * 224 * 4 / 1e6  # RGB images
    vae_latent = batch_size * vae_latent_dim * 4 / 1e6
    clip_embeddings = batch_size * 512 * 4 / 1e6
    gradients = (eeg_batch + image_batch + vae_latent + clip_embeddings) * 2  # Approximate
    
    total_mb = eeg_batch + image_batch + vae_latent + clip_embeddings + gradients
    
    print(f"Estimated memory usage per batch:")
    print(f"  EEG data: {eeg_batch:.1f} MB")
    print(f"  Image data: {image_batch:.1f} MB")
    print(f"  VAE latent: {vae_latent:.1f} MB")
    print(f"  CLIP embeddings: {clip_embeddings:.1f} MB")
    print(f"  Gradients (approx): {gradients:.1f} MB")
    print(f"  Total (approx): {total_mb:.1f} MB")
    
    return total_mb

def check_dataset_files(dataset_dir):
    """Check if required dataset files exist"""
    
    required_files = [
        os.path.join(dataset_dir, 'datasets', 'EP1.01.txt'),
    ]
    
    required_images = [
        os.path.join(dataset_dir, 'datasets', f'{i}.jpg') for i in range(10)
    ]
    
    missing_files = []
    
    # Check main dataset file
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    # Check stimulus images
    missing_images = []
    for img_path in required_images:
        if not os.path.exists(img_path):
            missing_images.append(img_path)
    
    if missing_files or missing_images:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        for img_path in missing_images:
            print(f"  - {img_path}")
        return False
    else:
        print("✅ All required dataset files found:")
        for file_path in required_files:
            size_mb = os.path.getsize(file_path) / 1e6
            print(f"  ✓ {file_path} ({size_mb:.1f} MB)")
        print(f"  ✓ {len(required_images)} stimulus images (0.jpg - 9.jpg)")
        return True

def print_progress_summary(phase, status, details=None):
    """Print formatted progress summary"""
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if status == "start":
        print(f"\n[{timestamp}] 🚀 Starting {phase}...")
    elif status == "complete":
        print(f"[{timestamp}] ✅ {phase} completed")
        if details:
            for detail in details:
                print(f"  {detail}")
    elif status == "error":
        print(f"[{timestamp}] ❌ {phase} failed")
        if details:
            for detail in details:
                print(f"  {detail}")
    
    print("-" * 50)

class Timer:
    """Simple timer for tracking execution time"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self):
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def elapsed_str(self):
        elapsed = self.elapsed()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

def validate_config(config):
    """Validate configuration parameters"""
    
    required_keys = [
        'max_train_samples', 'max_test_samples', 'batch_size', 
        'num_epochs', 'learning_rate', 'vae_latent_dim'
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    # Validate ranges
    if config['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")
    
    if config['num_epochs'] <= 0:
        raise ValueError("num_epochs must be positive")
    
    if config['learning_rate'] <= 0:
        raise ValueError("learning_rate must be positive")
    
    if config['vae_latent_dim'] <= 0:
        raise ValueError("vae_latent_dim must be positive")
    
    print("✅ Configuration validated successfully")

def create_final_summary(config, training_results, evaluation_results, total_time):
    """Create final summary of the entire pipeline"""
    
    summary = {
        'pipeline_info': {
            'total_execution_time': total_time,
            'timestamp': datetime.now().isoformat(),
            'configuration': config
        },
        'training_summary': {
            'epochs_completed': len(training_results['train_losses']),
            'final_train_loss': training_results['train_losses'][-1],
            'final_val_loss': training_results['val_losses'][-1],
            'training_time': training_results['training_time'],
            'final_temperature': training_results['final_temperature']
        },
        'evaluation_summary': {
            'samples_evaluated': evaluation_results['total_samples'],
            'top_1_accuracy': evaluation_results['accuracies'][1],
            'top_3_accuracy': evaluation_results['accuracies'][3],
            'top_5_accuracy': evaluation_results['accuracies'][5],
            'similarity_stats': evaluation_results['similarity_stats']
        }
    }
    
    return summary
