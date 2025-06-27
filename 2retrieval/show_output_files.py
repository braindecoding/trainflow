#!/usr/bin/env python3
"""
Show Output Files Location
==========================

Display the location and details of all generated output files.
"""

import os
import numpy as np
import json

def show_output_files():
    """Show all output files and their details"""
    
    print('📁 OUTPUT FILES LOCATION AND DETAILS')
    print('=' * 60)
    
    base_path = r'd:\trainflow\2retrieval\outputs\mindbigdata_pairs'
    print(f'📂 MAIN OUTPUT DIRECTORY:')
    print(f'   {base_path}')
    
    print(f'\n📊 TRAIN DATASET FILES:')
    train_files = [
        'train_eeg_data.npy',
        'train_labels.npy', 
        'train_text_features.npy',
        'train_img_features.npy',
        'train_image_paths.json',
        'train_metadata.json'
    ]
    
    total_train_size = 0
    for filename in train_files:
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024**2)
            total_train_size += size_mb
            
            if filename.endswith('.npy'):
                data = np.load(filepath)
                print(f'   ✅ {filename}')
                print(f'      Size: {size_mb:.1f} MB')
                print(f'      Shape: {data.shape}')
                print(f'      Path: {filepath}')
            else:
                print(f'   ✅ {filename}')
                print(f'      Size: {size_mb:.1f} MB')
                print(f'      Path: {filepath}')
        else:
            print(f'   ❌ {filename} - NOT FOUND')
    
    print(f'\n📊 TEST DATASET FILES:')
    test_files = [
        'test_eeg_data.npy',
        'test_labels.npy',
        'test_text_features.npy', 
        'test_img_features.npy',
        'test_image_paths.json',
        'test_metadata.json'
    ]
    
    total_test_size = 0
    for filename in test_files:
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024**2)
            total_test_size += size_mb
            
            if filename.endswith('.npy'):
                data = np.load(filepath)
                print(f'   ✅ {filename}')
                print(f'      Size: {size_mb:.1f} MB')
                print(f'      Shape: {data.shape}')
                print(f'      Path: {filepath}')
            else:
                print(f'   ✅ {filename}')
                print(f'      Size: {size_mb:.1f} MB')
                print(f'      Path: {filepath}')
        else:
            print(f'   ❌ {filename} - NOT FOUND')
    
    print(f'\n💾 STORAGE SUMMARY:')
    print(f'   Train files total: {total_train_size:.1f} MB')
    print(f'   Test files total: {total_test_size:.1f} MB')
    print(f'   Grand total: {total_train_size + total_test_size:.1f} MB')
    
    # Load metadata for additional info
    try:
        with open(os.path.join(base_path, 'train_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        print(f'\n📋 DATASET METADATA:')
        n_trials = metadata.get('n_trials', 0)
        n_classes = metadata.get('n_classes', 0)
        eeg_shape = metadata.get('eeg_shape', [])
        model_type = metadata.get('model_type', 'Unknown')
        time_window = metadata.get('time_window', [])
        
        print(f'   Total train trials: {n_trials:,}')
        print(f'   Number of classes: {n_classes}')
        print(f'   EEG shape: {eeg_shape}')
        print(f'   CLIP model: {model_type}')
        print(f'   Time window: {time_window}')
        
    except Exception as e:
        print(f'\n⚠️ Could not load metadata: {e}')
    
    # Show cached CLIP features
    print(f'\n🔮 CACHED CLIP FEATURES:')
    clip_files = [
        'ViT-B_32_features_mindbigdata_train.pt',
        'ViT-B_32_features_mindbigdata_test.pt'
    ]
    
    for filename in clip_files:
        filepath = os.path.join(r'd:\trainflow\2retrieval\mindbigdata', filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024**2)
            print(f'   ✅ {filename}')
            print(f'      Size: {size_mb:.1f} MB')
            print(f'      Path: {filepath}')
        else:
            print(f'   ❌ {filename} - NOT FOUND')
    
    print(f'\n🎯 USAGE INSTRUCTIONS:')
    print(f'   1. For 3contrastivelearning, use files in:')
    print(f'      {base_path}')
    print(f'   2. Main files needed:')
    print(f'      - train_eeg_data.npy (EEG signals)')
    print(f'      - train_text_features.npy (CLIP text embeddings)')
    print(f'      - train_img_features.npy (CLIP image embeddings)')
    print(f'      - test_* files for validation')
    print(f'   3. All files are ready for CLIP training!')
    
    return base_path


if __name__ == "__main__":
    output_path = show_output_files()
    print(f'\n✅ OUTPUT FILES READY AT: {output_path}')
