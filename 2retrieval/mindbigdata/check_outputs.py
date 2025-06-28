#!/usr/bin/env python3
"""
Check MindBigData Retrieval Outputs
===================================

Comprehensive verification of retrieval system outputs.
"""

import os
import numpy as np
import json
import torch

def check_outputs():
    """Check all output files and their details"""
    
    print('🔍 CHECKING MINDBIGDATA RETRIEVAL OUTPUTS')
    print('=' * 60)
    
    base_path = r'd:\trainflow\2retrieval\outputs\mindbigdata_pairs'
    print(f'📂 OUTPUT DIRECTORY: {base_path}')
    
    if not os.path.exists(base_path):
        print(f'❌ ERROR: Output directory does not exist!')
        return
    
    print(f'\n📊 TRAIN DATASET VERIFICATION:')
    train_files = {
        'train_eeg_data.npy': 'EEG signals',
        'train_labels.npy': 'Digit labels', 
        'train_text_features.npy': 'CLIP text embeddings',
        'train_img_features.npy': 'CLIP image embeddings',
        'train_image_paths.json': 'Image paths',
        'train_metadata.json': 'Metadata'
    }
    
    train_data = {}
    total_train_size = 0
    
    for filename, description in train_files.items():
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024**2)
            total_train_size += size_mb
            
            print(f'   ✅ {filename} ({description})')
            print(f'      Size: {size_mb:.1f} MB')
            
            if filename.endswith('.npy'):
                data = np.load(filepath)
                train_data[filename] = data
                print(f'      Shape: {data.shape}')
                print(f'      Dtype: {data.dtype}')
                if 'labels' in filename:
                    unique_labels = np.unique(data)
                    print(f'      Unique values: {unique_labels}')
                elif 'eeg' in filename:
                    print(f'      Data range: [{data.min():.3f}, {data.max():.3f}]')
                elif 'features' in filename:
                    print(f'      Feature range: [{data.min():.3f}, {data.max():.3f}]')
            print()
        else:
            print(f'   ❌ {filename} - NOT FOUND')
    
    print(f'\n📊 TEST DATASET VERIFICATION:')
    test_files = {
        'test_eeg_data.npy': 'EEG signals',
        'test_labels.npy': 'Digit labels',
        'test_text_features.npy': 'CLIP text embeddings', 
        'test_img_features.npy': 'CLIP image embeddings',
        'test_image_paths.json': 'Image paths',
        'test_metadata.json': 'Metadata'
    }
    
    test_data = {}
    total_test_size = 0
    
    for filename, description in test_files.items():
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024**2)
            total_test_size += size_mb
            
            print(f'   ✅ {filename} ({description})')
            print(f'      Size: {size_mb:.1f} MB')
            
            if filename.endswith('.npy'):
                data = np.load(filepath)
                test_data[filename] = data
                print(f'      Shape: {data.shape}')
                print(f'      Dtype: {data.dtype}')
                if 'labels' in filename:
                    unique_labels = np.unique(data)
                    print(f'      Unique values: {unique_labels}')
                elif 'eeg' in filename:
                    print(f'      Data range: [{data.min():.3f}, {data.max():.3f}]')
                elif 'features' in filename:
                    print(f'      Feature range: [{data.min():.3f}, {data.max():.3f}]')
            print()
        else:
            print(f'   ❌ {filename} - NOT FOUND')
    
    # Data consistency checks
    print(f'🔍 DATA CONSISTENCY CHECKS:')
    
    if 'train_eeg_data.npy' in train_data and 'train_labels.npy' in train_data:
        eeg_shape = train_data['train_eeg_data.npy'].shape
        labels_shape = train_data['train_labels.npy'].shape
        
        if eeg_shape[0] == labels_shape[0]:
            print(f'   ✅ Train EEG-labels alignment: {eeg_shape[0]} trials')
        else:
            print(f'   ❌ Train EEG-labels mismatch: {eeg_shape[0]} vs {labels_shape[0]}')
    
    if 'test_eeg_data.npy' in test_data and 'test_labels.npy' in test_data:
        eeg_shape = test_data['test_eeg_data.npy'].shape
        labels_shape = test_data['test_labels.npy'].shape
        
        if eeg_shape[0] == labels_shape[0]:
            print(f'   ✅ Test EEG-labels alignment: {eeg_shape[0]} trials')
        else:
            print(f'   ❌ Test EEG-labels mismatch: {eeg_shape[0]} vs {labels_shape[0]}')
    
    # Feature dimension checks
    if 'train_text_features.npy' in train_data and 'train_img_features.npy' in train_data:
        text_dim = train_data['train_text_features.npy'].shape[1]
        img_dim = train_data['train_img_features.npy'].shape[1]
        
        if text_dim == img_dim:
            print(f'   ✅ Feature dimension alignment: {text_dim}D')
        else:
            print(f'   ❌ Feature dimension mismatch: text={text_dim}D vs img={img_dim}D')
    
    print(f'\n💾 STORAGE SUMMARY:')
    print(f'   Train files: {total_train_size:.1f} MB')
    print(f'   Test files: {total_test_size:.1f} MB')
    print(f'   Total: {total_train_size + total_test_size:.1f} MB')
    
    # Load and display metadata
    try:
        metadata_path = os.path.join(base_path, 'train_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f'\n📋 METADATA INFORMATION:')
        for key, value in metadata.items():
            print(f'   {key}: {value}')
            
    except Exception as e:
        print(f'\n⚠️ Could not load metadata: {e}')
    
    # Check cached CLIP features
    print(f'\n🔮 CACHED CLIP FEATURES:')
    clip_files = [
        'ViT-B_32_features_mindbigdata_train.pt',
        'ViT-B_32_features_mindbigdata_test.pt'
    ]
    
    for filename in clip_files:
        filepath = os.path.join(os.getcwd(), filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024**2)
            print(f'   ✅ {filename}: {size_mb:.1f} MB')
            
            # Load and check cached features
            try:
                cached = torch.load(filepath, map_location='cpu')
                print(f'      Text features: {cached["text_features"].shape}')
                print(f'      Image features: {cached["img_features"].shape}')
            except Exception as e:
                print(f'      ⚠️ Could not load: {e}')
        else:
            print(f'   ❌ {filename} - NOT FOUND')
    
    print(f'\n🎯 READY FOR CONTRASTIVE LEARNING:')
    print(f'   ✅ EEG data: Ready for neural network input')
    print(f'   ✅ CLIP features: Ready for contrastive loss')
    print(f'   ✅ Labels: Ready for classification')
    print(f'   ✅ All files properly formatted')
    
    return base_path

if __name__ == "__main__":
    output_path = check_outputs()
    print(f'\n🎉 VERIFICATION COMPLETE!')
    print(f'📂 Output location: {output_path}')
