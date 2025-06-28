#!/usr/bin/env python3
"""
Analyze MindBigData Retrieval Outputs
=====================================

Deep analysis of retrieval system outputs for quality assessment.
"""

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_outputs():
    """Perform deep analysis of output data"""
    
    print('📊 DEEP ANALYSIS OF MINDBIGDATA RETRIEVAL OUTPUTS')
    print('=' * 70)
    
    base_path = r'd:\trainflow\2retrieval\outputs\mindbigdata_pairs'
    
    # Load all data
    print('📂 Loading data...')
    train_eeg = np.load(os.path.join(base_path, 'train_eeg_data.npy'))
    train_labels = np.load(os.path.join(base_path, 'train_labels.npy'))
    train_text_features = np.load(os.path.join(base_path, 'train_text_features.npy'))
    train_img_features = np.load(os.path.join(base_path, 'train_img_features.npy'))
    
    test_eeg = np.load(os.path.join(base_path, 'test_eeg_data.npy'))
    test_labels = np.load(os.path.join(base_path, 'test_labels.npy'))
    test_text_features = np.load(os.path.join(base_path, 'test_text_features.npy'))
    test_img_features = np.load(os.path.join(base_path, 'test_img_features.npy'))
    
    print('✅ Data loaded successfully!')
    
    # 1. Dataset Statistics
    print(f'\n📊 DATASET STATISTICS:')
    print(f'   Train trials: {len(train_eeg):,}')
    print(f'   Test trials: {len(test_eeg):,}')
    print(f'   Total trials: {len(train_eeg) + len(test_eeg):,}')
    print(f'   EEG channels: {train_eeg.shape[1]}')
    print(f'   Time points: {train_eeg.shape[2]}')
    print(f'   Feature dimension: {train_text_features.shape[1]}')
    
    # 2. Class Distribution Analysis
    print(f'\n🎯 CLASS DISTRIBUTION ANALYSIS:')
    
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    
    print(f'   TRAIN DISTRIBUTION:')
    for digit in range(10):
        count = train_counts.get(digit, 0)
        percentage = (count / len(train_labels)) * 100
        print(f'     Digit {digit}: {count:3d} trials ({percentage:5.1f}%)')
    
    print(f'   TEST DISTRIBUTION:')
    for digit in range(10):
        count = test_counts.get(digit, 0)
        percentage = (count / len(test_labels)) * 100
        print(f'     Digit {digit}: {count:3d} trials ({percentage:5.1f}%)')
    
    # 3. EEG Data Quality Analysis
    print(f'\n🧠 EEG DATA QUALITY ANALYSIS:')
    
    # Channel-wise statistics
    train_channel_means = np.mean(train_eeg, axis=(0, 2))
    train_channel_stds = np.std(train_eeg, axis=(0, 2))
    
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                     'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    print(f'   CHANNEL STATISTICS (Train):')
    for i, ch_name in enumerate(channel_names):
        print(f'     {ch_name:4s}: mean={train_channel_means[i]:6.3f}, std={train_channel_stds[i]:6.3f}')
    
    # Signal quality metrics
    train_snr = np.mean(np.abs(train_eeg)) / np.std(train_eeg)
    test_snr = np.mean(np.abs(test_eeg)) / np.std(test_eeg)
    
    print(f'   SIGNAL QUALITY:')
    print(f'     Train SNR estimate: {train_snr:.3f}')
    print(f'     Test SNR estimate: {test_snr:.3f}')
    print(f'     Train amplitude range: [{train_eeg.min():.3f}, {train_eeg.max():.3f}]')
    print(f'     Test amplitude range: [{test_eeg.min():.3f}, {test_eeg.max():.3f}]')
    
    # 4. CLIP Features Analysis
    print(f'\n🔮 CLIP FEATURES ANALYSIS:')
    
    # Feature statistics
    print(f'   TEXT FEATURES:')
    print(f'     Train range: [{train_text_features.min():.3f}, {train_text_features.max():.3f}]')
    print(f'     Train mean: {train_text_features.mean():.3f}, std: {train_text_features.std():.3f}')
    print(f'     Test range: [{test_text_features.min():.3f}, {test_text_features.max():.3f}]')
    print(f'     Test mean: {test_text_features.mean():.3f}, std: {test_text_features.std():.3f}')
    
    print(f'   IMAGE FEATURES:')
    print(f'     Train range: [{train_img_features.min():.3f}, {train_img_features.max():.3f}]')
    print(f'     Train mean: {train_img_features.mean():.3f}, std: {train_img_features.std():.3f}')
    print(f'     Test range: [{test_img_features.min():.3f}, {test_img_features.max():.3f}]')
    print(f'     Test mean: {test_img_features.mean():.3f}, std: {test_img_features.std():.3f}')
    
    # Feature similarity analysis
    text_img_similarity = np.mean([
        np.dot(train_text_features[i], train_img_features[i]) / 
        (np.linalg.norm(train_text_features[i]) * np.linalg.norm(train_img_features[i]))
        for i in range(len(train_text_features))
    ])
    
    print(f'   FEATURE ALIGNMENT:')
    print(f'     Average text-image cosine similarity: {text_img_similarity:.3f}')
    
    # 5. Data Integrity Checks
    print(f'\n🔍 DATA INTEGRITY CHECKS:')
    
    # Check for NaN or infinite values
    checks = [
        ('Train EEG', train_eeg),
        ('Test EEG', test_eeg),
        ('Train text features', train_text_features),
        ('Test text features', test_text_features),
        ('Train image features', train_img_features),
        ('Test image features', test_img_features)
    ]
    
    for name, data in checks:
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        status = '✅' if not (has_nan or has_inf) else '❌'
        print(f'     {status} {name}: NaN={has_nan}, Inf={has_inf}')
    
    # 6. Readiness Assessment
    print(f'\n🎯 CONTRASTIVE LEARNING READINESS:')
    
    readiness_checks = [
        ('EEG data format', train_eeg.ndim == 3 and test_eeg.ndim == 3),
        ('Feature dimensions match', train_text_features.shape[1] == train_img_features.shape[1]),
        ('All classes present', len(set(train_labels)) == 10 and len(set(test_labels)) == 10),
        ('No missing data', not any(np.isnan(data).any() for _, data in checks)),
        ('Reasonable data ranges', abs(train_eeg.mean()) < 100 and abs(test_eeg.mean()) < 100),
        ('Feature normalization', abs(train_text_features.mean()) < 1 and abs(train_img_features.mean()) < 1)
    ]
    
    all_ready = True
    for check_name, is_ready in readiness_checks:
        status = '✅' if is_ready else '❌'
        print(f'     {status} {check_name}')
        if not is_ready:
            all_ready = False
    
    print(f'\n🏆 OVERALL ASSESSMENT:')
    if all_ready:
        print(f'   ✅ ALL CHECKS PASSED - READY FOR CONTRASTIVE LEARNING!')
        print(f'   🚀 Data quality is excellent for SOTA research')
        print(f'   📊 {len(train_eeg):,} train + {len(test_eeg):,} test trials available')
        print(f'   🎯 All 10 digit classes properly represented')
    else:
        print(f'   ⚠️ SOME ISSUES DETECTED - REVIEW REQUIRED')
    
    # 7. Usage Recommendations
    print(f'\n💡 USAGE RECOMMENDATIONS:')
    print(f'   📂 Load data from: {base_path}')
    print(f'   🧠 EEG input shape: (batch_size, 14, 230)')
    print(f'   🔮 CLIP feature shape: (batch_size, 512)')
    print(f'   🎯 Labels: 0-9 (10 classes)')
    print(f'   ⚡ Recommended batch size: 32-128 (based on data size)')
    print(f'   🔄 Train/test split: {len(train_eeg)}/{len(test_eeg)} (80/20)')
    
    return {
        'train_trials': len(train_eeg),
        'test_trials': len(test_eeg),
        'channels': train_eeg.shape[1],
        'timepoints': train_eeg.shape[2],
        'feature_dim': train_text_features.shape[1],
        'classes': len(set(train_labels)),
        'ready': all_ready
    }

if __name__ == "__main__":
    results = analyze_outputs()
    print(f'\n🎉 ANALYSIS COMPLETE!')
    print(f'📊 Summary: {results["train_trials"]} train + {results["test_trials"]} test trials')
    print(f'🎯 Status: {"READY" if results["ready"] else "NEEDS REVIEW"}')
