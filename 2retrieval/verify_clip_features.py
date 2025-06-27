#!/usr/bin/env python3
"""
Verify CLIP Feature Extraction
==============================

Verify that CLIP features were properly extracted and saved.
"""

import numpy as np

def verify_clip_features():
    """Verify CLIP feature extraction results"""
    
    print('🔍 VERIFYING CLIP FEATURE EXTRACTION...')
    print('=' * 60)
    
    # Load CLIP features
    base_path = r'd:\trainflow\2retrieval\outputs\mindbigdata_pairs'
    train_text_feat = np.load(f'{base_path}/train_text_features.npy')
    train_img_feat = np.load(f'{base_path}/train_img_features.npy')
    test_text_feat = np.load(f'{base_path}/test_text_features.npy')
    test_img_feat = np.load(f'{base_path}/test_img_features.npy')
    
    print('📊 CLIP FEATURES VALIDATION:')
    print(f'   Train text features: {train_text_feat.shape}')
    print(f'   Train image features: {train_img_feat.shape}')
    print(f'   Test text features: {test_text_feat.shape}')
    print(f'   Test image features: {test_img_feat.shape}')
    
    print(f'\n🔬 CLIP FEATURE PROPERTIES:')
    print(f'   Text feature range: [{train_text_feat.min():.3f}, {train_text_feat.max():.3f}]')
    print(f'   Image feature range: [{train_img_feat.min():.3f}, {train_img_feat.max():.3f}]')
    
    # Check if features are normalized
    text_norms = np.linalg.norm(train_text_feat, axis=1)
    img_norms = np.linalg.norm(train_img_feat, axis=1)
    
    print(f'\n📐 NORMALIZATION CHECK:')
    print(f'   Text feature norms: [{text_norms.min():.3f}, {text_norms.max():.3f}]')
    print(f'   Image feature norms: [{img_norms.min():.3f}, {img_norms.max():.3f}]')
    print(f'   Text norm mean: {text_norms.mean():.3f}')
    print(f'   Image norm mean: {img_norms.mean():.3f}')
    
    print(f'\n🎯 CLIP MODEL VERIFICATION:')
    print(f'   Feature dimension: {train_text_feat.shape[1]} (ViT-B/32 = 512)')
    
    # Check normalization
    text_normalized = abs(text_norms.mean() - 1.0) < 0.01
    img_normalized = abs(img_norms.mean() - 1.0) < 0.01
    
    if text_normalized and img_normalized:
        print('   Normalization: ✅ Properly normalized')
    else:
        print('   Normalization: ⚠️ May not be normalized')
    
    # Check feature diversity
    print(f'\n🔍 FEATURE DIVERSITY CHECK:')
    
    # Check if all features are the same (would indicate a problem)
    text_unique = len(np.unique(train_text_feat.round(3), axis=0))
    img_unique = len(np.unique(train_img_feat.round(3), axis=0))
    
    print(f'   Unique text features: {text_unique} (should be 10 for digits)')
    print(f'   Unique image features: {img_unique} (should be 10 for digits)')
    
    # Sample some features
    print(f'\n📋 SAMPLE FEATURES:')
    print(f'   Text feature sample (first 5 dims):')
    for i in range(min(3, len(train_text_feat))):
        print(f'     Sample {i}: {train_text_feat[i][:5]}')
    
    print(f'   Image feature sample (first 5 dims):')
    for i in range(min(3, len(train_img_feat))):
        print(f'     Sample {i}: {train_img_feat[i][:5]}')
    
    # Verify CLIP feature extraction success
    print(f'\n✅ CLIP FEATURE EXTRACTION VERIFICATION:')
    
    checks = [
        (train_text_feat.shape[1] == 512, "Feature dimension is 512 (ViT-B/32)"),
        (text_normalized and img_normalized, "Features are normalized"),
        (text_unique <= 10, "Text features match digit count"),
        (img_unique <= 10, "Image features match digit count"),
        (train_text_feat.shape[0] == 51900, "Train set has correct size"),
        (test_text_feat.shape[0] == 12975, "Test set has correct size")
    ]
    
    all_passed = True
    for check, description in checks:
        status = "✅" if check else "❌"
        print(f'   {status} {description}')
        if not check:
            all_passed = False
    
    if all_passed:
        print(f'\n🎉 ALL CHECKS PASSED!')
        print(f'   CLIP feature extraction was successful')
        print(f'   Features are ready for contrastive learning')
    else:
        print(f'\n⚠️ SOME CHECKS FAILED!')
        print(f'   Please review the feature extraction process')
    
    return all_passed


if __name__ == "__main__":
    verify_clip_features()
