#!/usr/bin/env python3
"""
Check Data Source Consistency
============================

Verify if 2retrieval/outputs/mindbigdata_pairs uses input from 1loaddata/mindbigdata2/outputs_final
"""

import numpy as np
import json
import os
from collections import Counter

def check_data_consistency():
    """Check if retrieval data matches preprocessing output"""
    
    print("🔍 CHECKING DATA SOURCE CONSISTENCY")
    print("=" * 60)
    
    # Paths
    preprocessing_path = r"d:\trainflow\1loaddata\mindbigdata2\outputs_final"
    retrieval_path = r"d:\trainflow\2retrieval\outputs\mindbigdata_pairs"
    
    print(f"📂 Preprocessing path: {preprocessing_path}")
    print(f"📂 Retrieval path: {retrieval_path}")
    
    # Check if paths exist
    if not os.path.exists(preprocessing_path):
        print(f"❌ Preprocessing path not found!")
        return
    
    if not os.path.exists(retrieval_path):
        print(f"❌ Retrieval path not found!")
        return
    
    print(f"\n📊 LOADING DATA...")
    
    # Load preprocessing data
    try:
        prep_train_data = np.load(os.path.join(preprocessing_path, "train_data.npy"))
        prep_train_labels = np.load(os.path.join(preprocessing_path, "train_labels.npy"))
        prep_test_data = np.load(os.path.join(preprocessing_path, "test_data.npy"))
        prep_test_labels = np.load(os.path.join(preprocessing_path, "test_labels.npy"))
        
        print(f"✅ Preprocessing data loaded:")
        print(f"   Train: {prep_train_data.shape}, Labels: {prep_train_labels.shape}")
        print(f"   Test:  {prep_test_data.shape}, Labels: {prep_test_labels.shape}")
        
    except Exception as e:
        print(f"❌ Error loading preprocessing data: {e}")
        return
    
    # Load retrieval data
    try:
        retr_train_data = np.load(os.path.join(retrieval_path, "train_eeg_data.npy"))
        retr_train_labels = np.load(os.path.join(retrieval_path, "train_labels.npy"))
        retr_test_data = np.load(os.path.join(retrieval_path, "test_eeg_data.npy"))
        retr_test_labels = np.load(os.path.join(retrieval_path, "test_labels.npy"))
        
        print(f"✅ Retrieval data loaded:")
        print(f"   Train: {retr_train_data.shape}, Labels: {retr_train_labels.shape}")
        print(f"   Test:  {retr_test_data.shape}, Labels: {retr_test_labels.shape}")
        
    except Exception as e:
        print(f"❌ Error loading retrieval data: {e}")
        return
    
    print(f"\n🔍 COMPARING DIMENSIONS...")
    
    # Compare shapes
    train_shape_match = prep_train_data.shape == retr_train_data.shape
    test_shape_match = prep_test_data.shape == retr_test_data.shape
    train_labels_match = prep_train_labels.shape == retr_train_labels.shape
    test_labels_match = prep_test_labels.shape == retr_test_labels.shape
    
    print(f"   Train data shapes: {prep_train_data.shape} vs {retr_train_data.shape} {'✅' if train_shape_match else '❌'}")
    print(f"   Test data shapes:  {prep_test_data.shape} vs {retr_test_data.shape} {'✅' if test_shape_match else '❌'}")
    print(f"   Train label shapes: {prep_train_labels.shape} vs {retr_train_labels.shape} {'✅' if train_labels_match else '❌'}")
    print(f"   Test label shapes:  {prep_test_labels.shape} vs {retr_test_labels.shape} {'✅' if test_labels_match else '❌'}")
    
    print(f"\n🔍 COMPARING DATA CONTENT...")
    
    # Compare actual data values (sample comparison)
    if train_shape_match:
        train_data_identical = np.allclose(prep_train_data, retr_train_data, rtol=1e-10, atol=1e-10)
        print(f"   Train data identical: {'✅' if train_data_identical else '❌'}")
        
        if not train_data_identical:
            # Check if it's just a precision difference
            max_diff = np.max(np.abs(prep_train_data - retr_train_data))
            mean_diff = np.mean(np.abs(prep_train_data - retr_train_data))
            print(f"     Max difference: {max_diff}")
            print(f"     Mean difference: {mean_diff}")
    
    if test_shape_match:
        test_data_identical = np.allclose(prep_test_data, retr_test_data, rtol=1e-10, atol=1e-10)
        print(f"   Test data identical: {'✅' if test_data_identical else '❌'}")
        
        if not test_data_identical:
            max_diff = np.max(np.abs(prep_test_data - retr_test_data))
            mean_diff = np.mean(np.abs(prep_test_data - retr_test_data))
            print(f"     Max difference: {max_diff}")
            print(f"     Mean difference: {mean_diff}")
    
    # Compare labels
    if train_labels_match:
        train_labels_identical = np.array_equal(prep_train_labels, retr_train_labels)
        print(f"   Train labels identical: {'✅' if train_labels_identical else '❌'}")
    
    if test_labels_match:
        test_labels_identical = np.array_equal(prep_test_labels, retr_test_labels)
        print(f"   Test labels identical: {'✅' if test_labels_identical else '❌'}")
    
    print(f"\n📊 COMPARING CLASS DISTRIBUTIONS...")
    
    # Compare class distributions
    prep_train_dist = Counter(prep_train_labels)
    retr_train_dist = Counter(retr_train_labels)
    prep_test_dist = Counter(prep_test_labels)
    retr_test_dist = Counter(retr_test_labels)
    
    print(f"   TRAIN DISTRIBUTION COMPARISON:")
    print(f"   {'Digit':<6} {'Preprocessing':<15} {'Retrieval':<15} {'Match':<6}")
    print(f"   {'-'*6} {'-'*15} {'-'*15} {'-'*6}")
    
    for digit in range(10):
        prep_count = prep_train_dist.get(digit, 0)
        retr_count = retr_train_dist.get(digit, 0)
        match = prep_count == retr_count
        print(f"   {digit:<6} {prep_count:<15} {retr_count:<15} {'✅' if match else '❌':<6}")
    
    print(f"\n   TEST DISTRIBUTION COMPARISON:")
    print(f"   {'Digit':<6} {'Preprocessing':<15} {'Retrieval':<15} {'Match':<6}")
    print(f"   {'-'*6} {'-'*15} {'-'*15} {'-'*6}")
    
    for digit in range(10):
        prep_count = prep_test_dist.get(digit, 0)
        retr_count = retr_test_dist.get(digit, 0)
        match = prep_count == retr_count
        print(f"   {digit:<6} {prep_count:<15} {retr_count:<15} {'✅' if match else '❌':<6}")
    
    # Check metadata if available
    print(f"\n📋 CHECKING METADATA...")
    
    try:
        with open(os.path.join(retrieval_path, "train_metadata.json"), "r") as f:
            train_metadata = json.load(f)
        
        print(f"✅ Train metadata found:")
        for key, value in train_metadata.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"⚠️ Could not load train metadata: {e}")
    
    try:
        with open(os.path.join(retrieval_path, "test_metadata.json"), "r") as f:
            test_metadata = json.load(f)
        
        print(f"✅ Test metadata found:")
        for key, value in test_metadata.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"⚠️ Could not load test metadata: {e}")
    
    # Final assessment
    print(f"\n🏆 FINAL ASSESSMENT:")
    
    all_shapes_match = train_shape_match and test_shape_match and train_labels_match and test_labels_match
    
    if all_shapes_match:
        if train_shape_match and test_shape_match:
            data_match = (np.allclose(prep_train_data, retr_train_data, rtol=1e-6) and 
                         np.allclose(prep_test_data, retr_test_data, rtol=1e-6))
            labels_match = (np.array_equal(prep_train_labels, retr_train_labels) and
                           np.array_equal(prep_test_labels, retr_test_labels))
            
            if data_match and labels_match:
                print(f"   ✅ CONFIRMED: Retrieval data uses preprocessing output as input")
                print(f"   ✅ Data integrity: Perfect match")
                print(f"   ✅ Pipeline consistency: Maintained")
            else:
                print(f"   ⚠️ PARTIAL MATCH: Same dimensions but different values")
                print(f"   ⚠️ Possible data transformation or different source")
        else:
            print(f"   ❌ MISMATCH: Different data dimensions")
            print(f"   ❌ Likely using different input source")
    else:
        print(f"   ❌ MAJOR MISMATCH: Completely different datasets")
        print(f"   ❌ Retrieval data NOT using preprocessing output")
    
    # Additional file size comparison
    print(f"\n💾 FILE SIZE COMPARISON:")
    
    prep_train_size = os.path.getsize(os.path.join(preprocessing_path, "train_data.npy")) / (1024**2)
    prep_test_size = os.path.getsize(os.path.join(preprocessing_path, "test_data.npy")) / (1024**2)
    retr_train_size = os.path.getsize(os.path.join(retrieval_path, "train_eeg_data.npy")) / (1024**2)
    retr_test_size = os.path.getsize(os.path.join(retrieval_path, "test_eeg_data.npy")) / (1024**2)
    
    print(f"   Preprocessing train: {prep_train_size:.1f} MB")
    print(f"   Retrieval train:     {retr_train_size:.1f} MB")
    print(f"   Preprocessing test:  {prep_test_size:.1f} MB")
    print(f"   Retrieval test:      {retr_test_size:.1f} MB")
    
    size_match = (abs(prep_train_size - retr_train_size) < 1.0 and 
                  abs(prep_test_size - retr_test_size) < 1.0)
    print(f"   File sizes match: {'✅' if size_match else '❌'}")

if __name__ == "__main__":
    check_data_consistency()
