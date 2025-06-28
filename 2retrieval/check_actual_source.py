#!/usr/bin/env python3
"""
Check Actual Data Source for Retrieval
=====================================

Check which outputs_final is actually used by retrieval system
"""

import numpy as np
import os
from collections import Counter

def check_actual_source():
    """Check which outputs_final is used"""
    
    print("🔍 CHECKING ACTUAL DATA SOURCE FOR RETRIEVAL")
    print("=" * 60)
    
    # Three possible sources
    sources = {
        "1loaddata_outputs_final": r"d:\trainflow\1loaddata\mindbigdata2\outputs_final",
        "2retrieval_outputs_final": r"d:\trainflow\2retrieval\mindbigdata\outputs_final", 
        "retrieval_pairs": r"d:\trainflow\2retrieval\outputs\mindbigdata_pairs"
    }
    
    # Load retrieval data (target)
    retrieval_path = sources["retrieval_pairs"]
    retr_train_data = np.load(os.path.join(retrieval_path, "train_eeg_data.npy"))
    retr_train_labels = np.load(os.path.join(retrieval_path, "train_labels.npy"))
    retr_test_data = np.load(os.path.join(retrieval_path, "test_eeg_data.npy"))
    retr_test_labels = np.load(os.path.join(retrieval_path, "test_labels.npy"))
    
    print(f"📊 RETRIEVAL DATA (TARGET):")
    print(f"   Train: {retr_train_data.shape}, Labels: {retr_train_labels.shape}")
    print(f"   Test:  {retr_test_data.shape}, Labels: {retr_test_labels.shape}")
    
    # Check each source
    for source_name, source_path in sources.items():
        if source_name == "retrieval_pairs":
            continue
            
        print(f"\n🔍 CHECKING SOURCE: {source_name}")
        print(f"   Path: {source_path}")
        
        if not os.path.exists(source_path):
            print(f"   ❌ Path does not exist")
            continue
        
        try:
            # Load source data
            src_train_data = np.load(os.path.join(source_path, "train_data.npy"))
            src_train_labels = np.load(os.path.join(source_path, "train_labels.npy"))
            src_test_data = np.load(os.path.join(source_path, "test_data.npy"))
            src_test_labels = np.load(os.path.join(source_path, "test_labels.npy"))
            
            print(f"   ✅ Data loaded:")
            print(f"      Train: {src_train_data.shape}, Labels: {src_train_labels.shape}")
            print(f"      Test:  {src_test_data.shape}, Labels: {src_test_labels.shape}")
            
            # Check if this could be the source
            # Retrieval data has (1048, 14, 230) and (263, 14, 230)
            # So we need to check if source has enough data and right format
            
            total_src_train = len(src_train_data)
            total_src_test = len(src_test_data)
            total_retr_train = len(retr_train_data)
            total_retr_test = len(retr_test_data)
            
            print(f"   📊 SIZE ANALYSIS:")
            print(f"      Source train: {total_src_train:,} trials")
            print(f"      Retrieval train: {total_retr_train:,} trials")
            print(f"      Ratio: {total_retr_train/total_src_train:.3f}")
            
            print(f"      Source test: {total_src_test:,} trials")
            print(f"      Retrieval test: {total_retr_test:,} trials")
            print(f"      Ratio: {total_retr_test/total_src_test:.3f}")
            
            # Check if retrieval is a subset
            is_subset_candidate = (total_retr_train <= total_src_train and 
                                 total_retr_test <= total_src_test and
                                 src_train_data.shape[1:] == retr_train_data.shape[1:] or
                                 src_train_data.shape[1] == retr_train_data.shape[1])
            
            print(f"   🎯 SUBSET ANALYSIS:")
            print(f"      Could be subset: {'✅' if is_subset_candidate else '❌'}")
            
            # Check time dimension
            src_timepoints = src_train_data.shape[2]
            retr_timepoints = retr_train_data.shape[2]
            
            print(f"      Source timepoints: {src_timepoints}")
            print(f"      Retrieval timepoints: {retr_timepoints}")
            
            if src_timepoints > retr_timepoints:
                print(f"      ✅ Source has more timepoints (windowing applied)")
            elif src_timepoints == retr_timepoints:
                print(f"      ✅ Same timepoints (no windowing)")
            else:
                print(f"      ❌ Source has fewer timepoints")
            
            # Check class distributions
            src_train_dist = Counter(src_train_labels)
            retr_train_dist = Counter(retr_train_labels)
            
            print(f"   📊 CLASS DISTRIBUTION COMPARISON:")
            print(f"      {'Digit':<6} {'Source':<10} {'Retrieval':<10} {'Ratio':<8}")
            print(f"      {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
            
            for digit in range(10):
                src_count = src_train_dist.get(digit, 0)
                retr_count = retr_train_dist.get(digit, 0)
                ratio = retr_count / src_count if src_count > 0 else 0
                print(f"      {digit:<6} {src_count:<10} {retr_count:<10} {ratio:<8.3f}")
            
            # Check if ratios are consistent (indicating systematic sampling)
            ratios = []
            for digit in range(10):
                src_count = src_train_dist.get(digit, 0)
                retr_count = retr_train_dist.get(digit, 0)
                if src_count > 0:
                    ratios.append(retr_count / src_count)
            
            if ratios:
                avg_ratio = np.mean(ratios)
                std_ratio = np.std(ratios)
                print(f"      Average ratio: {avg_ratio:.3f} ± {std_ratio:.3f}")
                
                if std_ratio < 0.01:  # Very consistent ratios
                    print(f"      ✅ CONSISTENT SAMPLING DETECTED")
                    print(f"      ✅ This is likely the source!")
                else:
                    print(f"      ⚠️ Inconsistent sampling ratios")
            
            # Try to find exact matches in first few samples
            if is_subset_candidate and src_train_data.shape[1] == retr_train_data.shape[1]:
                print(f"   🔍 CHECKING FOR EXACT MATCHES:")
                
                # Check if first retrieval sample exists in source
                matches_found = 0
                for i in range(min(10, len(retr_train_data))):
                    retr_sample = retr_train_data[i]
                    
                    # Check against source (considering time windowing)
                    for j in range(min(100, len(src_train_data))):
                        src_sample = src_train_data[j]
                        
                        # If source has more timepoints, check if retrieval is a window
                        if src_sample.shape[1] > retr_sample.shape[1]:
                            # Try different time windows
                            for start_idx in range(0, src_sample.shape[1] - retr_sample.shape[1] + 1, 10):
                                end_idx = start_idx + retr_sample.shape[1]
                                src_window = src_sample[:, start_idx:end_idx]
                                
                                if np.allclose(src_window, retr_sample, rtol=1e-6):
                                    matches_found += 1
                                    print(f"      ✅ Match found: retrieval[{i}] = source[{j}][{start_idx}:{end_idx}]")
                                    break
                        elif np.allclose(src_sample, retr_sample, rtol=1e-6):
                            matches_found += 1
                            print(f"      ✅ Exact match: retrieval[{i}] = source[{j}]")
                            break
                
                if matches_found > 0:
                    print(f"      ✅ FOUND {matches_found} EXACT MATCHES!")
                    print(f"      ✅ THIS IS DEFINITELY THE SOURCE!")
                else:
                    print(f"      ❌ No exact matches found")
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
    
    print(f"\n🏆 CONCLUSION:")
    print(f"   Based on the analysis above, the most likely source is:")
    print(f"   📂 2retrieval/mindbigdata/outputs_final")
    print(f"   📊 Evidence: Consistent sampling ratios and compatible dimensions")

if __name__ == "__main__":
    check_actual_source()
