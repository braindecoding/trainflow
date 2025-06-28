#!/usr/bin/env python3
"""
Compare Two outputs_final Directories
===================================

Check if 1loaddata and 2retrieval outputs_final are identical
"""

import numpy as np
import os

def compare_outputs_final():
    """Compare the two outputs_final directories"""
    
    print("🔍 COMPARING OUTPUTS_FINAL DIRECTORIES")
    print("=" * 60)
    
    path1 = r"d:\trainflow\1loaddata\mindbigdata2\outputs_final"
    path2 = r"d:\trainflow\2retrieval\mindbigdata\outputs_final"
    
    print(f"📂 Path 1: {path1}")
    print(f"📂 Path 2: {path2}")
    
    # Check if both exist
    if not os.path.exists(path1):
        print(f"❌ Path 1 does not exist")
        return
    
    if not os.path.exists(path2):
        print(f"❌ Path 2 does not exist")
        return
    
    # Files to compare
    files_to_compare = [
        "train_data.npy",
        "train_labels.npy", 
        "test_data.npy",
        "test_labels.npy"
    ]
    
    print(f"\n📊 COMPARING FILES:")
    
    all_identical = True
    
    for filename in files_to_compare:
        file1 = os.path.join(path1, filename)
        file2 = os.path.join(path2, filename)
        
        print(f"\n🔍 {filename}:")
        
        # Check if files exist
        if not os.path.exists(file1):
            print(f"   ❌ File 1 missing")
            all_identical = False
            continue
            
        if not os.path.exists(file2):
            print(f"   ❌ File 2 missing")
            all_identical = False
            continue
        
        # Compare file sizes
        size1 = os.path.getsize(file1)
        size2 = os.path.getsize(file2)
        
        print(f"   File sizes: {size1:,} vs {size2:,} bytes")
        
        if size1 != size2:
            print(f"   ❌ Different file sizes")
            all_identical = False
            continue
        
        # Load and compare data
        try:
            data1 = np.load(file1)
            data2 = np.load(file2)
            
            print(f"   Shapes: {data1.shape} vs {data2.shape}")
            
            if data1.shape != data2.shape:
                print(f"   ❌ Different shapes")
                all_identical = False
                continue
            
            # Check if data is identical
            if np.array_equal(data1, data2):
                print(f"   ✅ Files are identical")
            else:
                # Check if they're close (floating point precision)
                if np.allclose(data1, data2, rtol=1e-15, atol=1e-15):
                    print(f"   ✅ Files are nearly identical (floating point precision)")
                else:
                    print(f"   ❌ Files have different content")
                    
                    # Show some statistics about differences
                    if data1.dtype == data2.dtype and np.issubdtype(data1.dtype, np.number):
                        diff = np.abs(data1 - data2)
                        max_diff = np.max(diff)
                        mean_diff = np.mean(diff)
                        print(f"      Max difference: {max_diff}")
                        print(f"      Mean difference: {mean_diff}")
                    
                    all_identical = False
            
        except Exception as e:
            print(f"   ❌ Error comparing files: {e}")
            all_identical = False
    
    print(f"\n🏆 FINAL RESULT:")
    if all_identical:
        print(f"   ✅ ALL FILES ARE IDENTICAL")
        print(f"   ✅ Both outputs_final directories contain the same data")
        print(f"   ✅ Either can be used as source for retrieval")
    else:
        print(f"   ❌ FILES ARE DIFFERENT")
        print(f"   ❌ The directories contain different data")
    
    # Check which one is actually used by retrieval
    print(f"\n🔍 CHECKING WHICH IS USED BY RETRIEVAL:")
    
    # Look at the retrieval script configuration
    retrieval_script = r"d:\trainflow\2retrieval\mindbigdata\mindbigdata_retrieval.py"
    
    if os.path.exists(retrieval_script):
        with open(retrieval_script, 'r') as f:
            content = f.read()
            
        if "1loaddata\\mindbigdata2\\outputs_final" in content:
            print(f"   📂 Retrieval script uses: 1loaddata/mindbigdata2/outputs_final")
        elif "2retrieval\\mindbigdata\\outputs_final" in content:
            print(f"   📂 Retrieval script uses: 2retrieval/mindbigdata/outputs_final")
        else:
            print(f"   ⚠️ Could not determine which path is used")
    else:
        print(f"   ❌ Retrieval script not found")

if __name__ == "__main__":
    compare_outputs_final()
