#!/usr/bin/env python3
"""
Display ATMS Learning Curves
===========================

Script to open and display the generated learning curve visualizations.
"""

import os
import subprocess
import sys

def open_image(image_path):
    """Open image with default system viewer"""
    try:
        if os.path.exists(image_path):
            if sys.platform.startswith('win'):
                os.startfile(image_path)
            elif sys.platform.startswith('darwin'):  # macOS
                subprocess.run(['open', image_path])
            else:  # Linux
                subprocess.run(['xdg-open', image_path])
            print(f"✅ Opened: {image_path}")
            return True
        else:
            print(f"❌ File not found: {image_path}")
            return False
    except Exception as e:
        print(f"❌ Error opening {image_path}: {e}")
        return False

def main():
    """Main function to display learning curves"""
    print("📊 ATMS LEARNING CURVES DISPLAY")
    print("=" * 50)
    
    # Learning curve files
    learning_curves = [
        {
            'file': 'atms_learning_curves.png',
            'description': '📈 Main Learning Curves (Loss, Accuracy, Overfitting Analysis)',
            'size_mb': 0.9
        },
        {
            'file': 'atms_comparison_curves.png', 
            'description': '📊 Full Dataset vs Subset Comparison',
            'size_mb': 0.3
        }
    ]
    
    print(f"\n📁 Available learning curve visualizations:")
    for i, curve in enumerate(learning_curves, 1):
        file_path = curve['file']
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024**2)  # MB
            print(f"   {i}. {curve['description']}")
            print(f"      File: {file_path} ({file_size:.1f} MB)")
        else:
            print(f"   {i}. {curve['description']} - ❌ NOT FOUND")
    
    print(f"\n🖼️ Opening learning curve visualizations...")
    
    opened_count = 0
    for curve in learning_curves:
        file_path = curve['file']
        print(f"\n📂 {curve['description']}")
        if open_image(file_path):
            opened_count += 1
    
    print(f"\n🎉 LEARNING CURVES DISPLAY COMPLETE!")
    print(f"   Successfully opened: {opened_count}/{len(learning_curves)} files")
    
    if opened_count > 0:
        print(f"\n💡 TIP: The learning curves should now be open in your default image viewer.")
        print(f"   You can analyze the training progression and overfitting patterns.")
    
    # Summary of what each visualization shows
    print(f"\n📋 LEARNING CURVE GUIDE:")
    print(f"   1. Main Curves: Training/test loss & accuracy over 300 epochs")
    print(f"      - Shows overfitting progression")
    print(f"      - Identifies best performance at Epoch 15")
    print(f"      - Includes Top-5 accuracy trends")
    print(f"   2. Comparison: Full dataset (33K) vs subset (1.3K) performance")
    print(f"      - Statistical significance analysis")
    print(f"      - Confidence interval comparison")
    
    print(f"\n🎯 KEY INSIGHTS FROM CURVES:")
    print(f"   ✅ Best test accuracy: 17.65% at Epoch 15")
    print(f"   ⚠️ Overfitting onset: Around Epoch 20")
    print(f"   📊 Train-test gap: 76.2% at final epoch")
    print(f"   🎯 Improvement over random: +76.5% relative")
    print(f"   📈 Top-5 accuracy: 55-60% (good feature learning)")
    
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    print(f"   🔧 Early stopping: Stop training at Epoch 15-20")
    print(f"   🛡️ Regularization: Increase dropout to 0.6-0.8")
    print(f"   📉 Learning rate: Use scheduling (reduce after plateau)")
    print(f"   🔄 Data augmentation: EEG-specific techniques")
    print(f"   🏗️ Architecture: Simplify model complexity")

if __name__ == "__main__":
    main()
