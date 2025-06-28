#!/usr/bin/env python3
"""
Open Loss Plot Visualizations
=============================

Script to open the generated loss plot visualizations.
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
            print(f"Opened: {image_path}")
            return True
        else:
            print(f"File not found: {image_path}")
            return False
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return False

def main():
    """Main function to display loss plots"""
    print("ATMS 1000 Epochs Loss Visualizations")
    print("=" * 50)
    
    # Loss plot files
    loss_plots = [
        {
            'file': 'simple_train_test_loss.png',
            'description': 'Simple Train vs Test Loss Plot',
        },
        {
            'file': 'atms_1000_epochs_loss_analysis.png',
            'description': 'Comprehensive 4-Panel Analysis',
        }
    ]
    
    print(f"\nAvailable loss plot visualizations:")
    for i, plot in enumerate(loss_plots, 1):
        file_path = plot['file']
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024**2)  # MB
            print(f"   {i}. {plot['description']}")
            print(f"      File: {file_path} ({file_size:.1f} MB)")
        else:
            print(f"   {i}. {plot['description']} - NOT FOUND")
    
    print(f"\nOpening loss plot visualizations...")
    
    opened_count = 0
    for plot in loss_plots:
        file_path = plot['file']
        print(f"\n{plot['description']}")
        if open_image(file_path):
            opened_count += 1
    
    print(f"\nLoss plots display complete!")
    print(f"Successfully opened: {opened_count}/{len(loss_plots)} files")
    
    if opened_count > 0:
        print(f"\nThe loss plots should now be open in your default image viewer.")
        print(f"You can analyze the extreme overfitting patterns.")
    
    # Summary of what each visualization shows
    print(f"\nLoss Plot Guide:")
    print(f"   1. Simple Plot: Clean train vs test loss over 1000 epochs")
    print(f"      - Shows clear divergence after epoch 17")
    print(f"      - Highlights optimal stopping point")
    print(f"   2. Comprehensive: 4-panel detailed analysis")
    print(f"      - Loss curves with phases marked")
    print(f"      - Accuracy progression")
    print(f"      - Early epochs focus (1-100)")
    print(f"      - Train-test gap progression")
    
    print(f"\nKey Insights:")
    print(f"   - Best performance: 17.56% at Epoch 17")
    print(f"   - Train loss: 4.14 -> 2.28 (45% reduction)")
    print(f"   - Test loss: 4.12 -> 6.24 (51% increase)")
    print(f"   - Final train-test gap: 82.34%")
    print(f"   - Computational waste: 98.3% of epochs")
    
    print(f"\nRecommendations:")
    print(f"   - Early stopping: Essential at epoch 17-20")
    print(f"   - Regularization: Strong dropout (0.8) needed")
    print(f"   - Model complexity: Reduce by 70%")
    print(f"   - Learning rate: Implement scheduling")

if __name__ == "__main__":
    main()
