#!/usr/bin/env python3
"""
Display MindBigData Visualization Results
========================================

Script to open and display all generated visualization plots.
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
    """Main function to display all visualizations"""
    print("🎨 MINDBIGDATA VISUALIZATION DISPLAY")
    print("=" * 50)
    
    # List of visualization files
    visualizations = [
        {
            'file': 'mindbigdata_overview.png',
            'description': '📊 Dataset Overview & Statistics'
        },
        {
            'file': 'mindbigdata_sample_signals.png', 
            'description': '🧠 Sample EEG Signals by Digit'
        },
        {
            'file': 'mindbigdata_spectral_analysis.png',
            'description': '🌊 Spectral Analysis & Frequency Bands'
        },
        {
            'file': 'mindbigdata_statistical_analysis.png',
            'description': '📊 Statistical Analysis & PCA'
        }
    ]
    
    print(f"\n📁 Available visualizations:")
    for i, viz in enumerate(visualizations, 1):
        file_path = viz['file']
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"   {i}. {viz['description']}")
            print(f"      File: {file_path} ({file_size:.1f} KB)")
        else:
            print(f"   {i}. {viz['description']} - ❌ NOT FOUND")
    
    print(f"\n🖼️ Opening visualizations...")
    
    opened_count = 0
    for viz in visualizations:
        file_path = viz['file']
        print(f"\n📂 {viz['description']}")
        if open_image(file_path):
            opened_count += 1
    
    print(f"\n🎉 VISUALIZATION DISPLAY COMPLETE!")
    print(f"   Successfully opened: {opened_count}/{len(visualizations)} files")
    
    if opened_count > 0:
        print(f"\n💡 TIP: The images should now be open in your default image viewer.")
        print(f"   You can navigate between them to see different aspects of the data.")
    
    # Summary of what each visualization shows
    print(f"\n📋 VISUALIZATION GUIDE:")
    print(f"   1. Overview: Dataset size, class distribution, signal statistics")
    print(f"   2. Sample Signals: Raw EEG waveforms for each digit class")
    print(f"   3. Spectral Analysis: Frequency content and power spectral density")
    print(f"   4. Statistical Analysis: Correlations, PCA, and quality metrics")

if __name__ == "__main__":
    main()
