#!/usr/bin/env python3
"""
Dependency Installation Script
=============================

This script installs all required dependencies for the EEG-to-Image reconstruction project.

Usage: python install_dependencies.py
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Install all dependencies"""
    
    print("🔧 Installing Dependencies for EEG-to-Image Reconstruction")
    print("=" * 60)
    
    # List of packages to install
    packages = [
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "torchaudio>=0.9.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "Pillow>=8.3.0",
        "tqdm>=4.62.0",
        "opencv-python>=4.5.0"
    ]
    
    # Install CLIP separately (from git)
    clip_package = "git+https://github.com/openai/CLIP.git"
    
    print("Installing standard packages...")
    failed_packages = []
    
    for package in packages:
        print(f"  Installing {package}...")
        if install_package(package):
            print(f"    ✅ {package} installed successfully")
        else:
            print(f"    ❌ Failed to install {package}")
            failed_packages.append(package)
    
    print(f"\nInstalling CLIP from GitHub...")
    if install_package(clip_package):
        print(f"    ✅ CLIP installed successfully")
    else:
        print(f"    ❌ Failed to install CLIP")
        failed_packages.append("CLIP")
    
    print("\n" + "=" * 60)
    
    if failed_packages:
        print("❌ Installation completed with errors:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nPlease install failed packages manually:")
        print("  pip install <package_name>")
    else:
        print("✅ All dependencies installed successfully!")
        print("\nYou can now run:")
        print("  python run_complete_pipeline.py")
        print("  or")
        print("  python quick_start.py")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
