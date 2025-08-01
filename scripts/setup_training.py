#!/usr/bin/env python3
"""
FoodGenius Setup and Dataset Collection
======================================

This script sets up the environment and runs the dataset collection pipeline
using USDA FoodData Central as the primary data source.

Steps:
1. Validates environment and dependencies
2. Downloads USDA FoodData Central datasets
3. Processes and organizes the data for training
4. Creates balanced train/val/test splits
5. Prepares the data for model training

Usage:
    python3 setup_training.py --target_classes 1000
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import argparse

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'requests', 'pandas', 'numpy', 'tensorflow', 
        'pillow', 'opencv-python', 'tqdm', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('opencv_python', 'cv2'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies available!")
    return True

def setup_environment():
    """Set up the training environment."""
    print("🛠️ Setting up training environment...")
    
    # Create necessary directories
    directories = [
        'food_dataset_large',
        'food_dataset_large/raw',
        'food_dataset_large/processed', 
        'food_dataset_large/train',
        'food_dataset_large/val',
        'food_dataset_large/test',
        'food_dataset_large/metadata',
        'models/checkpoints',
        'logs'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    print("✅ Environment setup complete!")

def run_usda_collection(target_classes: int = 1000):
    """Run USDA dataset collection."""
    print("📥 Starting USDA FoodData Central collection...")
    
    try:
        result = subprocess.run([
            sys.executable, 'scripts/collect_usda_dataset.py',
            '--output_dir', 'usda_food_dataset'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ USDA dataset collection successful!")
            print(result.stdout)
            return True
        else:
            print("❌ USDA dataset collection failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running USDA collection: {e}")
        return False

def run_demo_fallback(target_classes: int = 100):
    """Run demo dataset creation as fallback."""
    print("📝 Creating demo dataset as fallback...")
    
    try:
        result = subprocess.run([
            sys.executable, 'scripts/create_demo_dataset.py',
            '--output_dir', 'food_dataset_demo',
            '--classes', str(target_classes)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Demo dataset creation successful!")
            print(result.stdout)
            return True
        else:
            print("❌ Demo dataset creation failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error creating demo dataset: {e}")
        return False

def run_training_pipeline(data_dir: str = "food_dataset_demo"):
    """Run the training pipeline."""
    print("🚀 Starting training pipeline...")
    
    try:
        result = subprocess.run([
            sys.executable, 'scripts/collect_and_train.py',
            '--skip_collection',
            '--data_dir', data_dir,
            '--target_classes', '100',
            '--epochs', '10'  # Reduced for demo
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Training pipeline completed!")
            print(result.stdout)
            return True
        else:
            print("❌ Training pipeline failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running training pipeline: {e}")
        return False

def validate_flutter_integration():
    """Validate Flutter app integration."""
    print("📱 Validating Flutter app integration...")
    
    try:
        # Check if Flutter is available
        result = subprocess.run(['flutter', '--version'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Flutter available")
            
            # Check if pub get works
            result = subprocess.run(['flutter', 'pub', 'get'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Flutter dependencies resolved")
                return True
            else:
                print("❌ Flutter pub get failed")
                return False
        else:
            print("❌ Flutter not available")
            return False
            
    except Exception as e:
        print(f"❌ Error validating Flutter: {e}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup FoodGenius training pipeline")
    parser.add_argument("--target_classes", type=int, default=1000,
                       help="Target number of food classes")
    parser.add_argument("--skip_dependencies", action="store_true",
                       help="Skip dependency check")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training pipeline")
    parser.add_argument("--use_demo", action="store_true",
                       help="Use demo dataset instead of USDA")
    
    args = parser.parse_args()
    
    print("🍎 FoodGenius Training Setup")
    print("="*50)
    
    # Step 1: Check dependencies
    if not args.skip_dependencies:
        if not check_dependencies():
            print("\n❌ Setup failed: Missing dependencies")
            print("Install required packages and try again")
            return False
    
    # Step 2: Setup environment
    setup_environment()
    
    # Step 3: Collect dataset
    dataset_success = False
    data_dir = "food_dataset_demo"
    
    if args.use_demo:
        print("\n📝 Using demo dataset...")
        dataset_success = run_demo_fallback(args.target_classes)
    else:
        print("\n📥 Attempting USDA dataset collection...")
        dataset_success = run_usda_collection(args.target_classes)
        
        if dataset_success:
            data_dir = "usda_food_dataset"
        else:
            print("\n📝 Falling back to demo dataset...")
            dataset_success = run_demo_fallback(min(args.target_classes, 100))
    
    if not dataset_success:
        print("\n❌ Dataset collection failed!")
        return False
    
    # Step 4: Run training pipeline
    if not args.skip_training:
        training_success = run_training_pipeline(data_dir)
        
        if not training_success:
            print("\n❌ Training pipeline failed!")
            return False
    
    # Step 5: Validate Flutter integration
    flutter_success = validate_flutter_integration()
    
    # Final summary
    print("\n" + "="*50)
    print("🎉 SETUP SUMMARY")
    print("="*50)
    print(f"✅ Environment: Ready")
    print(f"{'✅' if dataset_success else '❌'} Dataset: {'Ready' if dataset_success else 'Failed'}")
    
    if not args.skip_training:
        print(f"{'✅' if training_success else '❌'} Training: {'Completed' if training_success else 'Failed'}")
    
    print(f"{'✅' if flutter_success else '❌'} Flutter: {'Ready' if flutter_success else 'Issues'}")
    
    if dataset_success and (args.skip_training or training_success) and flutter_success:
        print(f"\n🎯 Next Steps:")
        print(f"1. Test the app: flutter run")
        print(f"2. Deploy to device: flutter run --release")
        print(f"3. Test food recognition with real images")
        print(f"4. Scale up dataset and retrain for production")
        return True
    else:
        print(f"\n⚠️ Setup completed with issues. Check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
