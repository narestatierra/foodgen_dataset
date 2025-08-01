#!/usr/bin/env python3
"""
Model Architecture Comparison Script for FoodGenius
===================================================

This script demonstrates how to use the enhanced train_simple_model.py
to compare different architectures for food classification.

Usage Examples:
    # List all available architectures
    python train_simple_model.py --list_architectures
    
    # Train with default MobileNetV2
    Example (single architecture):
    python scripts/train_simple_model.py --dataset_dir ml_experiments/datasets/experiment_dataset --epochs 10

Example (specific architecture):
    python scripts/train_simple_model.py --architecture efficientnet_b1 --dataset_dir ml_experiments/datasets/experiment_dataset --epochs 15
    
    # Train with EfficientNet-B1 (recommended for food images)
    python train_simple_model.py --architecture efficientnet_b1 --dataset_dir experiment_dataset --epochs 15
    
    # Compare multiple architectures (run sequentially)
    python compare_architectures.py
"""

import subprocess
import time
import json
import argparse
import sys
import os
from pathlib import Path

# Ensure we can run from either root directory or scripts directory
if Path.cwd().name == 'scripts':
    os.chdir('..')  # Change to parent directory (project root)

def run_training(architecture, dataset_dir, epochs=10, batch_size=16):
    """Run training for a specific architecture."""
    
    print(f"\n🏗️ Testing {architecture.upper()}")
    print("="*50)
    print(f"🧠 Architecture: {architecture}")
    print(f"⚙️  Epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"📊 Dataset: {dataset_dir}")
    print("="*50)
    
    start_time = time.time()
    
    # Run the training script with real-time output
        command = [
            sys.executable, "scripts/train_simple_model.py",
            "--dataset_dir", dataset_dir,
            "--architecture", architecture,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size)
        ]    try:
        # Stream output in real-time instead of capturing it
        result = subprocess.run(cmd, check=True)
        
        training_time = time.time() - start_time
        
        # For real-time output, we can't easily extract accuracy from captured text
        # We'll need to rely on the training script to save results or estimate
        print(f"\n✅ {architecture.upper()} completed successfully!")
        print(f"⏱️  Training Time: {training_time/60:.1f} minutes")
        
        return {
            'architecture': architecture,
            'test_accuracy': None,  # Will be None with real-time output
            'training_time': training_time,
            'status': 'success'
        }
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {architecture.upper()} Failed:")
        print(f"   Error: {e}")
        print(f"   Time before failure: {(time.time() - start_time)/60:.1f} minutes")
        return {
            'architecture': architecture,
            'test_accuracy': None,
            'training_time': time.time() - start_time,
            'status': 'failed',
            'error': str(e)
        }

def compare_architectures(architectures=None, epochs=5, batch_size=16):
    """Compare different architectures for food classification."""
    
    # Configuration - Updated to use 300 category dataset
    # First check for datasets in the new ml_experiments/datasets folder
    dataset_dir = "ml_experiments/datasets/foodgenius_300cat_10img_20250730_230802"
    if not Path(dataset_dir).exists():
        # Fallback to old location
        dataset_dir = "foodgenius_300cat_10img_20250730_230802"  # 300-category dataset
    
    # Default mobile-friendly architectures if none provided
    if architectures is None:
        architectures_to_test = [
            'mobilenet_v2',
            'efficientnet_v2_b0'
        ]
    else:
        architectures_to_test = architectures
    
    # Check if dataset exists
    if not Path(dataset_dir).exists():
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        print("Please run the data collection script first or check ml_experiments/datasets/ folder.")
        print("Available options:")
        print("  1. Run: python scripts/food_pipeline.py --categories 10 --images 15")
        print("  2. Move existing dataset to ml_experiments/datasets/ folder")
        return
    
    print("🍎 FoodGenius Architecture Comparison - 300 Categories")
    print("="*72)
    print(f"🎯 Architectures to test: {len(architectures_to_test)}")
    for i, arch in enumerate(architectures_to_test, 1):
        print(f"  {i}. {arch}")
    print(f"📊 Dataset: {dataset_dir}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"📊 Categories: 300 (Large-scale comparison)")
    print("="*72)
    
    results = []
    
    for i, arch in enumerate(architectures_to_test, 1):
        # Use provided batch size or adjust based on architecture complexity
        current_batch_size = batch_size
        if arch in ['densenet121'] and batch_size > 8:
            current_batch_size = 8  # Smaller batch for memory-intensive models
        elif arch in ['efficientnet_v2_b2', 'efficientnet_v2_b3'] and batch_size > 12:
            current_batch_size = 12  # Medium batch size for these models
        
        print(f"\n🏗️ [{i}/{len(architectures_to_test)}] Starting {arch.upper()}")
        result = run_training(arch, dataset_dir, epochs, current_batch_size)
        results.append(result)
        
        # Show progress after each architecture
        if result['status'] == 'success':
            print(f"\n🎉 {arch.upper()} TRAINING COMPLETED!")
            print(f"   ⏱️  Training Time: {result['training_time']/60:.1f} minutes")
            print(f"   � Check terminal output above for accuracy details")
        else:
            print(f"\n💥 {arch.upper()} TRAINING FAILED!")
            print(f"   ⏱️  Time before failure: {result['training_time']/60:.1f} minutes")
        
        # Brief pause between trainings to let system cool down
        time.sleep(3)
    
    # Print comparison results
    print(f"\n{'='*72}")
    print("🏆 ARCHITECTURE COMPARISON SUMMARY")
    print(f"{'='*72}")
    
    # Since we're using real-time output, we don't capture accuracy automatically
    # Show timing comparison instead
    successful_results = [r for r in results if r['status'] == 'success']
    
    print(f"{'Architecture':<20} | {'Time (min)':<12} | {'Status'}")
    print("-" * 50)
    
    for result in results:
        time_str = f"{result['training_time']/60:.1f}"
        status = result['status']
        
        print(f"{result['architecture']:<20} | {time_str:<12} | {status}")
    
    if successful_results:
        # Sort by training time (fastest first)
        successful_results.sort(key=lambda x: x['training_time'])
        fastest = successful_results[0]
        
        print(f"\n⚡ Fastest training: {fastest['architecture'].upper()}")
        print(f"⏱️  Training time: {fastest['training_time']/60:.1f} minutes")
        
        # Mobile-friendly recommendations
        mobile_friendly = ['mobilenet_v2', 'efficientnet_v2_b0', 'efficientnet_v2_b1']
        mobile_results = [r for r in successful_results if r['architecture'] in mobile_friendly]
        
        if mobile_results:
            fastest_mobile = min(mobile_results, key=lambda x: x['training_time'])
            print(f"\n📱 Fastest Mobile Architecture: {fastest_mobile['architecture'].upper()}")
            print(f"   Training Time: {fastest_mobile['training_time']/60:.1f} minutes")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        print(f"⚡ For speed: {fastest['architecture'].upper()} was fastest to train")
        print(f"📱 For mobile: Check the real-time output above for accuracy comparisons")
        print(f"🎯 Review the training logs above to compare final accuracies")
    else:
        print(f"\n❌ No architectures completed successfully")
    
    print(f"\n📊 Note: With real-time output mode, check the training logs above")
    print(f"    for detailed accuracy metrics and validation results.")
    
    # Save results to JSON
    results_path = Path("model_comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'comparison_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': dataset_dir,
            'epochs': epochs,
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_path}")

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Compare different ML architectures for food classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default comparison (MobileNetV2 vs EfficientNetV2B0, 5 epochs, batch size 16)
  python compare_architectures.py
  
  # Custom architectures with 10 epochs
  python compare_architectures.py --architectures mobilenet_v2 efficientnet_v2_b1 efficientnet_v2_b2 --epochs 10
  
  # Quick test with smaller batch size
  python compare_architectures.py --batch-size 8 --epochs 3
  
  # Compare many architectures
  python compare_architectures.py --architectures mobilenet_v2 efficientnet_v2_b0 efficientnet_v2_b1 resnet50 --epochs 15 --batch-size 12

Available architectures:
  mobilenet_v2, efficientnet_v2_b0, efficientnet_v2_b1, efficientnet_v2_b2, efficientnet_v2_b3,
  efficientnet_v2_s, resnet50, resnet101, densenet121, xception, inception_v3
        """
    )
    
    parser.add_argument(
        '--architectures', 
        nargs='+', 
        default=['mobilenet_v2', 'efficientnet_v2_b0'],
        help='List of architectures to compare (default: mobilenet_v2 efficientnet_v2_b0)'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=5,
        help='Number of training epochs (default: 5)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=16,
        help='Batch size for training (default: 16, automatically reduced for memory-intensive models)'
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Starting architecture comparison with:")
    print(f"   📦 Batch size: {args.batch_size}")
    print(f"   🔄 Epochs: {args.epochs}")
    print(f"   🧠 Architectures: {', '.join(args.architectures)}")
    print()
    
    compare_architectures(
        architectures=args.architectures,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
