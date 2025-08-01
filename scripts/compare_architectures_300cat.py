#!/usr/bin/env python3
"""
🏗️ FoodGenius - Architecture Comparison on 300-Category Dataset
Systematically compare multiple architectures on the large-scale food dataset
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path
from datetime import datetime

# Ensure we can run from either root directory or scripts directory
if Path.cwd().name == 'scripts':
    os.chdir('..')  # Change to parent directory (project root)

# Suppress TensorFlow warnings and PyDataset warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Import our training class
sys.path.append('scripts')  # Add scripts to path for imports
from train_simple_model import SimpleFoodTrainer

def print_header():
    """Print formatted header."""
    print("🏗️" + "="*70)
    print("🍎 FoodGenius - Multi-Architecture Comparison")
    print("📊 Dataset: 300-Category Food Classification")
    print("="*72)

def compare_architectures():
    """Compare multiple architectures on the 300-category dataset."""
    
    # Dataset path
    dataset_path = "foodgenius_300cat_10img_20250730_230802"
    
    # Architectures to test (excluding known problematic ones)
    architectures = [
        'mobilenet_v2',      # ✅ Verified working
        'resnet50',          # Should work - stable architecture  
        'densenet121',       # Should work - stable architecture
        'xception',          # Should work - stable architecture
        'inception_v3',      # Should work - stable architecture
        # 'efficientnet_b0',   # ❌ Skip - corrupted weights
    ]
    
    # Training configuration
    config = {
        'epochs': 10,  # 5 epochs frozen + 5 epochs fine-tuning
        'batch_size': 8,  # Conservative for memory
        'patience': 3,    # Early stopping patience
    }
    
    print_header()
    print(f"🎯 Architectures to test: {len(architectures)}")
    for i, arch in enumerate(architectures, 1):
        print(f"  {i}. {arch}")
    print(f"📊 Dataset: {dataset_path}")
    print(f"⚙️  Configuration: {config['epochs']} epochs, batch size {config['batch_size']}")
    print("="*72)
    
    # Results storage
    results = {
        'dataset': dataset_path,
        'config': config,
        'start_time': datetime.now().isoformat(),
        'architectures': {}
    }
    
    # Test each architecture
    for i, architecture in enumerate(architectures, 1):
        print(f"\n🏗️ [{i}/{len(architectures)}] Testing {architecture.upper()}")
        print("="*50)
        
        start_time = time.time()
        
        try:
            # Initialize trainer
            trainer = SimpleFoodTrainer(
                dataset_dir=dataset_path,
                architecture=architecture
            )
            
            # Train model (it creates data generators internally)
            model, history, test_accuracy = trainer.train_model(
                epochs=config['epochs'],
                batch_size=config['batch_size']
            )
            
            training_time = time.time() - start_time
            
            # Extract best metrics
            best_val_acc = max(history.history['val_accuracy'])
            final_train_acc = history.history['accuracy'][-1]
            
            # Model size info
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            
            # Store results
            arch_results = {
                'test_accuracy': float(test_accuracy),
                'best_val_accuracy': float(best_val_acc),
                'final_train_accuracy': float(final_train_acc),
                'training_time_minutes': float(training_time / 60),
                'total_params': int(total_params),
                'trainable_params': int(trainable_params),
                'mobile_friendly': trainer.supported_architectures[architecture]['mobile_friendly'],
                'status': 'success'
            }
            
            results['architectures'][architecture] = arch_results
            
            print(f"✅ {architecture.upper()} Results:")
            print(f"   📊 Test Accuracy: {test_accuracy:.1%}")
            print(f"   🎯 Best Val Accuracy: {best_val_acc:.1%}")
            print(f"   ⏱️  Training Time: {training_time/60:.1f} minutes")
            print(f"   📱 Mobile Friendly: {'Yes' if arch_results['mobile_friendly'] else 'No'}")
            print(f"   🧠 Parameters: {total_params:,} total, {trainable_params:,} trainable")
            
        except Exception as e:
            error_msg = str(e)
            training_time = time.time() - start_time
            
            results['architectures'][architecture] = {
                'status': 'failed',
                'error': error_msg,
                'training_time_minutes': float(training_time / 60),
            }
            
            print(f"❌ {architecture.upper()} Failed:")
            print(f"   Error: {error_msg}")
            print(f"   Time before failure: {training_time/60:.1f} minutes")
            
            # Continue with next architecture
            continue
    
    # Calculate completion time
    results['end_time'] = datetime.now().isoformat()
    total_time = time.time() - time.mktime(datetime.fromisoformat(results['start_time']).timetuple())
    results['total_time_minutes'] = float(total_time / 60)
    
    # Save results
    results_file = f"architecture_comparison_300cat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*72)
    print("🏆 ARCHITECTURE COMPARISON SUMMARY")
    print("="*72)
    
    successful_results = [(arch, data) for arch, data in results['architectures'].items() 
                         if data['status'] == 'success']
    
    if successful_results:
        # Sort by test accuracy
        successful_results.sort(key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        print(f"📊 Successful architectures: {len(successful_results)}")
        print("\n🥇 Rankings by Test Accuracy:")
        for i, (arch, data) in enumerate(successful_results, 1):
            mobile_icon = "📱" if data['mobile_friendly'] else "🖥️ "
            print(f"  {i}. {mobile_icon} {arch.upper()}: {data['test_accuracy']:.1%} "
                  f"({data['training_time_minutes']:.1f}min, {data['total_params']:,} params)")
        
        # Mobile-friendly recommendations
        mobile_results = [(arch, data) for arch, data in successful_results 
                         if data['mobile_friendly']]
        
        if mobile_results:
            best_mobile = mobile_results[0]
            print(f"\n📱 Best Mobile Architecture: {best_mobile[0].upper()}")
            print(f"   Accuracy: {best_mobile[1]['test_accuracy']:.1%}")
            print(f"   Parameters: {best_mobile[1]['total_params']:,}")
            print(f"   Training Time: {best_mobile[1]['training_time_minutes']:.1f} minutes")
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        print(f"⏱️  Total comparison time: {results['total_time_minutes']:.1f} minutes")
    
    else:
        print("❌ No architectures completed successfully!")
    
    # Failed architectures
    failed_results = [(arch, data) for arch, data in results['architectures'].items() 
                     if data['status'] == 'failed']
    
    if failed_results:
        print(f"\n❌ Failed architectures: {len(failed_results)}")
        for arch, data in failed_results:
            print(f"  • {arch.upper()}: {data['error']}")
    
    print("\n" + "="*72)
    return results

def main():
    """Main execution function."""
    if not os.path.exists("foodgenius_300cat_10img_20250730_230802"):
        print("❌ Dataset not found: foodgenius_300cat_10img_20250730_230802")
        print("   Please ensure the dataset directory exists in the current path.")
        sys.exit(1)
    
    # Run comparison
    results = compare_architectures()
    
    # Exit with appropriate code
    successful_count = sum(1 for data in results['architectures'].values() 
                          if data['status'] == 'success')
    
    if successful_count > 0:
        print(f"🎉 Comparison complete! {successful_count} architectures succeeded.")
        sys.exit(0)
    else:
        print("💥 All architectures failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
