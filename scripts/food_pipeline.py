#!/usr/bin/env python3
"""
FoodGenius Classification Pipeline
=================================

Complete automated pipeline for food classification experiments with all features:

🚀 Key Features:
- Loads 419+ food categories from food_labels.txt
- Category-level AND image-level parallelization 
- Enhanced category selection display with numbered lists
- Real-time download progress monitoring
- Detailed download statistics with success rates
- Multiple output modes (verbose, normal, quiet, log-saving)
- MobileNetV2 transfer learning with 2-phase training
- Automatic TensorFlow Lite conversion for mobile deployment
- Comprehensive experiment reporting and analysis
- Reproducible experiments with random seed support

🎯 Usage Examples:
    # Quick 3-category test
    python food_pipeline.py --categories 3 --images 8 --epochs 10
    
    # Production experiment with all features
    python food_pipeline.py --categories 5 --images 15 --epochs 20 --verbose --seed 42
    
    # Specific categories (no random selection)
    python food_pipeline.py --specific apple,banana,pizza --images 12 --epochs 15
    
    # Large experiment with logging
    python food_pipeline.py --categories 10 --images 25 --epochs 30 --save-logs --quiet

📊 Output Control:
    --verbose: Real-time output from all processes
    --save-logs: Save detailed logs to files
    --quiet: Minimal output, just results
    --progress: Show progress bars (default)
"""

import os
import sys
import json
import argparse
import subprocess
import random
import time
from pathlib import Path
from typing import List, Dict, Optional
import threading
import shutil


class FoodGeniusPipeline:
    """Complete food classification pipeline with all advanced features."""
    
    def __init__(self, 
                 num_categories: int = 5,
                 specific_categories: Optional[List[str]] = None,
                 images_per_category: int = 15,
                 epochs: int = 15,
                 batch_size: int = 8,
                 max_workers: int = 6,
                 max_category_workers: int = 3,
                 parallel_categories: bool = False,
                 verbose: bool = False,
                 save_logs: bool = False,
                 quiet: bool = False,
                 progress: bool = True):
        
        # Core experiment parameters
        self.num_categories = num_categories
        self.specific_categories = specific_categories
        self.images_per_category = images_per_category
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_category_workers = max_category_workers
        self.parallel_categories = parallel_categories
        
        # Output control
        self.verbose = verbose
        self.save_logs = save_logs
        self.quiet = quiet
        self.progress = progress and not quiet
        
        # Setup paths
        self.project_root = Path(__file__).parent.parent  # Go up one level from scripts/
        self.food_labels_file = self.project_root / "assets" / "models" / "food_labels.txt"
        self.collection_script = self.project_root / "scripts" / "collect_food_data.py"
        self.training_script = self.project_root / "scripts" / "train_simple_model.py"
        
        # Create unique experiment identifier
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if specific_categories:
            exp_name = f"specific_{len(specific_categories)}cat_{images_per_category}img_{timestamp}"
        else:
            exp_name = f"foodgenius_{num_categories}cat_{images_per_category}img_{timestamp}"
        
        self.experiment_name = exp_name
        self.dataset_dir = self.project_root / exp_name
        self.model_dir = self.project_root / f"{exp_name}_models"
        
        # Setup logging
        if save_logs:
            self.logs_dir = self.project_root / "logs" / exp_name
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.logs_dir = None
        
        # Initialize experiment
        if not self.quiet:
            self.print_header()
    
    def print_header(self):
        """Print comprehensive experiment header."""
        print(f"🍎 FoodGenius Classification Pipeline")
        print(f"=" * 70)
        print(f"🧪 Experiment: {self.experiment_name}")
        print(f"📊 Configuration:")
        
        if self.specific_categories:
            print(f"   📋 Specific categories: {', '.join(self.specific_categories)}")
            print(f"   🔢 Category count: {len(self.specific_categories)}")
        else:
            print(f"   🎲 Random categories: {self.num_categories}")
        
        print(f"   📷 Images per category: {self.images_per_category}")
        print(f"   🎯 Total target images: {(len(self.specific_categories) if self.specific_categories else self.num_categories) * self.images_per_category}")
        print(f"   🏋️  Training epochs: {self.epochs}")
        print(f"   🔢 Batch size: {self.batch_size}")
        print(f"   🔧 Output mode: {'Verbose' if self.verbose else 'Quiet' if self.quiet else 'Normal'}")
        
        if self.save_logs:
            print(f"   📝 Logs directory: {self.logs_dir}")
        
        print(f"   📁 Dataset: {self.dataset_dir}")
        print(f"   🧠 Models: {self.model_dir}")
        print(f"=" * 70)
    
    def load_food_labels(self) -> List[str]:
        """Load comprehensive food categories from food_labels.txt."""
        
        if self.progress:
            print(f"📋 Loading food categories from {self.food_labels_file}...")
        
        if not self.food_labels_file.exists():
            if not self.quiet:
                print(f"❌ Food labels file not found: {self.food_labels_file}")
                print(f"🔄 Using fallback categories...")
            return ["apple", "banana", "pizza", "bread", "chicken", "carrot", "orange", "pasta"]
        
        labels = []
        with open(self.food_labels_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Clean up label (remove underscores, normalize)
                    clean_label = line.replace('_', ' ').lower().strip()
                    if clean_label:
                        labels.append(clean_label)
        
        if self.progress:
            print(f"✅ Loaded {len(labels)} food categories from database")
        
        return labels
    
    def select_categories(self, all_labels: List[str]) -> List[str]:
        """Select categories for experiment (specific or random)."""
        
        if self.specific_categories:
            # Use user-specified categories
            if self.progress:
                print(f"👤 Using {len(self.specific_categories)} user-specified categories...")
                print(f"✅ Categories for experiment:")
                for i, category in enumerate(self.specific_categories, 1):
                    print(f"   {i}. {category}")
            
            return self.specific_categories
        
        else:
            # Random selection from database
            if self.progress:
                print(f"🎲 Selecting {self.num_categories} random categories from {len(all_labels)} available...")
            
            # Prefer single-word categories for better search results
            single_word_labels = [label for label in all_labels if ' ' not in label and len(label) > 2]
            
            if len(single_word_labels) >= self.num_categories:
                selected = random.sample(single_word_labels, self.num_categories)
            else:
                # Fall back to all labels if not enough single-word ones
                selected = random.sample(all_labels, min(self.num_categories, len(all_labels)))
            
            if self.progress:
                print(f"✅ Selected categories for experiment:")
                for i, category in enumerate(selected, 1):
                    print(f"   {i}. {category}")
            
            return selected
        
        if self.progress:
            categories = self.specific_categories or selected
            print(f"📋 Target: {self.images_per_category} images per category")
            print(f"🎯 Total target images: {len(categories) * self.images_per_category}")
    
    def run_subprocess_with_output(self, cmd: List[str], step_name: str, log_file: str = None) -> bool:
        """Run subprocess with sophisticated output handling."""
        
        if self.progress:
            print(f"\\n🚀 {step_name}")
            if self.verbose:
                print(f"Command: {' '.join(cmd)}")
        
        # Determine output handling strategy
        if self.verbose:
            # Show all output in real-time
            stdout = None
            stderr = None
            if self.progress:
                print(f"📊 Real-time output:")
                print(f"{'='*60}")
                
        elif self.save_logs and log_file:
            # Save to log file
            log_path = self.logs_dir / log_file
            stdout = open(log_path, 'w')
            stderr = subprocess.STDOUT
            if self.progress:
                print(f"📝 Saving output to: {log_path}")
                print(f"⏳ Running... (check log file for progress)")
                
        elif self.quiet:
            # Suppress all output
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
            
        else:
            # Show progress dots only
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE
            if self.progress:
                print(f"⏳ Running... (use --verbose for real-time output)")
        
        try:
            if self.verbose:
                # Real-time output
                result = subprocess.run(cmd, stdout=stdout, stderr=stderr).returncode
            else:
                # Show progress indicator for non-verbose modes
                process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
                
                if self.progress and not self.quiet:
                    # Show progress dots
                    dot_count = 0
                    while process.poll() is None:
                        print(".", end="", flush=True)
                        time.sleep(3)
                        dot_count += 1
                        if dot_count % 20 == 0:  # New line every 60 seconds
                            print()
                    
                    if dot_count > 0:
                        print()  # Final newline
                
                result = process.wait()
            
            # Close log file if opened
            if self.save_logs and log_file and stdout != subprocess.PIPE and stdout != subprocess.DEVNULL:
                stdout.close()
            
            if self.verbose and self.progress:
                print(f"{'='*60}")
            
            # Report result
            if result == 0:
                if self.progress:
                    print(f"✅ {step_name} completed successfully!")
                return True
            else:
                if not self.quiet:
                    print(f"❌ {step_name} failed with return code: {result}")
                return False
                
        except Exception as e:
            if not self.quiet:
                print(f"❌ Error during {step_name}: {e}")
            return False
    
    def collect_dataset(self, categories: List[str]) -> bool:
        """Collect images with parallel processing at category level."""
        
        if self.progress:
            print(f"\\n📥 Dataset Collection Phase")
            print(f"🎯 Categories: {', '.join(categories)}")
            print(f"📊 Target: {self.images_per_category} images per category")
            if self.parallel_categories:
                print(f"🚀 Mode: DUAL-LEVEL parallel (categories + images)")
                print(f"⚡ Category workers: {self.max_category_workers}")
                print(f"⚡ Image workers: {self.max_workers}")
            else:
                print(f"⚡ Mode: IMAGE-LEVEL parallel only")
                print(f"⚡ Image workers: {self.max_workers}")
        
        # Prepare collection command with appropriate parallelization
        categories_str = ','.join(categories)
        cmd = [
            sys.executable,
            str(self.collection_script),
            "--categories", categories_str,
            "--images_per_category", str(self.images_per_category),
            "--output_dir", str(self.dataset_dir),
            "--parallel",
            "--max_workers", str(self.max_workers),
            "--train_split", "0.6", 
            "--val_split", "0.2"
        ]
        
        # Add category-level parallelization if enabled
        if self.parallel_categories:
            cmd.extend([
                "--parallel_categories",
                "--max_category_workers", str(self.max_category_workers)
            ])
        
        success = self.run_subprocess_with_output(
            cmd, 
            "Image Collection & Dataset Organization", 
            f"collection_{self.experiment_name}.log"
        )
        
        if success:
            self.show_download_statistics(categories)
        
        return success
    
    def show_download_statistics(self, categories: List[str]) -> None:
        """Display comprehensive download statistics."""
        
        if self.quiet:
            return
            
        print(f"\\n📊 DOWNLOAD STATISTICS")
        print(f"=" * 60)
        
        # Try to read dataset stats
        stats_file = self.dataset_dir / "dataset_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            # Overall statistics
            print(f"📈 Overall Statistics:")
            print(f"   Total categories: {stats.get('total_categories', 0)}")
            print(f"   Total images collected: {stats.get('total_images', 0)}")
            print(f"   Train images: {stats.get('train_images', 0)}")
            print(f"   Validation images: {stats.get('val_images', 0)}")
            print(f"   Test images: {stats.get('test_images', 0)}")
            print(f"   Raw images downloaded: {stats.get('raw_images', 0)}")
            
            # Per-category breakdown
            if 'categories' in stats:
                print(f"\\n📂 Per-Category Performance:")
                print(f"{'Category':<15} {'Target':<8} {'Downloaded':<12} {'Processed':<10} {'Success %':<10} {'Train':<6} {'Val':<4} {'Test':<4}")
                print(f"{'-'*15} {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*6} {'-'*4} {'-'*4}")
                
                total_success = 0
                for category, cat_stats in stats['categories'].items():
                    train_count = cat_stats.get('train', 0)
                    val_count = cat_stats.get('val', 0)
                    test_count = cat_stats.get('test', 0)
                    raw_count = cat_stats.get('raw', 0)
                    total_processed = train_count + val_count + test_count
                    
                    success_rate = (total_processed / self.images_per_category) * 100 if self.images_per_category > 0 else 0
                    total_success += success_rate
                    
                    print(f"{category:<15} {self.images_per_category:<8} {raw_count:<12} {total_processed:<10} {success_rate:<9.1f}% {train_count:<6} {val_count:<4} {test_count:<4}")
                
                avg_success = total_success / len(stats['categories']) if stats['categories'] else 0
                print(f"{'-'*15} {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*6} {'-'*4} {'-'*4}")
                print(f"{'AVERAGE':<15} {'':<8} {'':<12} {'':<10} {avg_success:<9.1f}% {'':<6} {'':<4} {'':<4}")
        
        else:
            # Fallback: manual file counting
            print(f"📁 Manual file count (dataset_stats.json not found):")
            total_images = 0
            
            for category in categories:
                category_total = 0
                splits_info = []
                
                for split in ['train', 'val', 'test']:
                    split_dir = self.dataset_dir / split / category
                    if split_dir.exists():
                        images = len([f for f in split_dir.iterdir() 
                                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']])
                        category_total += images
                        splits_info.append(f"{split}: {images}")
                
                total_images += category_total
                success_rate = (category_total / self.images_per_category) * 100 if self.images_per_category > 0 else 0
                
                print(f"   {category}: {category_total} images ({success_rate:.1f}% success) - {', '.join(splits_info)}")
            
            print(f"\\n📊 Total images: {total_images}")
        
        # Check label mapping
        label_file = self.dataset_dir / "label_mapping.json"
        if label_file.exists():
            with open(label_file, 'r') as f:
                labels = json.load(f)
            print(f"\\n🏷️  Label Mapping:")
            for label, index in sorted(labels.items(), key=lambda x: x[1]):
                print(f"   {index}: {label}")
        
        print(f"=" * 60)
    
    def train_model(self) -> bool:
        """Train MobileNetV2 model with transfer learning."""
        
        if self.progress:
            print(f"\\n🏋️  Model Training Phase")
            print(f"🧠 Architecture: MobileNetV2 + Transfer Learning")
            print(f"📊 Training epochs: {self.epochs}")
            print(f"🔢 Batch size: {self.batch_size}")
        
        cmd = [
            sys.executable,
            str(self.training_script),
            "--dataset_dir", str(self.dataset_dir),
            "--model_dir", str(self.model_dir),
            "--epochs", str(self.epochs),
            "--batch_size", str(self.batch_size)
        ]
        
        return self.run_subprocess_with_output(
            cmd, 
            "MobileNetV2 Transfer Learning", 
            f"training_{self.experiment_name}.log"
        )
    
    def analyze_results(self) -> Dict:
        """Comprehensive result analysis and reporting."""
        
        if self.progress:
            print(f"\\n📊 Results Analysis")
        
        results = {
            "experiment_name": self.experiment_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "categories": self.specific_categories or self.num_categories,
                "images_per_category": self.images_per_category,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "output_mode": "verbose" if self.verbose else "quiet" if self.quiet else "normal"
            },
            "dataset": {
                "categories": [],
                "total_images": 0,
                "splits": {}
            },
            "model": {
                "final_accuracy": None,
                "best_val_accuracy": None,
                "keras_size_mb": None,
                "tflite_size_mb": None,
                "compression_ratio": None
            },
            "files": {
                "dataset_dir": str(self.dataset_dir),
                "model_dir": str(self.model_dir),
                "logs_dir": str(self.logs_dir) if self.logs_dir else None
            }
        }
        
        # Analyze dataset
        if self.dataset_dir.exists():
            label_file = self.dataset_dir / "label_mapping.json"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    labels = json.load(f)
                results["dataset"]["categories"] = list(labels.keys())
            
            stats_file = self.dataset_dir / "dataset_stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                results["dataset"]["total_images"] = stats.get("total_images", 0)
                results["dataset"]["splits"] = {
                    "train": stats.get("train_images", 0),
                    "val": stats.get("val_images", 0),
                    "test": stats.get("test_images", 0)
                }
        
        # Analyze model performance
        if self.model_dir.exists():
            # Training history
            history_file = self.model_dir / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                if "val_accuracy" in history and history["val_accuracy"]:
                    results["model"]["best_val_accuracy"] = round(max(history["val_accuracy"]), 4)
                
                if "accuracy" in history and history["accuracy"]:
                    results["model"]["final_accuracy"] = round(history["accuracy"][-1], 4)
            
            # Model file sizes
            keras_model = self.model_dir / "food_classifier.keras"
            tflite_model = self.model_dir / "food_classifier.tflite"
            
            if keras_model.exists():
                size_mb = keras_model.stat().st_size / (1024 * 1024)
                results["model"]["keras_size_mb"] = round(size_mb, 2)
            
            if tflite_model.exists():
                size_mb = tflite_model.stat().st_size / (1024 * 1024)
                results["model"]["tflite_size_mb"] = round(size_mb, 2)
                
                if results["model"]["keras_size_mb"]:
                    ratio = results["model"]["keras_size_mb"] / results["model"]["tflite_size_mb"]
                    results["model"]["compression_ratio"] = round(ratio, 1)
        
        # Print summary
        if not self.quiet:
            print(f"✅ Experiment Analysis Complete")
            print(f"📂 Categories: {len(results['dataset']['categories'])} - {', '.join(results['dataset']['categories'][:5])}{'...' if len(results['dataset']['categories']) > 5 else ''}")
            print(f"📊 Dataset: {results['dataset']['total_images']} total images")
            
            if results["model"]["best_val_accuracy"]:
                print(f"🎯 Best Validation Accuracy: {results['model']['best_val_accuracy']:.1%}")
            
            if results["model"]["keras_size_mb"]:
                print(f"📦 Model Size: {results['model']['keras_size_mb']} MB (Keras)")
            
            if results["model"]["tflite_size_mb"]:
                print(f"📱 Mobile Size: {results['model']['tflite_size_mb']} MB (TFLite)")
                
            if results["model"]["compression_ratio"]:
                print(f"🗜️  Compression: {results['model']['compression_ratio']}x smaller for mobile")
        
        return results
    
    def save_comprehensive_report(self, results: Dict) -> None:
        """Save detailed experiment report with all metrics."""
        
        report_file = self.project_root / f"{self.experiment_name}_REPORT.json"
        
        # Add system info
        results["system_info"] = {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": str(self.project_root)
        }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True)
        
        if self.progress:
            print(f"📄 Comprehensive report saved: {report_file}")
    
    def cleanup_experiment_files(self, keep_models: bool = True, keep_datasets: bool = False) -> None:
        """Optional cleanup of experiment files."""
        
        if not self.quiet:
            print(f"\\n🧹 Cleanup Options:")
            print(f"   Models: {'Keep' if keep_models else 'Remove'}")
            print(f"   Datasets: {'Keep' if keep_datasets else 'Remove'}")
        
        if not keep_datasets and self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
            if self.progress:
                print(f"🗑️  Removed dataset: {self.dataset_dir}")
        
        if not keep_models and self.model_dir.exists():
            shutil.rmtree(self.model_dir)
            if self.progress:
                print(f"🗑️  Removed models: {self.model_dir}")
    
    def run_complete_pipeline(self) -> bool:
        """Execute the complete food classification pipeline."""
        
        start_time = time.time()
        
        try:
            # Phase 1: Load and select categories
            all_labels = self.load_food_labels()
            if not all_labels:
                if not self.quiet:
                    print(f"❌ Failed to load food categories")
                return False
            
            selected_categories = self.select_categories(all_labels)
            
            # Phase 2: Collect dataset
            if not self.collect_dataset(selected_categories):
                if not self.quiet:
                    print(f"❌ Dataset collection failed")
                return False
            
            # Phase 3: Train model
            if not self.train_model():
                if not self.quiet:
                    print(f"❌ Model training failed")
                return False
            
            # Phase 4: Analyze results
            results = self.analyze_results()
            
            # Phase 5: Save comprehensive report
            self.save_comprehensive_report(results)
            
            # Phase 6: Final summary
            end_time = time.time()
            duration = end_time - start_time
            
            if not self.quiet:
                print(f"\\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
                print(f"=" * 70)
                print(f"⏱️  Total Duration: {duration/60:.1f} minutes")
                print(f"🎯 Final Accuracy: {results['model'].get('best_val_accuracy', 'N/A')}")
                print(f"📊 Categories: {len(results['dataset']['categories'])}")
                print(f"📷 Images: {results['dataset']['total_images']}")
                print(f"📁 Results: {self.model_dir}")
                print(f"📄 Report: {self.experiment_name}_REPORT.json")
                print(f"🚀 Ready for Flutter deployment!")
                print(f"=" * 70)
            
            return True
            
        except KeyboardInterrupt:
            if not self.quiet:
                print(f"\\n⚠️  Experiment interrupted by user")
            return False
        except Exception as e:
            if not self.quiet:
                print(f"\\n❌ Experiment failed: {e}")
            return False


def main():
    """Main function with comprehensive argument parsing."""
    
    parser = argparse.ArgumentParser(
        description='FoodGenius Classification Pipeline - Complete automated food classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick 3-category test
    python food_pipeline.py --categories 3 --images 8 --epochs 10
    
    # Production experiment
    python food_pipeline.py --categories 5 --images 15 --epochs 20 --verbose --seed 42
    
    # Specific categories
    python food_pipeline.py --specific apple,banana,pizza --images 12 --epochs 15
    
    # Large experiment with logging
    python food_pipeline.py --categories 10 --images 25 --epochs 30 --save-logs --quiet

For more information, see EXPERIMENT_GUIDE.md
        """
    )
    
    # Core experiment parameters
    parser.add_argument('--categories', type=int, default=5,
                      help='Number of random categories to select (default: 5)')
    parser.add_argument('--specific', type=str, default=None,
                      help='Specific categories (comma-separated, e.g., "apple,banana,pizza")')
    parser.add_argument('--images', type=int, default=15,
                      help='Images per category (default: 15)')
    parser.add_argument('--epochs', type=int, default=15,
                      help='Training epochs (default: 15)')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Training batch size (default: 8)')
    
    # Performance control
    parser.add_argument('--max-workers', type=int, default=6,
                      help='Parallel image download workers per category (default: 6)')
    parser.add_argument('--max-category-workers', type=int, default=3,
                      help='Maximum categories to process simultaneously when using --parallel-categories (default: 3)')
    parser.add_argument('--parallel-categories', action='store_true',
                      help='Enable category-level parallelization (dual-level: categories + images in parallel)')
    
    # Output control
    parser.add_argument('--verbose', action='store_true',
                      help='Show real-time output from all processes')
    parser.add_argument('--save-logs', action='store_true',
                      help='Save detailed logs to files')
    parser.add_argument('--quiet', action='store_true',
                      help='Minimal output, just results')
    parser.add_argument('--no-progress', action='store_true',
                      help='Disable progress indicators')
    
    # Experiment control
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducible results')
    parser.add_argument('--cleanup', action='store_true',
                      help='Remove dataset after completion (keep models)')
    parser.add_argument('--cleanup-all', action='store_true',
                      help='Remove both dataset and models after completion')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.quiet and args.verbose:
        print("❌ Cannot use both --quiet and --verbose")
        sys.exit(1)
    
    # Set random seed for reproducibility
    if args.seed:
        random.seed(args.seed)
        if not args.quiet:
            print(f"🌱 Using random seed: {args.seed}")
    
    # Parse specific categories
    specific_categories = None
    if args.specific:
        specific_categories = [cat.strip() for cat in args.specific.split(',')]
        if not args.quiet:
            print(f"👤 Using specific categories: {specific_categories}")
    
    # Create and run pipeline
    pipeline = FoodGeniusPipeline(
        num_categories=args.categories,
        specific_categories=specific_categories,
        images_per_category=args.images,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_category_workers=args.max_category_workers,
        parallel_categories=args.parallel_categories,
        verbose=args.verbose,
        save_logs=args.save_logs,
        quiet=args.quiet,
        progress=not args.no_progress
    )
    
    # Run the complete pipeline
    success = pipeline.run_complete_pipeline()
    
    # Optional cleanup
    if success and (args.cleanup or args.cleanup_all):
        pipeline.cleanup_experiment_files(
            keep_models=not args.cleanup_all,
            keep_datasets=False
        )
    
    # Exit with appropriate code
    if success:
        if not args.quiet:
            print(f"\\n✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        if not args.quiet:
            print(f"\\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
