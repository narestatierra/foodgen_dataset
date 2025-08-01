#!/usr/bin/env python3
"""
Verbose Food Classification Pipeline
===================================

Enhanced pipeline with detailed output control and real-time progress monitoring.

Features:
- Real-time output from all sub-processes
- Optional verbose modes for debugging
- Progress indicators and status updates
- Flexible output redirection options

Usage:
    # Show all output in real-time
    python verbose_food_pipeline.py --categories 3 --images 10 --epochs 15 --verbose

    # Save output to files while showing progress
    python verbose_food_pipeline.py --categories 5 --images 15 --epochs 20 --save-logs

    # Silent mode with progress bars only
    python verbose_food_pipeline.py --categories 4 --images 12 --epochs 18 --quiet
"""

import os
import sys
import json
import argparse
import subprocess
import random
import time
from pathlib import Path
from typing import List, Dict


class VerboseFoodPipeline:
    """Enhanced pipeline with detailed output control."""
    
    def __init__(self, 
                 num_categories: int = 5,
                 images_per_category: int = 15,
                 epochs: int = 15,
                 batch_size: int = 8,
                 verbose: bool = False,
                 save_logs: bool = False,
                 quiet: bool = False,
                 use_existing_dataset: str = None,
                 architecture: str = "mobilenet_v2"):
        
        self.num_categories = num_categories
        self.images_per_category = images_per_category
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.save_logs = save_logs
        self.quiet = quiet
        self.architecture = architecture
        
        # Paths
        self.project_root = Path(__file__).parent
        self.food_labels_file = self.project_root / "assets" / "models" / "food_labels.txt"
        self.collection_script = self.project_root / "scripts" / "collect_food_data.py"
        self.training_script = self.project_root / "train_simple_model.py"
        
        # Dataset configuration
        if use_existing_dataset:
            # Use existing dataset
            self.dataset_dir = self.project_root / use_existing_dataset
            self.dataset_name = use_existing_dataset
            if not self.dataset_dir.exists():
                raise ValueError(f"Dataset directory '{use_existing_dataset}' not found!")
        else:
            # Create new dataset
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.dataset_name = f"verbose_exp_{num_categories}cat_{images_per_category}img_{timestamp}"
            self.dataset_dir = self.project_root / self.dataset_name
        
        self.model_dir = self.project_root / f"{self.dataset_name}_models_{self.architecture}"
        self.logs_dir = self.project_root / "logs" if save_logs else None
        
        if self.logs_dir:
            self.logs_dir.mkdir(exist_ok=True)
        
        if not self.quiet:
            self.print_header()
    
    def print_header(self):
        """Print experiment configuration."""
        print(f"🔍 Verbose Food Classification Pipeline")
        print(f"=" * 60)
        print(f"📊 Configuration:")
        print(f"   Architecture: {self.architecture}")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Training epochs: {self.epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Verbose mode: {self.verbose}")
        print(f"   Save logs: {self.save_logs}")
        print(f"   Quiet mode: {self.quiet}")
        print(f"   Model output: {self.model_dir}")
        print(f"=" * 60)
    
    def load_food_labels(self) -> List[str]:
        """Load food categories from food_labels.txt file."""
        
        if not self.quiet:
            print(f"📋 Loading food categories...")
        
        if not self.food_labels_file.exists():
            if not self.quiet:
                print(f"❌ Food labels file not found: {self.food_labels_file}")
            return ["apple", "banana", "pizza", "bread", "chicken"]
        
        labels = []
        with open(self.food_labels_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    clean_label = line.replace('_', ' ').lower()
                    labels.append(clean_label)
        
        if not self.quiet:
            print(f"✅ Loaded {len(labels)} food categories")
        
        return labels
    
    def select_random_categories(self, all_labels: List[str]) -> List[str]:
        """Randomly select categories for the experiment."""
        
        if not self.quiet:
            print(f"🎲 Selecting {self.num_categories} random categories from {len(all_labels)} available...")
        
        # Prefer single-word categories for better search results
        single_word_labels = [label for label in all_labels if ' ' not in label]
        
        if len(single_word_labels) >= self.num_categories:
            selected = random.sample(single_word_labels, self.num_categories)
        else:
            selected = random.sample(all_labels, min(self.num_categories, len(all_labels)))
        
        if not self.quiet:
            print(f"✅ Selected categories for experiment:")
            for i, category in enumerate(selected, 1):
                print(f"   {i}. {category}")
            
            print(f"📋 Target: {self.images_per_category} images per category")
            print(f"🎯 Total target images: {len(selected) * self.images_per_category}")
        
        return selected
    
    def run_subprocess_with_output(self, cmd: List[str], step_name: str, log_file: str = None) -> bool:
        """Run subprocess with controlled output handling."""
        
        if not self.quiet:
            print(f"\\n🚀 {step_name}")
            print(f"Command: {' '.join(cmd)}")
            
            if self.verbose:
                print(f"📊 Real-time output:")
                print(f"{'='*60}")
            elif self.save_logs and log_file:
                print(f"📝 Saving output to: {log_file}")
                print(f"⏳ Running... (check log file for progress)")
            else:
                print(f"⏳ Running... (use --verbose to see real-time output)")
        
        # Determine output handling
        if self.verbose:
            # Show all output in real-time
            stdout = None
            stderr = None
        elif self.save_logs and log_file:
            # Save to log file
            log_path = self.logs_dir / log_file
            stdout = open(log_path, 'w')
            stderr = subprocess.STDOUT
        else:
            # Capture but don't show
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE
        
        try:
            # Run the process
            if self.verbose:
                process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
                
                # Show progress dots if not verbose
                if not self.verbose and not self.quiet:
                    while process.poll() is None:
                        print(".", end="", flush=True)
                        time.sleep(2)
                    print()  # New line after dots
                
                result = process.wait()
            else:
                result = subprocess.run(cmd, stdout=stdout, stderr=stderr).returncode
            
            if self.save_logs and log_file and stdout != subprocess.PIPE:
                stdout.close()
            
            if self.verbose and not self.quiet:
                print(f"{'='*60}")
            
            if result == 0:
                if not self.quiet:
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
        """Collect images for the selected categories."""
        
        categories_str = ','.join(categories)
        cmd = [
            sys.executable,
            str(self.collection_script),
            "--categories", categories_str,
            "--images_per_category", str(self.images_per_category),
            "--output_dir", str(self.dataset_dir),
            "--parallel",
            "--max_workers", "4",
            "--train_split", "0.6",
            "--val_split", "0.2"
        ]
        
        return self.run_subprocess_with_output(
            cmd, 
            "Dataset Collection", 
            f"collection_{self.dataset_name}.log"
        )
    
    def show_download_statistics_verbose(self, categories: List[str]) -> None:
        """Show detailed download statistics after collection."""
        
        if self.quiet:
            return
            
        print(f"\\n📊 DOWNLOAD STATISTICS")
        print(f"=" * 50)
        
        # Check if dataset stats file exists
        stats_file = self.dataset_dir / "dataset_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            print(f"📈 Overall Statistics:")
            print(f"   Total categories: {stats.get('total_categories', 0)}")
            print(f"   Total images collected: {stats.get('total_images', 0)}")
            print(f"   Train images: {stats.get('train_images', 0)}")
            print(f"   Validation images: {stats.get('val_images', 0)}")
            print(f"   Test images: {stats.get('test_images', 0)}")
            print(f"   Raw images: {stats.get('raw_images', 0)}")
            
            if 'categories' in stats:
                print(f"\\n📂 Per-Category Breakdown:")
                for category, cat_stats in stats['categories'].items():
                    train_count = cat_stats.get('train', 0)
                    val_count = cat_stats.get('val', 0)
                    test_count = cat_stats.get('test', 0)
                    raw_count = cat_stats.get('raw', 0)
                    total_processed = train_count + val_count + test_count
                    
                    success_rate = (total_processed / self.images_per_category) * 100 if self.images_per_category > 0 else 0
                    
                    print(f"   {category}:")
                    print(f"     🎯 Target: {self.images_per_category} images")
                    print(f"     📥 Raw downloaded: {raw_count}")
                    print(f"     ✅ Successfully processed: {total_processed}")
                    print(f"     📊 Success rate: {success_rate:.1f}%")
                    print(f"     📁 Train: {train_count} | Val: {val_count} | Test: {test_count}")
        else:
            # Fallback: count files manually
            print(f"📁 Manual file count (stats file not found):")
            total_images = 0
            
            for category in categories:
                category_total = 0
                
                # Count in each split
                for split in ['train', 'val', 'test']:
                    split_dir = self.dataset_dir / split / category
                    if split_dir.exists():
                        images = len([f for f in split_dir.iterdir() 
                                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']])
                        category_total += images
                        print(f"   {category} ({split}): {images} images")
                
                total_images += category_total
                success_rate = (category_total / self.images_per_category) * 100 if self.images_per_category > 0 else 0
                print(f"   {category} total: {category_total} images ({success_rate:.1f}% success)")
            
            print(f"\\n📊 Total images across all categories: {total_images}")
        
        print(f"=" * 50)
    
    def train_model(self) -> bool:
        """Train the model on the dataset using the specified architecture."""
        
        cmd = [
            sys.executable,
            str(self.training_script),
            "--architecture", self.architecture,
            "--dataset_dir", str(self.dataset_dir),
            "--model_dir", str(self.model_dir),
            "--epochs", str(self.epochs),
            "--batch_size", str(self.batch_size)
        ]
        
        return self.run_subprocess_with_output(
            cmd, 
            f"Model Training ({self.architecture})", 
            f"training_{self.dataset_name}_{self.architecture}.log"
        )
    
    def verify_and_report(self) -> Dict:
        """Verify dataset and analyze results."""
        
        if not self.quiet:
            print(f"\\n📊 Analyzing Results...")
        
        results = {
            "experiment_name": self.dataset_name,
            "categories": [],
            "total_images": 0,
            "training_epochs": self.epochs,
            "final_accuracy": None,
            "best_val_accuracy": None,
            "model_size_mb": None,
            "tflite_size_mb": None
        }
        
        # Check dataset
        if self.dataset_dir.exists():
            label_file = self.dataset_dir / "label_mapping.json"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    labels = json.load(f)
                results["categories"] = list(labels.keys())
                
                if not self.quiet:
                    print(f"📂 Dataset categories: {', '.join(results['categories'])}")
        
        # Check model files
        if self.model_dir.exists():
            keras_model = self.model_dir / "food_classifier.keras"
            tflite_model = self.model_dir / "food_classifier.tflite"
            history_file = self.model_dir / "training_history.json"
            
            if keras_model.exists():
                size_mb = keras_model.stat().st_size / (1024 * 1024)
                results["model_size_mb"] = round(size_mb, 2)
                if not self.quiet:
                    print(f"📦 Keras model: {size_mb:.2f} MB")
            
            if tflite_model.exists():
                size_mb = tflite_model.stat().st_size / (1024 * 1024)
                results["tflite_size_mb"] = round(size_mb, 2)
                if not self.quiet:
                    print(f"📱 TFLite model: {size_mb:.2f} MB")
            
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                if "val_accuracy" in history and history["val_accuracy"]:
                    best_val_acc = max(history["val_accuracy"])
                    results["best_val_accuracy"] = round(best_val_acc, 4)
                    
                if "accuracy" in history and history["accuracy"]:
                    final_acc = history["accuracy"][-1]
                    results["final_accuracy"] = round(final_acc, 4)
                
                if not self.quiet:
                    if results["best_val_accuracy"]:
                        print(f"🎯 Best validation accuracy: {results['best_val_accuracy']:.1%}")
                    if results["final_accuracy"]:
                        print(f"🏋️  Final training accuracy: {results['final_accuracy']:.1%}")
        
        # Save detailed report
        report_file = self.project_root / f"{self.dataset_name}_detailed_report.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if not self.quiet:
            print(f"📄 Detailed report saved: {report_file}")
        
        return results
    
    def run_pipeline(self, skip_collection: bool = False) -> bool:
        """Run the complete pipeline."""
        
        if not skip_collection:
            # Load and select categories
            all_labels = self.load_food_labels()
            if not all_labels:
                return False
            
            selected_categories = self.select_random_categories(all_labels)
            
            # Run collection
            if not self.collect_dataset(selected_categories):
                return False
            
            # Show download statistics
            self.show_download_statistics_verbose(selected_categories)
        else:
            if not self.quiet:
                print(f"📁 Using existing dataset: {self.dataset_dir}")
                # Show existing dataset info
                if (self.dataset_dir / "dataset_stats.json").exists():
                    with open(self.dataset_dir / "dataset_stats.json", 'r') as f:
                        stats = json.load(f)
                    print(f"📊 Dataset contains {stats.get('total_categories', 0)} categories")
                    print(f"📊 Total images: {stats.get('total_images', 0)}")
        
        # Run training
        if not self.train_model():
            return False
        
        # Analyze results
        results = self.verify_and_report()
        
        if not self.quiet:
            print(f"\\n🎉 Pipeline completed successfully!")
            print(f"📊 Final Summary:")
            print(f"   Architecture: {self.architecture}")
            print(f"   Categories: {len(results['categories'])}")
            print(f"   Best accuracy: {results.get('best_val_accuracy', 'N/A')}")
            if results.get('model_size_mb'):
                print(f"   Model size: {results['model_size_mb']} MB")
        
        return True


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Verbose Food Classification Pipeline')
    parser.add_argument('--categories', type=int, default=5,
                      help='Number of food categories (default: 5)')
    parser.add_argument('--images', type=int, default=15,
                      help='Images per category (default: 15)')
    parser.add_argument('--epochs', type=int, default=15,
                      help='Training epochs (default: 15)')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Training batch size (default: 8)')
    parser.add_argument('--architecture', type=str, default='mobilenet_v2',
                      choices=['mobilenet_v2', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
                              'efficientnet_b3', 'efficientnet_v2_b0', 'efficientnet_v2_b1', 'efficientnet_v2_s',
                              'resnet50', 'resnet101', 'densenet121', 'xception', 'inception_v3'],
                      help='Model architecture to use (default: mobilenet_v2)')
    parser.add_argument('--use_existing_dataset', type=str, default=None,
                      help='Use existing dataset directory instead of creating new one')
    parser.add_argument('--skip_collection', action='store_true',
                      help='Skip data collection and use existing dataset')
    parser.add_argument('--verbose', action='store_true',
                      help='Show real-time output from all processes')
    parser.add_argument('--save-logs', action='store_true',
                      help='Save all output to log files')
    parser.add_argument('--quiet', action='store_true',
                      help='Minimal output, just results')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducible category selection')
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed:
        random.seed(args.seed)
        if not args.quiet:
            print(f"🌱 Using random seed: {args.seed}")
    
    # Create and run pipeline
    pipeline = VerboseFoodPipeline(
        num_categories=args.categories,
        images_per_category=args.images,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        save_logs=args.save_logs,
        quiet=args.quiet,
        use_existing_dataset=args.use_existing_dataset,
        architecture=args.architecture
    )
    
    success = pipeline.run_pipeline(skip_collection=args.skip_collection)
    
    if success:
        if not args.quiet:
            print(f"\\n✅ Experiment completed successfully!")
        sys.exit(0)
    else:
        if not args.quiet:
            print(f"\\n❌ Experiment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
