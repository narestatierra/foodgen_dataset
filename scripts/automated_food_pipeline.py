#!/usr/bin/env python3
"""
Automated Food Classification Pipeline
=====================================

This script automates the entire pipeline:
1. Loads food categories from food_labels.txt
2. Downloads specified number of images per category
3. Organizes dataset into train/val/test splits
4. Automatically trains MobileNetV2 model
5. Evaluates and reports results

Usage:
    python automated_food_pipeline.py --categories 5 --images_per_category 15 --epochs 20
"""

import os
import sys
import json
import argparse
import subprocess
import random
from pathlib import Path
from typing import List, Dict
import time

class AutomatedFoodPipeline:
    """Complete automated pipeline for food classification."""
    
    def __init__(self, 
                 num_categories: int = 5,
                 images_per_category: int = 15,
                 epochs: int = 15,
                 batch_size: int = 8):
        
        self.num_categories = num_categories
        self.images_per_category = images_per_category
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Paths
        self.project_root = Path(__file__).parent.parent  # Go up one level from scripts/
        self.food_labels_file = self.project_root / "assets" / "models" / "food_labels.txt"
        self.collection_script = self.project_root / "scripts" / "collect_food_data.py"
        self.training_script = self.project_root / "scripts" / "train_simple_model.py"
        
        # Dataset configuration - now in ml_experiments/datasets folder
        self.dataset_name = f"auto_dataset_{num_categories}cat_{images_per_category}img"
        self.experiments_dir = self.project_root / "ml_experiments"
        self.datasets_dir = self.experiments_dir / "datasets"
        self.dataset_dir = self.datasets_dir / self.dataset_name
        self.model_dir = self.experiments_dir / "models" / f"{self.dataset_name}_models"
        
        # Ensure experiment directories exist
        self.experiments_dir.mkdir(exist_ok=True)
        self.datasets_dir.mkdir(exist_ok=True)
        (self.experiments_dir / "models").mkdir(exist_ok=True)
        
        print(f"🤖 Automated Food Classification Pipeline")
        print(f"=" * 50)
        print(f"📊 Configuration:")
        print(f"   Categories: {self.num_categories}")
        print(f"   Images per category: {self.images_per_category}")
        print(f"   Training epochs: {self.epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Dataset directory: {self.dataset_dir}")
        print(f"   Model directory: {self.model_dir}")
        print(f"   Experiments root: {self.experiments_dir}")
        print(f"=" * 50)
    
    def load_food_labels(self) -> List[str]:
        """Load food categories from food_labels.txt file."""
        
        print(f"📋 Loading food categories from {self.food_labels_file}...")
        
        if not self.food_labels_file.exists():
            print(f"❌ Food labels file not found: {self.food_labels_file}")
            # Fallback to basic categories
            return ["apple", "banana", "pizza", "bread", "chicken"]
        
        labels = []
        with open(self.food_labels_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Clean up the label (remove underscores, etc.)
                    clean_label = line.replace('_', ' ').lower()
                    labels.append(clean_label)
        
        print(f"✅ Loaded {len(labels)} food categories")
        return labels
    
    def select_random_categories(self, all_labels: List[str]) -> List[str]:
        """Randomly select categories for the experiment."""
        
        print(f"🎲 Selecting {self.num_categories} random categories from {len(all_labels)} available...")
        
        # Prefer single-word categories for better search results
        single_word_labels = [label for label in all_labels if ' ' not in label]
        
        if len(single_word_labels) >= self.num_categories:
            selected = random.sample(single_word_labels, self.num_categories)
        else:
            # Fall back to all labels if not enough single-word ones
            selected = random.sample(all_labels, min(self.num_categories, len(all_labels)))
        
        # Print detailed category selection
        print(f"✅ Selected categories for experiment:")
        for i, category in enumerate(selected, 1):
            print(f"   {i}. {category}")
        
        print(f"📋 Target: {self.images_per_category} images per category")
        print(f"🎯 Total target images: {len(selected) * self.images_per_category}")
        
        return selected
    
    def collect_dataset(self, categories: List[str]) -> bool:
        """Collect images for the selected categories."""
        
        print(f"\\n📥 Starting dataset collection...")
        print(f"🎯 Categories to process: {', '.join(categories)}")
        print(f"📊 Target: {self.images_per_category} images per category")
        print(f"🔢 Total target images: {len(categories) * self.images_per_category}")
        
        # Prepare command for collection script
        categories_str = ','.join(categories)
        
        cmd = [
            sys.executable,
            str(self.collection_script),
            "--categories", categories_str,
            "--images_per_category", str(self.images_per_category),
            "--output_dir", str(self.dataset_dir),
            "--parallel",
            "--max_workers", "4",
            "--train_split", "0.6",  # 60% train
            "--val_split", "0.2"     # 20% val, 20% test
        ]
        
        print(f"🚀 Running collection command:")
        print(f"   {' '.join(cmd)}")
        
        try:
            # Run the collection script with real-time output
            print(f"📊 Starting data collection (real-time output):")
            print(f"-" * 60)
            
            result = subprocess.run(cmd, timeout=1800)  # 30 min timeout, show output directly
            
            print(f"-" * 60)
            if result.returncode == 0:
                print(f"✅ Dataset collection completed successfully!")
                
                # Show download statistics
                self.show_download_statistics(categories)
                return True
            else:
                print(f"❌ Dataset collection failed with return code: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Dataset collection timed out after 30 minutes")
            return False
        except Exception as e:
            print(f"❌ Error during collection: {e}")
            return False
    
    def show_download_statistics(self, categories: List[str]) -> None:
        """Show detailed download statistics after collection."""
        
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
        
        # Check label mapping
        label_file = self.dataset_dir / "label_mapping.json"
        if label_file.exists():
            with open(label_file, 'r') as f:
                labels = json.load(f)
            print(f"\\n🏷️  Label Mapping:")
            for label, index in labels.items():
                print(f"   {index}: {label}")
        
        print(f"=" * 50)
    
    def verify_dataset(self) -> bool:
        """Verify that the dataset was created correctly."""
        
        print(f"\\n🔍 Verifying dataset structure...")
        
        # Check if dataset directory exists
        if not self.dataset_dir.exists():
            print(f"❌ Dataset directory not found: {self.dataset_dir}")
            return False
        
        # Check for required subdirectories
        required_dirs = ["train", "val", "test"]
        for dir_name in required_dirs:
            dir_path = self.dataset_dir / dir_name
            if not dir_path.exists():
                print(f"❌ Missing directory: {dir_path}")
                return False
        
        # Check if train directory has categories
        train_dir = self.dataset_dir / "train"
        categories = [d.name for d in train_dir.iterdir() if d.is_dir()]
        
        if len(categories) == 0:
            print(f"❌ No categories found in train directory")
            return False
        
        print(f"✅ Found {len(categories)} categories: {', '.join(categories)}")
        
        # Count images per category
        total_images = 0
        for cat_dir in train_dir.iterdir():
            if cat_dir.is_dir():
                images = len([f for f in cat_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                total_images += images
                print(f"   {cat_dir.name}: {images} training images")
        
        print(f"📊 Total training images: {total_images}")
        
        # Check label mapping file
        label_file = self.dataset_dir / "label_mapping.json"
        if label_file.exists():
            with open(label_file, 'r') as f:
                labels = json.load(f)
            print(f"✅ Label mapping: {labels}")
        else:
            print(f"⚠️  No label mapping file found")
        
        return True
    
    def train_model(self) -> bool:
        """Train the MobileNetV2 model on the collected dataset."""
        
        print(f"\\n🏋️  Starting model training...")
        
        # Prepare training command
        cmd = [
            sys.executable,
            str(self.training_script),
            "--dataset_dir", str(self.dataset_dir),
            "--model_dir", str(self.model_dir),
            "--epochs", str(self.epochs),
            "--batch_size", str(self.batch_size)
        ]
        
        print(f"🚀 Running training command:")
        print(f"   {' '.join(cmd)}")
        
        try:
            # Run the training script with real-time output
            print(f"🚀 Starting model training (real-time output):")
            print(f"-" * 60)
            
            result = subprocess.run(cmd, timeout=3600)  # 1 hour timeout, show output directly
            
            print(f"-" * 60)
            if result.returncode == 0:
                print(f"✅ Model training completed successfully!")
                return True
            else:
                print(f"❌ Model training failed with return code: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Model training timed out after 1 hour")
            return False
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False
    
    def analyze_results(self) -> Dict:
        """Analyze and report the training results."""
        
        print(f"\\n📊 Analyzing training results...")
        
        results = {
            "categories": [],
            "total_images": 0,
            "training_epochs": self.epochs,
            "final_accuracy": None,
            "model_size_mb": None,
            "tflite_size_mb": None
        }
        
        # Check if model files exist
        model_keras = self.model_dir / "food_classifier.keras"
        model_tflite = self.model_dir / "food_classifier.tflite"
        history_file = self.model_dir / "training_history.json"
        
        if model_keras.exists():
            size_mb = model_keras.stat().st_size / (1024 * 1024)
            results["model_size_mb"] = round(size_mb, 2)
            print(f"📦 Keras model size: {size_mb:.2f} MB")
        
        if model_tflite.exists():
            size_mb = model_tflite.stat().st_size / (1024 * 1024)
            results["tflite_size_mb"] = round(size_mb, 2)
            print(f"📱 TFLite model size: {size_mb:.2f} MB")
        
        # Read training history
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            # Get final validation accuracy
            if "val_accuracy" in history and history["val_accuracy"]:
                final_acc = max(history["val_accuracy"])
                results["final_accuracy"] = round(final_acc, 4)
                print(f"🎯 Best validation accuracy: {final_acc:.1%}")
            
            # Plot would be saved as training_plot.png
            plot_file = self.model_dir / "training_plot.png"
            if plot_file.exists():
                print(f"📈 Training plot saved: {plot_file}")
        
        # Get dataset info
        label_file = self.dataset_dir / "label_mapping.json"
        if label_file.exists():
            with open(label_file, 'r') as f:
                labels = json.load(f)
            results["categories"] = list(labels.keys())
        
        dataset_stats_file = self.dataset_dir / "dataset_stats.json"
        if dataset_stats_file.exists():
            with open(dataset_stats_file, 'r') as f:
                stats = json.load(f)
            results["total_images"] = stats.get("total_images", 0)
        
        return results
    
    def save_experiment_report(self, results: Dict) -> None:
        """Save a comprehensive experiment report."""
        
        report = {
            "experiment_config": {
                "num_categories": self.num_categories,
                "images_per_category": self.images_per_category,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "dataset_name": self.dataset_name
            },
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save experiment report in experiments directory
        report_file = self.experiments_dir / f"{self.dataset_name}_experiment_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Experiment report saved: {report_file}")
        
        # Print summary
        print(f"\\n🎉 EXPERIMENT SUMMARY")
        print(f"=" * 50)
        print(f"📊 Dataset: {len(results['categories'])} categories, {results['total_images']} total images")
        print(f"🏋️  Training: {results['training_epochs']} epochs")
        if results['final_accuracy']:
            print(f"🎯 Best Accuracy: {results['final_accuracy']:.1%}")
        if results['model_size_mb']:
            print(f"📦 Model Size: {results['model_size_mb']} MB (Keras)")
        if results['tflite_size_mb']:
            print(f"📱 TFLite Size: {results['tflite_size_mb']} MB")
        print(f"📁 Categories: {', '.join(results['categories'])}")
        print(f"=" * 50)
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete automated pipeline."""
        
        print(f"🚀 Starting complete automated pipeline...")
        
        # Step 1: Load food labels
        all_labels = self.load_food_labels()
        if not all_labels:
            print(f"❌ Failed to load food labels")
            return False
        
        # Step 2: Select categories
        selected_categories = self.select_random_categories(all_labels)
        
        # Step 3: Collect dataset
        if not self.collect_dataset(selected_categories):
            print(f"❌ Dataset collection failed")
            return False
        
        # Step 4: Verify dataset
        if not self.verify_dataset():
            print(f"❌ Dataset verification failed")
            return False
        
        # Step 5: Train model
        if not self.train_model():
            print(f"❌ Model training failed")
            return False
        
        # Step 6: Analyze results
        results = self.analyze_results()
        
        # Step 7: Save report
        self.save_experiment_report(results)
        
        print(f"\\n🎉 Pipeline completed successfully!")
        return True


def main():
    """Main function to run the automated pipeline."""
    
    parser = argparse.ArgumentParser(description='Automated Food Classification Pipeline')
    parser.add_argument('--categories', type=int, default=5,
                      help='Number of food categories to select (default: 5)')
    parser.add_argument('--images_per_category', type=int, default=15,
                      help='Number of images to collect per category (default: 15)')
    parser.add_argument('--epochs', type=int, default=15,
                      help='Number of training epochs (default: 15)')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Training batch size (default: 8)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducible category selection')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed:
        random.seed(args.seed)
        print(f"🌱 Using random seed: {args.seed}")
    
    # Create and run pipeline
    pipeline = AutomatedFoodPipeline(
        num_categories=args.categories,
        images_per_category=args.images_per_category,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    success = pipeline.run_complete_pipeline()
    
    if success:
        print(f"\\n✅ Automated pipeline completed successfully!")
        sys.exit(0)
    else:
        print(f"\\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
