#!/usr/bin/env python3
"""
Large-Scale Food Dataset Collection and Training Pipeline
========================================================

This script implements a complete pipeline for collecting and training on
thousands of food categories for the FoodGenius app.

Steps:
1. Download public food datasets
2. Web scraping for additional data (responsibly)
3. Data validation and preprocessing
4. Create balanced train/val/test splits
5. Train EfficientNet model on thousands of categories
6. Optimize for mobile deployment

Usage:
    python collect_and_train.py --target_classes 2000 --min_images_per_class 1000
"""

import os
import sys
import json
import time
import shutil
import hashlib
import random
import requests
import tarfile
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import argparse
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageOps
    import cv2
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    print(f"Warning: Some packages not available: {e}")
    print("Install with: pip install tensorflow pandas pillow opencv-python tqdm scikit-learn")

class LargeFoodDatasetCollector:
    """Collect and prepare large-scale food dataset."""
    
    def __init__(self, output_dir: str = "food_dataset_mega", target_classes: int = 2000):
        self.output_dir = Path(output_dir)
        self.target_classes = target_classes
        self.min_images_per_class = 1000
        
        # Create directory structure
        self.setup_directories()
        
        # Load comprehensive food labels
        self.food_labels = self.load_comprehensive_labels()
        
        # Statistics
        self.stats = {
            'downloaded_datasets': [],
            'collected_images': defaultdict(int),
            'total_images': 0,
            'rejected_images': 0,
            'duplicate_images': 0
        }
        
        # Session for web requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def setup_directories(self):
        """Create directory structure for dataset."""
        dirs = ['downloads', 'raw', 'processed', 'train', 'val', 'test', 'metadata', 'rejected']
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def load_comprehensive_labels(self) -> List[str]:
        """Load comprehensive food labels."""
        labels_file = Path("assets/models/comprehensive_food_labels.txt")
        if not labels_file.exists():
            return self.get_default_labels()
        
        with open(labels_file, 'r') as f:
            labels = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    labels.append(line.lower().replace(' ', '_'))
        
        print(f"📋 Loaded {len(labels)} comprehensive food labels")
        return labels[:self.target_classes]  # Limit to target number
    
    def get_default_labels(self) -> List[str]:
        """Default food labels if comprehensive file not found."""
        return [
            'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry',
            'chicken', 'beef', 'pork', 'salmon', 'tuna', 'shrimp',
            'bread', 'rice', 'pasta', 'pizza', 'burger', 'sandwich',
            'salad', 'soup', 'cake', 'cookie', 'ice_cream', 'coffee'
        ]
    
    def download_food101_dataset(self) -> bool:
        """Download Food-101 dataset."""
        print("📥 Downloading Food-101 dataset...")
        try:
            url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
            download_path = self.output_dir / "downloads" / "food-101.tar.gz"
            
            if not download_path.exists():
                print("   Downloading Food-101 archive...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(download_path, 'wb') as f, tqdm(
                    desc="Food-101", total=total_size, unit='B', unit_scale=True
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
            
            # Extract
            extract_path = self.output_dir / "raw" / "food-101"
            if not extract_path.exists():
                print("   Extracting Food-101...")
                with tarfile.open(download_path, 'r:gz') as tar:
                    tar.extractall(self.output_dir / "raw")
            
            self.stats['downloaded_datasets'].append('Food-101')
            return True
            
        except Exception as e:
            print(f"❌ Error downloading Food-101: {e}")
            return False
    
    def download_recipe1m_dataset(self) -> bool:
        """Download Recipe1M+ dataset (images only)."""
        print("📥 Attempting to download Recipe1M+ dataset...")
        try:
            # Note: Recipe1M+ requires special access, this is a placeholder
            # In practice, you would need to request access from the authors
            print("   ⚠️ Recipe1M+ requires special access - skipping for now")
            print("   Visit: http://pic2recipe.csail.mit.edu/ to request access")
            return False
            
        except Exception as e:
            print(f"❌ Error with Recipe1M+: {e}")
            return False
    
    def collect_web_images(self, food_label: str, max_images: int = 200) -> List[str]:
        """
        Collect images for a specific food from web sources.
        
        Note: This is a simplified example. In production, use official APIs
        and respect robots.txt and rate limits.
        """
        print(f"🔍 Searching web for '{food_label}' images...")
        
        # This is a placeholder - in practice you would use:
        # 1. Google Custom Search API
        # 2. Bing Image Search API
        # 3. Flickr API
        # 4. Unsplash API
        # 5. Food-specific websites with permission
        
        # Example search queries
        search_queries = [
            f"{food_label} food",
            f"fresh {food_label}",
            f"{food_label} dish",
            f"cooked {food_label}",
            f"{food_label} recipe"
        ]
        
        collected_urls = []
        
        # Placeholder: return empty list for now
        # In production, implement actual web scraping here
        print(f"   ⚠️ Web scraping not implemented - would search for {max_images} images")
        
        return collected_urls
    
    def download_image(self, url: str, save_path: Path) -> bool:
        """Download and validate a single image."""
        try:
            response = self.session.get(url, timeout=10, stream=True)
            response.raise_for_status()
            
            # Save image
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validate image
            if self.validate_image(save_path):
                return True
            else:
                save_path.unlink()  # Delete invalid image
                return False
                
        except Exception as e:
            if save_path.exists():
                save_path.unlink()
            return False
    
    def validate_image(self, img_path: Path) -> bool:
        """Validate image quality and format."""
        try:
            # Check file size (minimum 5KB)
            if img_path.stat().st_size < 5 * 1024:
                return False
            
            # Try to open with PIL
            with Image.open(img_path) as img:
                # Check minimum dimensions
                if img.width < 150 or img.height < 150:
                    return False
                
                # Check aspect ratio (not too extreme)
                aspect_ratio = max(img.width, img.height) / min(img.width, img.height)
                if aspect_ratio > 3.0:
                    return False
                
                # Check format
                if img.format not in ['JPEG', 'PNG', 'RGB']:
                    return False
                
                # Verify image
                img.verify()
            
            # Additional check with OpenCV
            cv_img = cv2.imread(str(img_path))
            if cv_img is None:
                return False
                
            return True
            
        except Exception:
            return False
    
    def process_food101_images(self):
        """Process Food-101 dataset images."""
        food101_path = self.output_dir / "raw" / "food-101"
        
        if not food101_path.exists():
            print("❌ Food-101 not found, skipping...")
            return
        
        print("📊 Processing Food-101 images...")
        
        # Load train/test splits
        train_file = food101_path / "meta" / "train.txt"
        test_file = food101_path / "meta" / "test.txt"
        
        if not (train_file.exists() and test_file.exists()):
            print("❌ Food-101 metadata not found")
            return
        
        # Read image lists
        with open(train_file, 'r') as f:
            train_images = [line.strip() for line in f]
        with open(test_file, 'r') as f:
            test_images = [line.strip() for line in f]
        
        all_images = train_images + test_images
        images_dir = food101_path / "images"
        
        # Process images
        for img_file in tqdm(all_images, desc="Processing Food-101"):
            src_path = images_dir / f"{img_file}.jpg"
            if not src_path.exists():
                continue
                
            # Extract class name
            class_name = img_file.split('/')[0].lower().replace(' ', '_')
            
            # Map to our comprehensive labels if possible
            if class_name in self.food_labels:
                self.copy_image_to_processed(src_path, class_name)
    
    def copy_image_to_processed(self, src_path: Path, class_name: str):
        """Copy and deduplicate image to processed directory."""
        try:
            # Validate image first
            if not self.validate_image(src_path):
                self.stats['rejected_images'] += 1
                return False
            
            # Create class directory
            class_dir = self.output_dir / "processed" / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Generate unique filename based on content hash
            img_hash = self.get_image_hash(src_path)
            dst_path = class_dir / f"{img_hash}.jpg"
            
            # Check for duplicates
            if dst_path.exists():
                self.stats['duplicate_images'] += 1
                return False
            
            # Copy image
            shutil.copy2(src_path, dst_path)
            
            # Update statistics
            self.stats['collected_images'][class_name] += 1
            self.stats['total_images'] += 1
            
            return True
            
        except Exception as e:
            print(f"Error processing {src_path}: {e}")
            return False
    
    def get_image_hash(self, img_path: Path) -> str:
        """Generate hash for duplicate detection."""
        with open(img_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def create_balanced_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Create balanced train/validation/test splits."""
        print("🔄 Creating balanced dataset splits...")
        
        processed_dir = self.output_dir / "processed"
        
        # Find classes with sufficient images
        valid_classes = []
        for class_dir in processed_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg"))
                if len(images) >= self.min_images_per_class:
                    valid_classes.append((class_dir.name, len(images)))
                    
        # Sort by number of images and take top classes
        valid_classes.sort(key=lambda x: x[1], reverse=True)
        selected_classes = valid_classes[:self.target_classes]
        
        print(f"✅ Selected {len(selected_classes)} classes with sufficient images")
        
        # Create splits
        for class_name, num_images in tqdm(selected_classes, desc="Creating splits"):
            class_dir = processed_dir / class_name
            images = list(class_dir.glob("*.jpg"))
            
            # Shuffle
            random.shuffle(images)
            
            # Calculate split sizes
            n_train = int(len(images) * train_ratio)
            n_val = int(len(images) * val_ratio)
            
            # Create directories
            for split in ['train', 'val', 'test']:
                (self.output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
            
            # Copy images to splits
            for i, img_path in enumerate(images):
                if i < n_train:
                    dst_dir = self.output_dir / "train" / class_name
                elif i < n_train + n_val:
                    dst_dir = self.output_dir / "val" / class_name
                else:
                    dst_dir = self.output_dir / "test" / class_name
                
                shutil.copy2(img_path, dst_dir / img_path.name)
        
        print(f"✅ Created splits for {len(selected_classes)} classes")
    
    def generate_statistics(self):
        """Generate comprehensive dataset statistics."""
        print("📊 Generating dataset statistics...")
        
        stats = {
            'dataset_info': {
                'total_classes': 0,
                'total_images': 0,
                'target_classes': self.target_classes,
                'min_images_per_class': self.min_images_per_class
            },
            'collection_stats': self.stats,
            'splits': {}
        }
        
        # Analyze splits
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            if split_dir.exists():
                classes = 0
                images = 0
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        class_images = len(list(class_dir.glob("*.jpg")))
                        if class_images > 0:
                            classes += 1
                            images += class_images
                
                stats['splits'][split] = {
                    'classes': classes,
                    'images': images
                }
        
        # Overall stats
        if 'train' in stats['splits']:
            stats['dataset_info']['total_classes'] = stats['splits']['train']['classes']
            stats['dataset_info']['total_images'] = sum(
                split_info['images'] for split_info in stats['splits'].values()
            )
        
        # Save statistics
        with open(self.output_dir / "metadata" / "collection_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 DATASET COLLECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Target Classes: {stats['dataset_info']['target_classes']}")
        print(f"Collected Classes: {stats['dataset_info']['total_classes']}")
        print(f"Total Images: {stats['dataset_info']['total_images']}")
        
        for split, info in stats['splits'].items():
            print(f"{split.capitalize()}: {info['classes']} classes, {info['images']} images")
        
        print(f"Rejected Images: {self.stats['rejected_images']}")
        print(f"Duplicate Images: {self.stats['duplicate_images']}")
        print(f"Downloaded Datasets: {', '.join(self.stats['downloaded_datasets'])}")
        print(f"{'='*60}")
        
        return stats
    
    def run_collection_pipeline(self):
        """Run the complete data collection pipeline."""
        print("🚀 Starting large-scale food dataset collection...")
        print(f"🎯 Target: {self.target_classes} classes with {self.min_images_per_class}+ images each")
        
        # Phase 1: Download USDA FoodData Central dataset
        print("\n📥 Phase 1: Collecting USDA FoodData Central dataset...")
        if self.collect_usda_dataset():
            print("✅ USDA dataset collection successful!")
        else:
            print("⚠️ USDA dataset collection failed, continuing with other sources...")
        
        # Phase 2: Download public image datasets
        print("\n📥 Phase 2: Downloading public image datasets...")
        self.download_food101_dataset()
        self.download_recipe1m_dataset()
        
        # Phase 3: Process downloaded datasets
        print("\n🔄 Phase 3: Processing downloaded images...")
        self.process_food101_images()
        
        # TODO: Add more dataset processing here
        # - iFood2019
        # - Open Images food subset
        # - Custom web scraping
        
        # Phase 4: Create balanced splits
        print("\n🎲 Phase 4: Creating balanced splits...")
        self.create_balanced_splits()
        
        # Phase 5: Generate statistics
        print("\n📊 Phase 5: Generating statistics...")
        stats = self.generate_statistics()
        
        print("\n✅ Dataset collection completed!")
        return stats
    
    def collect_usda_dataset(self) -> bool:
        """Collect USDA FoodData Central dataset."""
        try:
            print("📋 Running USDA FoodData Central collector...")
            
            # Try to run the USDA collector
            import subprocess
            import sys
            
            usda_output_dir = self.output_dir / "usda_data"
            
            result = subprocess.run([
                sys.executable, 'scripts/collect_usda_dataset.py',
                '--output_dir', str(usda_output_dir)
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                print("✅ USDA dataset collected successfully!")
                
                # Process USDA data into our format
                self.process_usda_data(usda_output_dir)
                return True
            else:
                print(f"❌ USDA collection failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error collecting USDA dataset: {e}")
            return False
    
    def process_usda_data(self, usda_dir: Path):
        """Process USDA data into training format."""
        print("🔄 Processing USDA data for training...")
        
        try:
            usda_file = usda_dir / "processed" / "usda_food_dataset.json"
            if not usda_file.exists():
                print(f"❌ USDA dataset file not found: {usda_file}")
                return
            
            with open(usda_file, 'r') as f:
                usda_data = json.load(f)
            
            # Process each category
            for category, foods in usda_data['categories'].items():
                if len(foods) < 50:  # Skip categories with too few items
                    continue
                
                category_dir = self.output_dir / "processed" / category
                category_dir.mkdir(exist_ok=True)
                
                # Create placeholder images for each food item
                for i, food in enumerate(foods[:500]):  # Limit to 500 per category
                    # Create a simple text file as placeholder
                    # In production, this would download actual food images
                    placeholder_path = category_dir / f"{food['name']}_{i:03d}.txt"
                    
                    with open(placeholder_path, 'w') as f:
                        f.write(json.dumps(food, indent=2))
                    
                    # Update stats
                    self.stats['collected_images'][category] += 1
            
            print(f"✅ Processed USDA data: {len(usda_data['categories'])} categories")
            
        except Exception as e:
            print(f"❌ Error processing USDA data: {e}")


class LargeFoodModelTrainer:
    """Train large-scale food recognition model."""
    
    def __init__(self, data_dir: str, num_classes: int = 2000):
        self.data_dir = Path(data_dir)
        self.num_classes = num_classes
        self.input_size = (224, 224)  # Mobile-optimized
        self.batch_size = 32
        self.epochs = 100
        
        # Model will be set during training
        self.model = None
        
    def create_data_generators(self):
        """Create data generators with augmentation."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available for training")
        
        print("📊 Creating data generators...")
        
        # Training data with heavy augmentation
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.1,
            fill_mode='nearest'
        )
        
        # Validation data (no augmentation)
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.data_dir / "train",
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.data_dir / "val",
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Update num_classes if different
        if train_generator.num_classes != self.num_classes:
            print(f"📝 Updating num_classes: {self.num_classes} -> {train_generator.num_classes}")
            self.num_classes = train_generator.num_classes
        
        print(f"✅ Train: {train_generator.samples} images, {train_generator.num_classes} classes")
        print(f"✅ Val: {val_generator.samples} images, {val_generator.num_classes} classes")
        
        return train_generator, val_generator
    
    def create_efficientnet_model(self):
        """Create EfficientNet model for large-scale classification."""
        print(f"🏗️ Creating EfficientNet model for {self.num_classes} classes...")
        
        # Use EfficientNetB1 for good balance of accuracy and speed
        base_model = tf.keras.applications.EfficientNetB1(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.input_size, 3),
            pooling='avg'
        )
        
        # Add custom classification head
        inputs = base_model.input
        x = base_model.output
        
        # Add dropout and dense layers
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Final classification layer
        outputs = tf.keras.layers.Dense(
            self.num_classes,
            activation='softmax',
            dtype='float32'
        )(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Compile with appropriate settings for large-scale classification
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=1e-4,
            weight_decay=1e-4
        )
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy', 'top_5_categorical_accuracy']
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        return model
    
    def create_callbacks(self):
        """Create training callbacks."""
        callbacks = [
            # Save best model
            tf.keras.callbacks.ModelCheckpoint(
                self.data_dir / "best_food_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Reduce learning rate
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir=self.data_dir / "logs",
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train_model(self):
        """Train the food recognition model."""
        print("🚀 Starting model training...")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators()
        
        # Create model
        self.model = self.create_efficientnet_model()
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        print(f"🏋️ Training for {self.epochs} epochs...")
        history = self.model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return history
    
    def convert_to_tflite(self, model_path: Optional[Path] = None):
        """Convert model to TensorFlow Lite for mobile deployment."""
        if model_path is None:
            model_path = self.data_dir / "best_food_model.h5"
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        print("📱 Converting to TensorFlow Lite...")
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Convert to TFLite with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.lite.constants.INT8]
        
        # Representative dataset for quantization
        def representative_dataset():
            # Use some validation data
            val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
            val_data = val_gen.flow_from_directory(
                self.data_dir / "val",
                target_size=self.input_size,
                batch_size=1,
                class_mode=None,
                shuffle=False
            )
            
            for i, batch in enumerate(val_data):
                if i >= 100:  # Use 100 samples
                    break
                yield [batch.astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        
        # Convert
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = Path("assets/models/efficientnet_food_classifier.tflite")
        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite model saved: {tflite_path}")
        print(f"   Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Collect large food dataset and train model")
    parser.add_argument("--target_classes", type=int, default=2000,
                       help="Target number of food classes")
    parser.add_argument("--min_images_per_class", type=int, default=1000,
                       help="Minimum images per class")
    parser.add_argument("--data_dir", default="food_dataset_mega",
                       help="Dataset output directory")
    parser.add_argument("--skip_collection", action="store_true",
                       help="Skip data collection, use existing dataset")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training, only collect data")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Phase 1: Data Collection
    if not args.skip_collection:
        print("🎯 PHASE 1: LARGE-SCALE DATASET COLLECTION")
        print("=" * 60)
        
        collector = LargeFoodDatasetCollector(
            output_dir=args.data_dir,
            target_classes=args.target_classes
        )
        collector.min_images_per_class = args.min_images_per_class
        
        stats = collector.run_collection_pipeline()
        
        if stats['dataset_info']['total_classes'] < 100:
            print("❌ Insufficient classes collected for meaningful training")
            sys.exit(1)
    
    # Phase 2: Model Training
    if not args.skip_training:
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available for training")
            print("Install with: pip install tensorflow")
            sys.exit(1)
        
        print(f"\n🎯 PHASE 2: LARGE-SCALE MODEL TRAINING")
        print("=" * 60)
        
        trainer = LargeFoodModelTrainer(
            data_dir=args.data_dir,
            num_classes=args.target_classes
        )
        trainer.epochs = args.epochs
        
        # Train model
        history = trainer.train_model()
        
        # Convert to TFLite
        trainer.convert_to_tflite()
        
        print("\n🎉 TRAINING PIPELINE COMPLETED!")
        print("=" * 60)
        print("✅ Large-scale food recognition model ready for deployment")
        print("✅ TensorFlow Lite model created for mobile app")
        print("\n🎯 Next steps:")
        print("   1. Test the model in the Flutter app")
        print("   2. Collect user feedback and iterate")
        print("   3. Fine-tune model based on real usage")


if __name__ == "__main__":
    main()
