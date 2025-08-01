#!/usr/bin/env python3
"""
Food Dataset Collection and Preparation Script
==============================================

This script helps collect and organize food images from various sources for training
a large-scale food recognition model.

Features:
- Web scraping from food websites
- Image downloading and validation
- Automatic labeling and organization
- Data quality checks
- Dataset splitting

Requirements:
- requests
- beautifulsoup4
- Pillow
- opencv-python
- pandas
- tqdm
"""

import requests
from bs4 import BeautifulSoup
import os
import json
import pandas as pd
from pathlib import Path
import hashlib
from PIL import Image
import cv2
import numpy as np
from urllib.parse import urljoin, urlparse
import time
import random
import re
import shutil
from tqdm import tqdm
from typing import List, Dict, Tuple
import argparse
import concurrent.futures
import threading
from functools import partial

class FoodDatasetCollector:
    """
    Collect and organize food images for training dataset.
    """
    
    def __init__(self, output_dir: str = "food_dataset", food_labels: List[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "raw").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "val").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Use custom food labels if provided, otherwise load from file
        if food_labels:
            self.food_labels = food_labels
            print(f"📝 Using custom food labels: {len(self.food_labels)} categories")
        else:
            self.food_labels = self.load_food_labels()
            print(f"📝 Loaded food labels from file: {len(self.food_labels)} categories")
        
    def load_food_labels(self) -> List[str]:
        """Load comprehensive food labels from file."""
        # Try multiple locations for comprehensive food labels
        labels_file = Path("../assets/models/comprehensive_food_labels.txt")
        if not labels_file.exists():
            labels_file = Path("assets/models/comprehensive_food_labels.txt")
        if not labels_file.exists():
            labels_file = Path("../assets/models/food_labels.txt")
        if not labels_file.exists():
            labels_file = Path("assets/models/food_labels.txt")
        
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                labels = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            return labels
        else:
            # Fallback to basic labels
            return [
                "apple", "banana", "orange", "chicken", "beef", "salmon",
                "bread", "rice", "pasta", "pizza", "burger", "salad"
            ]
    
    def search_food_images(self, food_name: str, max_images: int = 100) -> List[str]:
        """
        Search for food images using DuckDuckGo search only.
        Retrieves 2x max_images to account for failed downloads.
        
        Args:
            food_name: Name of the food to search for
            max_images: Target number of images to collect
            
        Returns:
            List of image URLs
        """
        urls = []
        
        # Search for 2x the target to account for failed downloads
        search_limit = max_images * 2
        
        print(f"Searching for {food_name} images using DuckDuckGo (target: {search_limit} URLs)...")
        try:
            ddg_urls = self._search_duckduckgo_images_primary(food_name, search_limit)
            urls.extend(ddg_urls)
            print(f"DuckDuckGo found {len(ddg_urls)} images for {food_name}")
        except Exception as e:
            print(f"DuckDuckGo search failed for {food_name}: {e}")
            return []
        
        # Remove duplicates and return
        unique_urls = list(dict.fromkeys(urls))
        return unique_urls[:search_limit]
    

    

    

    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL points to a valid image."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Check file extension
            path = parsed.path.lower()
            valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            if any(path.endswith(ext) for ext in valid_extensions):
                return True
            
            # Check if URL contains image indicators
            image_indicators = ['image', 'img', 'photo', 'picture']
            return any(indicator in url.lower() for indicator in image_indicators)
            
        except Exception:
            return False
    
    def download_images_parallel(self, food_name: str, target_images: int = 20, max_workers: int = 4) -> int:
        """
        Download images for a food category with parallel processing and smart distribution.
        
        Args:
            food_name: Food category name
            target_images: Target number of successful downloads
            max_workers: Number of parallel download threads
            
        Returns:
            Number of successfully downloaded images
        """
        print(f"Starting parallel download for {food_name} (target: {target_images} images)")
        
        # Get URLs (2x target to account for failures)
        urls = self.search_food_images(food_name, target_images)
        
        if not urls:
            print(f"No URLs found for {food_name}")
            return 0
        
        print(f"Found {len(urls)} URLs for {food_name}, starting download...")
        
        # Shared state for tracking downloads
        download_state = {
            'successful_downloads': 0,
            'processed_urls': 0,
            'lock': threading.Lock()
        }
        
        def download_worker(url_index_pair):
            url, index = url_index_pair
            
            # Check if we've reached our target
            with download_state['lock']:
                if download_state['successful_downloads'] >= target_images:
                    return False
                download_state['processed_urls'] += 1
            
            # Attempt download
            success = self.download_image(url, food_name, index)
            
            # Update shared state
            with download_state['lock']:
                if success:
                    download_state['successful_downloads'] += 1
                    print(f"✓ {food_name}: {download_state['successful_downloads']}/{target_images} downloaded")
                
                # Check if we've reached our target
                if download_state['successful_downloads'] >= target_images:
                    return True
            
            return success
        
        # Create URL-index pairs to ensure unique indices
        url_index_pairs = [(url, idx) for idx, url in enumerate(urls)]
        
        # Use ThreadPoolExecutor for parallel downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_url = {
                executor.submit(download_worker, pair): pair[0] 
                for pair in url_index_pairs
            }
            
            # Process completed downloads
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    result = future.result()
                    
                    # Check if we've reached our target
                    with download_state['lock']:
                        if download_state['successful_downloads'] >= target_images:
                            # Cancel remaining tasks
                            for remaining_future in future_to_url:
                                if not remaining_future.done():
                                    remaining_future.cancel()
                            break
                            
                except Exception as e:
                    url = future_to_url[future]
                    print(f"Download error for {url}: {e}")
        
        final_count = download_state['successful_downloads']
        print(f"Completed {food_name}: {final_count}/{target_images} images downloaded successfully")
        return final_count
    def download_image(self, url: str, food_name: str, index: int) -> bool:
        """
        Download and validate a single image.
        
        Args:
            url: Image URL
            food_name: Food category name
            index: Image index for naming
            
        Returns:
            True if successfully downloaded and validated
        """
        try:
            # Create directory for this food category
            category_dir = self.output_dir / "raw" / food_name
            category_dir.mkdir(exist_ok=True)
            
            # Generate filename based on URL or default to jpg
            parsed_url = urlparse(url)
            file_extension = 'jpg'
            
            # Try to detect extension from URL
            if '.' in parsed_url.path:
                potential_ext = parsed_url.path.split('.')[-1].lower()
                if potential_ext in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
                    file_extension = potential_ext
            
            filename = f"{food_name}_{index:05d}.{file_extension}"
            filepath = category_dir / filename
            
            # Skip if file already exists and is valid
            if filepath.exists():
                if self.validate_image(filepath):
                    return True
                else:
                    filepath.unlink()  # Remove invalid existing file
            
            # Set up headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            # Download image with proper headers and timeout
            response = self.session.get(url, headers=headers, timeout=15, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'webp']):
                return False
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 20_000_000:  # 20MB limit
                return False
            
            # Save image with size checking
            total_size = 0
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        total_size += len(chunk)
                        if total_size > 20_000_000:  # 20MB limit
                            filepath.unlink()  # Delete partial file
                            return False
                        f.write(chunk)
            
            # Validate image after download
            if self.validate_image(filepath):
                return True
            else:
                filepath.unlink()  # Delete invalid image
                return False
                
        except Exception as e:
            # Clean up partial file if it exists
            try:
                if 'filepath' in locals() and filepath.exists():
                    filepath.unlink()
            except:
                pass
            return False
    
    def validate_image(self, filepath: Path) -> bool:
        """
        Validate that an image is suitable for training.
        
        Args:
            filepath: Path to image file
            
        Returns:
            True if image is valid
        """
        try:
            # Check file size (should be reasonable)
            file_size = filepath.stat().st_size
            if file_size < 5000 or file_size > 20_000_000:  # 5KB to 20MB
                return False
            
            # Open with PIL to check format
            with Image.open(filepath) as img:
                # Check image size
                width, height = img.size
                if width < 100 or height < 100 or width > 5000 or height > 5000:
                    return False
                
                # Check aspect ratio (should be reasonable)
                aspect_ratio = width / height
                if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                    return False
                
                # Check if image has content (not all black/white)
                img_array = np.array(img.convert('RGB'))
                if img_array.std() < 10:  # Too uniform
                    return False
            
            return True
            
        except Exception:
            return False
    
    def process_images(self, target_size: Tuple[int, int] = (380, 380)) -> None:
        """
        Process raw images for training.
        
        Args:
            target_size: Target image size for training
        """
        raw_dir = self.output_dir / "raw"
        processed_dir = self.output_dir / "processed"
        
        print("Processing images...")
        
        for food_dir in tqdm(list(raw_dir.iterdir())):
            if not food_dir.is_dir():
                continue
                
            food_name = food_dir.name
            processed_food_dir = processed_dir / food_name
            processed_food_dir.mkdir(exist_ok=True)
            
            for img_path in food_dir.glob("*"):
                if not img_path.is_file():
                    continue
                
                try:
                    # Load and process image
                    with Image.open(img_path) as img:
                        # Convert to RGB
                        img = img.convert('RGB')
                        
                        # Resize maintaining aspect ratio
                        img.thumbnail(target_size, Image.Resampling.LANCZOS)
                        
                        # Create new image with target size and center the thumbnail
                        new_img = Image.new('RGB', target_size, (255, 255, 255))
                        paste_x = (target_size[0] - img.width) // 2
                        paste_y = (target_size[1] - img.height) // 2
                        new_img.paste(img, (paste_x, paste_y))
                        
                        # Save processed image
                        output_path = processed_food_dir / f"{img_path.stem}_processed.jpg"
                        new_img.save(output_path, 'JPEG', quality=85)
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    def split_dataset(self, 
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, 
                     test_ratio: float = 0.15) -> None:
        """
        Split processed images into train/validation/test sets.
        
        Args:
            train_ratio: Ratio of images for training
            val_ratio: Ratio of images for validation  
            test_ratio: Ratio of images for testing
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        processed_dir = self.output_dir / "processed"
        
        print("Splitting dataset...")
        
        for food_dir in tqdm(list(processed_dir.iterdir())):
            if not food_dir.is_dir():
                continue
                
            food_name = food_dir.name
            images = list(food_dir.glob("*.jpg"))
            
            if len(images) < 3:  # Skip categories with too few images (need at least 3 for train/val/test)
                print(f"Skipping {food_name}: only {len(images)} images")
                continue
            
            # Shuffle images
            random.shuffle(images)
            
            # Calculate split indices with better handling for small datasets
            n_images = len(images)
            
            if n_images >= 10:
                # Standard splitting for larger datasets
                train_end = int(n_images * train_ratio)
                val_end = train_end + int(n_images * val_ratio)
            else:
                # For small datasets, ensure each split gets at least 1 image
                if n_images >= 5:
                    train_end = max(1, int(n_images * 0.6))  # At least 60% for train
                    val_end = train_end + max(1, int(n_images * 0.2))  # At least 1 for val
                elif n_images >= 3:
                    train_end = n_images - 2  # Leave 2 for val+test
                    val_end = train_end + 1   # 1 for val
                else:
                    # Very small datasets - just use for training
                    train_end = n_images
                    val_end = n_images
            
            # Split images
            train_images = images[:train_end]
            val_images = images[train_end:val_end] if val_end > train_end else []
            test_images = images[val_end:] if val_end < n_images else []
            
            # Create directories and copy images
            for split_name, split_images in [
                ("train", train_images),
                ("val", val_images), 
                ("test", test_images)
            ]:
                split_dir = self.output_dir / split_name / food_name
                split_dir.mkdir(parents=True, exist_ok=True)
                
                for img_path in split_images:
                    output_path = split_dir / img_path.name
                    # Create hard link to save space
                    try:
                        output_path.hardlink_to(img_path)
                    except:
                        # Fallback to copy if hardlink fails
                        import shutil
                        shutil.copy2(img_path, output_path)
    
    def generate_dataset_statistics(self) -> Dict:
        """
        Generate statistics about the collected dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "total_categories": 0,
            "total_images": 0,
            "train_images": 0,
            "val_images": 0,
            "test_images": 0,
            "raw_images": 0,
            "categories": {}
        }
        
        # Check split datasets first
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / split
            if split_dir.exists():
                split_count = 0
                
                for food_dir in split_dir.iterdir():
                    if not food_dir.is_dir():
                        continue
                        
                    food_name = food_dir.name
                    image_count = len(list(food_dir.glob("*.jpg")))
                    split_count += image_count
                    
                    if food_name not in stats["categories"]:
                        stats["categories"][food_name] = {"train": 0, "val": 0, "test": 0, "raw": 0}
                    
                    stats["categories"][food_name][split] = image_count
                
                stats[f"{split}_images"] = split_count
        
        # If no split data exists, check raw data
        raw_dir = self.output_dir / "raw"
        if raw_dir.exists() and stats["total_images"] == 0:
            raw_count = 0
            
            for food_dir in raw_dir.iterdir():
                if not food_dir.is_dir():
                    continue
                    
                food_name = food_dir.name
                # Count all image files
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                    image_files.extend(food_dir.glob(ext))
                
                image_count = len(image_files)
                raw_count += image_count
                
                if food_name not in stats["categories"]:
                    stats["categories"][food_name] = {"train": 0, "val": 0, "test": 0, "raw": 0}
                
                stats["categories"][food_name]["raw"] = image_count
            
            stats["raw_images"] = raw_count
        
        stats["total_categories"] = len(stats["categories"])
        stats["total_images"] = sum(stats[f"{split}_images"] for split in ["train", "val", "test"]) + stats.get("raw_images", 0)
        
        # Save statistics to file
        with open(self.output_dir / "dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def create_label_mapping(self) -> None:
        """Create label mapping file for training."""
        train_dir = self.output_dir / "train"
        
        # Get all food categories
        categories = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        
        # Create mapping
        label_mapping = {category: idx for idx, category in enumerate(categories)}
        
        # Save mapping
        with open(self.output_dir / "label_mapping.json", "w") as f:
            json.dump(label_mapping, f, indent=2)
        
        # Save labels list
        with open(self.output_dir / "labels.txt", "w") as f:
            for category in categories:
                f.write(f"{category}\n")


    

    



    



    

    


    def _search_duckduckgo_images_primary(self, query: str, max_results: int = 40) -> List[str]:
        """
        Primary DuckDuckGo image search using the ddgs library.
        
        Args:
            query: Search query
            max_results: Maximum number of images to return
            
        Returns:
            List of image URLs
        """
        urls = []
        
        try:
            # Use the ddgs library for reliable DuckDuckGo search
            from ddgs import DDGS
            
            print(f"  Searching DuckDuckGo for: {query}")
            
            # Create multiple search variations to get more diverse results
            search_variations = [
                query,
                f"{query} food",
                f"fresh {query}",
            ]
            
            for search_query in search_variations:
                if len(urls) >= max_results:
                    break
                
                try:
                    with DDGS() as ddgs:
                        # Search for images with the correct API
                        images_per_variation = max_results // len(search_variations) + 5
                        results = ddgs.images(
                            query=search_query,
                            max_results=min(images_per_variation, 30)
                        )
                        
                        for result in results:
                            if len(urls) >= max_results:
                                break
                                
                            img_url = None
                            
                            # Extract image URL from result dictionary
                            if isinstance(result, dict):
                                img_url = result.get('image') or result.get('url') or result.get('src')
                            
                            if img_url and self._is_valid_image_url(img_url):
                                urls.append(img_url)
                    
                    # Small delay between search variations
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"  Error with search variation '{search_query}': {e}")
                    continue
            
            print(f"  DuckDuckGo found {len(urls)} images for {query}")
                    
        except ImportError:
            print("Error: ddgs library not found. Install with: pip install duckduckgo-search")
            return []
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
        
        return urls[:max_results]




    

    

    

    


def main():
    """Main function to run the food dataset collection from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect food images for training dataset')
    parser.add_argument('--categories', type=str, required=True,
                      help='Comma-separated list of food categories to collect')
    parser.add_argument('--images_per_category', type=int, default=50,
                      help='Number of images to collect per category')
    parser.add_argument('--output_dir', type=str, default='food_dataset',
                      help='Output directory for the dataset')
    parser.add_argument('--max_workers', type=int, default=4,
                      help='Maximum number of parallel image download workers per category')
    parser.add_argument('--max_category_workers', type=int, default=3,
                      help='Maximum number of categories to process simultaneously (category-level parallelization)')
    parser.add_argument('--parallel_categories', action='store_true',
                      help='Enable category-level parallelization for maximum speed')
    parser.add_argument('--target_size', type=int, nargs=2, default=[380, 380],
                      help='Target image size (width height)')
    parser.add_argument('--train_split', type=float, default=0.7,
                      help='Fraction of data for training')
    parser.add_argument('--val_split', type=float, default=0.2,
                      help='Fraction of data for validation')
    
    args = parser.parse_args()
    
    # Parse categories
    categories = [cat.strip() for cat in args.categories.split(',')]
    
    print("🍎 FoodGenius - Food Dataset Collection")
    print("=" * 40)
    print(f"Categories: {categories}")
    print(f"Images per category: {args.images_per_category}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max image workers per category: {args.max_workers}")
    if args.parallel_categories:
        print(f"Max category workers: {args.max_category_workers}")
        print(f"🚀 DUAL-LEVEL parallelization enabled")
    else:
        print(f"🔄 Sequential category processing")
    
    # Create collector with custom categories
    collector = FoodDatasetCollector(output_dir=args.output_dir, food_labels=categories)
    
    try:
        if args.parallel_categories:
            # Use dual-level parallelization: categories + images in parallel
            print(f"\n🚀 Starting DUAL-LEVEL parallel collection")
            print(f"⚡ Processing up to {args.max_category_workers} categories simultaneously")
            print(f"⚡ Each category downloads {args.max_workers} images in parallel")
            
            # Function to collect images for a single category
            def collect_category(category_info):
                idx, category = category_info
                print(f"\n[{idx+1}/{len(categories)}] Starting: {category}")
                downloaded_count = collector.download_images_parallel(
                    food_name=category, 
                    target_images=args.images_per_category,
                    max_workers=args.max_workers
                )
                print(f"✅ Completed {category}: {downloaded_count} images downloaded")
                return category, downloaded_count
            
            # Process categories in parallel using ThreadPoolExecutor
            total_stats = {
                'total_downloaded': 0,
                'successful_categories': 0,
                'category_results': []
            }
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_category_workers) as executor:
                # Submit all category collection tasks
                category_info_list = [(idx, category) for idx, category in enumerate(categories)]
                future_to_category = {
                    executor.submit(collect_category, category_info): category_info[1] 
                    for category_info in category_info_list
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_category):
                    category = future_to_category[future]
                    try:
                        category_name, downloaded_count = future.result()
                        total_stats['total_downloaded'] += downloaded_count
                        if downloaded_count > 0:
                            total_stats['successful_categories'] += 1
                        total_stats['category_results'].append({
                            'category': category_name,
                            'downloaded': downloaded_count
                        })
                    except Exception as e:
                        print(f"❌ Error processing category {category}: {e}")
            
            # Print summary
            print(f"\n🎉 DUAL-LEVEL parallel collection complete!")
            print(f"✅ Categories processed: {len(categories)}")
            print(f"✅ Successful categories: {total_stats['successful_categories']}")
            print(f"📥 Total images downloaded: {total_stats['total_downloaded']}")
            
        else:
            # Sequential category processing (original behavior)
            print(f"\n🚀 Starting sequential category collection")
            for idx, category in enumerate(categories):
                print(f"\n[{idx+1}/{len(categories)}] Processing: {category}")
                downloaded_count = collector.download_images_parallel(
                    food_name=category, 
                    target_images=args.images_per_category,
                    max_workers=args.max_workers
                )
                print(f"✅ Completed {category}: {downloaded_count} images downloaded")
        
        # Process images (resize, validate)
        print(f"\n🔄 Processing images...")
        collector.process_images(target_size=tuple(args.target_size))
        
        # Split dataset
        print(f"\n📊 Splitting dataset...")
        test_split = 1.0 - args.train_split - args.val_split
        collector.split_dataset(
            train_ratio=args.train_split,
            val_ratio=args.val_split,
            test_ratio=test_split
        )
        
        # Generate statistics
        print(f"\n📈 Generating statistics...")
        stats = collector.generate_dataset_statistics()
        
        # Create label mapping
        print(f"\n🏷️  Creating label mapping...")
        collector.create_label_mapping()
        
        print(f"\n🎉 Dataset collection complete!")
        print(f"📁 Dataset saved to: {args.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
