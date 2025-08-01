#!/usr/bin/env python3
"""
Food Category Consolidation Script
=================================

This script consolidates collected food data by grouping similar foods into broader categories.
This increases the number of samples per category, improving model training performance.

Example usage:
python scripts/consolidate_food_categories.py --input_dir ml_experiments/datasets/experiment_dataset --output_dir ml_experiments/datasets/consolidated_dataset
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

# Ensure we can run from either root directory or scripts directory
if Path.cwd().name == 'scripts':
    os.chdir('..')  # Change to parent directory (project root)

class FoodCategoryConsolidator:
    def __init__(self):
        """Initialize the food category consolidation mappings."""
        
        # Define comprehensive category mappings
        self.category_mappings = {
            # FRESH FRUITS - Group similar fruits but keep unique ones separate
            'citrus_fruits': [
                'orange', 'lemon', 'lime', 'grapefruit'
            ],
            'berries': [
                'strawberry', 'blueberry', 'raspberry', 'blackberry', 'cranberry'
            ],
            'tropical_fruits': [
                'pineapple', 'mango', 'papaya', 'dragon_fruit', 'passion_fruit', 
                'star_fruit', 'lychee', 'rambutan', 'durian', 'jackfruit', 'guava'
            ],
            'stone_fruits': [
                'peach', 'plum', 'apricot', 'cherry'
            ],
            'melons': [
                'watermelon', 'cantaloupe', 'honeydew'
            ],
            'dried_fruits': [
                'date', 'prune', 'fig'
            ],
            # Keep unique: apple, banana, grape, pear, kiwi, avocado, coconut, pomegranate, persimmon
            
            # VEGETABLES - Group by type/family
            'leafy_greens': [
                'lettuce', 'spinach', 'kale', 'arugula', 'bok_choy'
            ],
            'cruciferous_vegetables': [
                'broccoli', 'cauliflower', 'brussels_sprouts', 'cabbage'
            ],
            'root_vegetables': [
                'carrot', 'beet', 'radish', 'turnip', 'sweet_potato'
            ],
            'alliums': [
                'onion', 'shallot', 'garlic', 'leek', 'scallion', 'chives'
            ],
            'peppers': [
                'bell_pepper', 'jalapeño', 'habanero'
            ],
            'herbs': [
                'parsley', 'cilantro', 'basil', 'oregano', 'thyme', 'rosemary', 
                'sage', 'dill', 'mint', 'fennel'
            ],
            'squash': [
                'zucchini', 'eggplant'
            ],
            # Keep unique: tomato, cucumber, asparagus, green_beans, peas, corn, potato, mushroom, okra, artichoke, celery, ginger, turmeric, horseradish, wasabi
            
            # MEAT & POULTRY - Group by animal type
            'chicken': [
                'chicken_breast', 'chicken_thigh', 'chicken_wing', 'chicken_drumstick', 
                'whole_chicken', 'ground_chicken', 'chicken_liver'
            ],
            'beef': [
                'beef_steak', 'ground_beef', 'beef_roast', 'brisket', 'short_ribs'
            ],
            'pork': [
                'pork_chop', 'pork_tenderloin', 'pork_shoulder', 'pork_belly', 'pork_ribs', 'ground_pork'
            ],
            'cured_meats': [
                'bacon', 'ham', 'prosciutto', 'sausage', 'chorizo', 'pepperoni', 'salami'
            ],
            'game_meat': [
                'lamb', 'venison', 'rabbit', 'veal'
            ],
            'poultry_other': [
                'turkey', 'duck', 'goose', 'quail'
            ],
            
            # SEAFOOD & FISH - Group by type
            'white_fish': [
                'cod', 'halibut', 'sea_bass', 'red_snapper', 'flounder', 'sole', 'tilapia'
            ],
            'oily_fish': [
                'salmon', 'tuna', 'mackerel', 'sardines', 'anchovies', 'herring'
            ],
            'freshwater_fish': [
                'trout', 'catfish', 'carp', 'pike', 'perch'
            ],
            'exotic_fish': [
                'mahi_mahi', 'swordfish'
            ],
            'shellfish': [
                'shrimp', 'prawns', 'crab', 'lobster', 'crawfish'
            ],
            'mollusks': [
                'scallops', 'mussels', 'clams', 'oysters', 'squid', 'octopus'
            ],
            'fish_products': [
                'sea_urchin', 'caviar', 'roe'
            ],
            
            # DAIRY & ALTERNATIVES
            'milk_alternatives': [
                'almond_milk', 'soy_milk', 'oat_milk', 'coconut_milk', 'rice_milk'
            ],
            'soft_cheese': [
                'cottage_cheese', 'ricotta_cheese', 'cream_cheese', 'feta_cheese', 
                'goat_cheese', 'brie_cheese'
            ],
            'hard_cheese': [
                'cheddar_cheese', 'swiss_cheese', 'parmesan_cheese', 'gouda_cheese'
            ],
            'cream_products': [
                'heavy_cream', 'sour_cream', 'butter'
            ],
            'frozen_dairy': [
                'ice_cream', 'gelato', 'frozen_yogurt'
            ],
            # Keep unique: milk, yogurt, mozzarella_cheese, blue_cheese, eggs
            
            # GRAINS & STARCHES
            'whole_grains': [
                'quinoa', 'oats', 'barley', 'bulgur', 'farro', 'spelt', 'millet', 
                'buckwheat', 'amaranth', 'teff'
            ],
            'pasta': [
                'pasta', 'spaghetti', 'linguine', 'fettuccine', 'penne', 'rigatoni', 
                'fusilli', 'farfalle', 'macaroni', 'ravioli', 'tortellini'
            ],
            'noodles': [
                'ramen_noodles', 'udon_noodles', 'soba_noodles', 'rice_noodles', 'egg_noodles'
            ],
            'breakfast_grains': [
                'cereal', 'oatmeal', 'granola'
            ],
            'baked_goods': [
                'bread', 'bagel', 'baguette', 'croissant', 'crackers'
            ],
            'breakfast_items': [
                'pancake', 'waffle', 'french_toast'
            ],
            # Keep unique: rice, wheat, couscous, polenta, gnocchi, tortilla, flour
            
            # LEGUMES & NUTS
            'beans': [
                'beans', 'black_beans', 'pinto_beans', 'kidney_beans', 'chickpeas'
            ],
            'lentils_peas': [
                'lentils', 'split_peas', 'edamame'
            ],
            'soy_products': [
                'tofu', 'tempeh'
            ],
            'tree_nuts': [
                'almonds', 'walnuts', 'pecans', 'cashews', 'pistachios', 'hazelnuts', 
                'macadamia_nuts', 'brazil_nuts', 'pine_nuts'
            ],
            'seeds': [
                'sunflower_seeds', 'pumpkin_seeds', 'chia_seeds', 'flax_seeds', 'sesame_seeds'
            ],
            'nut_butters': [
                'peanut_butter', 'almond_butter', 'cashew_butter', 'tahini'
            ],
            # Keep unique: peanuts (technically legume)
            
            # BEVERAGES
            'water_variants': [
                'water', 'sparkling_water', 'coconut_water'
            ],
            'hot_beverages': [
                'coffee', 'espresso', 'tea'
            ],
            'cold_beverages': [
                'juice', 'smoothie', 'soda'
            ],
            'alcoholic_beverages': [
                'wine', 'beer', 'spirits'
            ],
            
            # DESSERTS & SWEETS
            'cakes_pastries': [
                'cake', 'cupcake', 'muffin', 'pie', 'tart', 'donut'
            ],
            'sweet_treats': [
                'cookie', 'brownie', 'candy', 'chocolate'
            ],
            'puddings_creams': [
                'pudding', 'custard', 'mousse'
            ],
            
            # CONDIMENTS & SAUCES
            'basic_condiments': [
                'ketchup', 'mustard', 'mayo'
            ],
            'sauces': [
                'hot_sauce', 'barbecue_sauce', 'soy_sauce'
            ],
            'cooking_oils': [
                'oil', 'olive_oil'
            ],
            'sweeteners': [
                'jam', 'honey', 'maple_syrup'
            ],
            # Keep unique: vinegar, butter (already in cream_products)
            
            # INTERNATIONAL DISHES - Keep most as unique due to complexity
            'italian_dishes': [
                'pizza', 'pasta', 'risotto'
            ],
            'spanish_dishes': [
                'paella'
            ],
            'mexican_dishes': [
                'tacos', 'burritos', 'quesadilla'
            ],
            'asian_dishes': [
                'sushi', 'ramen', 'fried_rice', 'pho', 'dumplings'
            ],
            'indian_dishes': [
                'curry', 'biryani', 'naan'
            ],
            'mediterranean_dishes': [
                'kebab', 'gyros'
            ],
            
            # PROCESSED FOODS
            'snack_foods': [
                'chips', 'pretzels', 'popcorn', 'granola_bar'
            ],
            'preserved_foods': [
                'canned_soup', 'pickles', 'olives'
            ]
        }
        
        # Create reverse mapping for easy lookup
        self.item_to_category = {}
        for category, items in self.category_mappings.items():
            for item in items:
                self.item_to_category[item] = category
        
        # Items that should remain as individual categories (unique foods)
        self.unique_items = {
            # Unique fruits
            'apple', 'banana', 'grape', 'pear', 'kiwi', 'avocado', 'coconut', 'pomegranate', 'persimmon',
            
            # Unique vegetables
            'tomato', 'cucumber', 'asparagus', 'green_beans', 'peas', 'corn', 'potato', 
            'mushroom', 'okra', 'artichoke', 'celery', 'ginger', 'turmeric', 'horseradish', 'wasabi',
            
            # Unique dairy
            'milk', 'yogurt', 'mozzarella_cheese', 'blue_cheese', 'eggs',
            
            # Unique grains
            'rice', 'wheat', 'couscous', 'polenta', 'gnocchi', 'tortilla', 'flour',
            
            # Unique legumes
            'peanuts',
            
            # Unique condiments
            'vinegar'
        }

    def analyze_dataset_structure(self, input_dir):
        """Analyze the current dataset structure and show category distribution."""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"❌ Input directory '{input_dir}' does not exist!")
            return
        
        print(f"📊 Analyzing dataset structure in '{input_dir}'...")
        
        # Check for standard dataset structure
        train_dir = input_path / "train"
        val_dir = input_path / "val"
        test_dir = input_path / "test"
        
        if not train_dir.exists():
            print(f"❌ Training directory not found: {train_dir}")
            return
        
        # Analyze current categories
        current_categories = [d.name for d in train_dir.iterdir() if d.is_dir()]
        
        print(f"📁 Found {len(current_categories)} categories in training set")
        
        # Count samples per category
        category_counts = {}
        total_samples = 0
        
        for category in current_categories:
            cat_dir = train_dir / category
            if cat_dir.is_dir():
                sample_count = len([f for f in cat_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                category_counts[category] = sample_count
                total_samples += sample_count
        
        print(f"📈 Total training samples: {total_samples}")
        print(f"📊 Average samples per category: {total_samples / len(current_categories):.1f}")
        
        # Show categories that will be consolidated
        consolidation_preview = defaultdict(list)
        unique_categories = set()
        
        for category in current_categories:
            if category in self.item_to_category:
                target_category = self.item_to_category[category]
                consolidation_preview[target_category].append(category)
            elif category in self.unique_items:
                unique_categories.add(category)
            else:
                unique_categories.add(category)  # Unknown items remain unique
        
        print(f"\n🔄 Consolidation Preview:")
        print(f"   Will create {len(consolidation_preview)} consolidated categories")
        print(f"   Will keep {len(unique_categories)} unique categories")
        print(f"   Total final categories: {len(consolidation_preview) + len(unique_categories)}")
        
        # Show some examples
        print(f"\n📋 Examples of consolidations:")
        example_count = 0
        for target_category, source_categories in consolidation_preview.items():
            if example_count < 5:
                total_samples_in_group = sum(category_counts.get(cat, 0) for cat in source_categories)
                print(f"   {target_category}: {source_categories} → {total_samples_in_group} samples")
                example_count += 1
        
        if len(consolidation_preview) > 5:
            print(f"   ... and {len(consolidation_preview) - 5} more consolidated categories")
        
        return {
            'current_categories': current_categories,
            'category_counts': category_counts,
            'consolidation_preview': consolidation_preview,
            'unique_categories': unique_categories
        }

    def consolidate_dataset(self, input_dir, output_dir, dry_run=False):
        """Consolidate the dataset by merging similar categories."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            print(f"❌ Input directory '{input_dir}' does not exist!")
            return False
        
        if dry_run:
            print(f"🔍 DRY RUN: Simulating consolidation from '{input_dir}' to '{output_dir}'")
        else:
            print(f"🔄 Consolidating dataset from '{input_dir}' to '{output_dir}'")
            
            if output_path.exists():
                print(f"⚠️  Output directory already exists. Removing...")
                shutil.rmtree(output_path)
            
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each split (train, val, test)
        splits = ['train', 'val', 'test']
        consolidation_stats = defaultdict(dict)
        
        for split in splits:
            split_input = input_path / split
            split_output = output_path / split
            
            if not split_input.exists():
                print(f"⚠️  Skipping missing split: {split}")
                continue
            
            if not dry_run:
                split_output.mkdir(exist_ok=True)
            
            print(f"\n📁 Processing {split} split...")
            
            # Get all source categories
            source_categories = [d.name for d in split_input.iterdir() if d.is_dir()]
            
            # Track consolidation for this split
            target_category_files = defaultdict(list)
            
            # Process each source category
            for source_category in source_categories:
                source_dir = split_input / source_category
                
                # Get all image files
                image_files = [f for f in source_dir.iterdir() 
                             if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                
                # Determine target category
                if source_category in self.item_to_category:
                    target_category = self.item_to_category[source_category]
                else:
                    target_category = source_category  # Keep as unique
                
                # Add files to target category
                target_category_files[target_category].extend(
                    [(source_category, f) for f in image_files]
                )
            
            # Create consolidated directories and copy files
            for target_category, file_list in target_category_files.items():
                if not dry_run:
                    target_dir = split_output / target_category
                    target_dir.mkdir(exist_ok=True)
                
                file_counter = 0
                source_breakdown = defaultdict(int)
                
                for source_category, file_path in file_list:
                    source_breakdown[source_category] += 1
                    
                    if not dry_run:
                        # Create unique filename to avoid conflicts
                        new_filename = f"{source_category}_{file_counter:04d}{file_path.suffix}"
                        target_file = target_dir / new_filename
                        shutil.copy2(file_path, target_file)
                    
                    file_counter += 1
                
                consolidation_stats[split][target_category] = {
                    'total_files': file_counter,
                    'source_breakdown': dict(source_breakdown)
                }
                
                print(f"   {target_category}: {file_counter} files from {list(source_breakdown.keys())}")
        
        # Create new label mapping
        if not dry_run:
            self.create_consolidated_label_mapping(output_path, consolidation_stats)
        
        # Print summary
        print(f"\n📊 Consolidation Summary:")
        for split, categories in consolidation_stats.items():
            total_files = sum(cat_info['total_files'] for cat_info in categories.values())
            print(f"   {split}: {len(categories)} categories, {total_files} total files")
        
        if not dry_run:
            print(f"\n✅ Dataset consolidation complete!")
            print(f"📁 Consolidated dataset saved to: {output_path}")
        else:
            print(f"\n🔍 Dry run complete. Use --execute to perform actual consolidation.")
        
        return True

    def create_consolidated_label_mapping(self, output_dir, consolidation_stats):
        """Create label mapping for the consolidated dataset."""
        output_path = Path(output_dir)
        
        # Get all unique categories from train split
        train_stats = consolidation_stats.get('train', {})
        categories = sorted(train_stats.keys())
        
        # Create label mapping (category -> index format)
        label_mapping = {category: i for i, category in enumerate(categories)}
        
        # Save label mapping
        label_mapping_file = output_path / "label_mapping.json"
        with open(label_mapping_file, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        # Save labels list
        labels_file = output_path / "labels.txt"
        with open(labels_file, 'w') as f:
            for category in categories:
                f.write(f"{category}\n")
        
        # Create dataset statistics
        dataset_stats = {
            'total_categories': len(categories),
            'splits': consolidation_stats,
            'consolidation_mappings': {
                category: info['source_breakdown'] 
                for category, info in train_stats.items()
            }
        }
        
        stats_file = output_path / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        print(f"📋 Created label mapping with {len(categories)} categories")
        print(f"💾 Saved: {label_mapping_file}")
        print(f"💾 Saved: {labels_file}")
        print(f"💾 Saved: {stats_file}")

def main():
    """Main function for food category consolidation."""
    parser = argparse.ArgumentParser(description='Consolidate food categories to increase samples per class')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input dataset directory (with train/val/test subdirs)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for consolidated dataset')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze current dataset structure without consolidating')
    parser.add_argument('--dry_run', action='store_true',
                       help='Simulate consolidation without actually moving files')
    parser.add_argument('--execute', action='store_true',
                       help='Execute the actual consolidation (required if not dry_run)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.analyze_only and not args.output_dir:
        parser.error("--output_dir is required unless using --analyze_only")
    
    print("🍎 FoodGenius - Category Consolidation Tool")
    print("=" * 50)
    
    # Create consolidator
    consolidator = FoodCategoryConsolidator()
    
    # Analyze dataset structure
    analysis = consolidator.analyze_dataset_structure(args.input_dir)
    
    if analysis is None:
        return
    
    if args.analyze_only:
        print("\n📊 Analysis complete. Use without --analyze_only to consolidate.")
        return
    
    # Perform consolidation
    if args.dry_run or not args.execute:
        success = consolidator.consolidate_dataset(args.input_dir, args.output_dir, dry_run=True)
    else:
        success = consolidator.consolidate_dataset(args.input_dir, args.output_dir, dry_run=False)
    
    if success and not args.dry_run and args.execute:
        print(f"\n🎉 Consolidation complete!")
        print(f"📈 Benefits:")
        print(f"   • Reduced number of categories for better training")
        print(f"   • Increased samples per category")
        print(f"   • Maintained food diversity through smart grouping")
        print(f"   • Ready for training with train_simple_model.py!")

if __name__ == "__main__":
    main()
