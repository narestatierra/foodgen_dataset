#!/usr/bin/env python3
"""
FoodGenius Mobile Deployment Script
==================================

This script handles the complete pipeline from training to mobile deployment
for the FoodGenius large-scale food recognition system.

Features:
- Model validation and optimization
- TensorFlow Lite conversion with quantization
- Mobile performance testing
- Asset preparation for Flutter
- Deployment verification
"""

import os
import sys
import shutil
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import time

try:
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Some features will be disabled.")

class FoodGeniusDeployer:
    """Complete deployment pipeline for FoodGenius mobile app."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.assets_dir = self.project_root / "assets" / "models"
        self.scripts_dir = self.project_root / "scripts"
        
        # Ensure directories exist
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configuration
        self.model_config = {
            'input_size': 224,  # Mobile-optimized size
            'num_classes': 2000,  # Support for large-scale classification
            'quantization': True,
            'optimization': True
        }
    
    def validate_environment(self) -> bool:
        """Validate that all required tools and dependencies are available."""
        print("🔍 Validating deployment environment...")
        
        issues = []
        
        # Check Flutter
        try:
            result = subprocess.run(['flutter', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Flutter available")
            else:
                issues.append("Flutter not found or not working properly")
        except FileNotFoundError:
            issues.append("Flutter not found in PATH")
        
        # Check Python and TensorFlow
        if not TF_AVAILABLE:
            issues.append("TensorFlow not available for model conversion")
        else:
            print(f"✅ TensorFlow {tf.__version__} available")
        
        # Check project structure
        required_files = [
            'pubspec.yaml',
            'lib/main.dart',
            'lib/services/food_classification_service.dart',
            'assets/models/comprehensive_food_labels.txt'
        ]
        
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                issues.append(f"Required file missing: {file_path}")
        
        if issues:
            print("❌ Environment validation failed:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print("✅ Environment validation passed")
        return True
    
    def create_placeholder_model(self) -> bool:
        """Create a placeholder TensorFlow Lite model for development."""
        if not TF_AVAILABLE:
            print("⚠️ TensorFlow not available, skipping model creation")
            return False
        
        print("🔧 Creating placeholder EfficientNet TFLite model...")
        
        try:
            # Create a simple model with EfficientNet-like structure
            inputs = tf.keras.Input(shape=(self.model_config['input_size'], 
                                         self.model_config['input_size'], 3))
            
            # Use EfficientNetB0 as base (lighter for mobile)
            backbone = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_tensor=inputs,
                pooling='avg'
            )
            
            # Add classification head
            x = tf.keras.layers.Dropout(0.2)(backbone.output)
            outputs = tf.keras.layers.Dense(
                self.model_config['num_classes'], 
                activation='softmax'
            )(x)
            
            model = tf.keras.Model(inputs, outputs)
            
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if self.model_config['quantization']:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.lite.constants.INT8]
                
                # Representative dataset for quantization
                def representative_dataset():
                    for _ in range(100):
                        data = np.random.random((1, self.model_config['input_size'], 
                                               self.model_config['input_size'], 3)).astype(np.float32)
                        yield [data]
                
                converter.representative_dataset = representative_dataset
            
            # Convert
            tflite_model = converter.convert()
            
            # Save model
            model_path = self.assets_dir / "efficientnet_food_classifier.tflite"
            with open(model_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✅ Created placeholder model: {model_path}")
            print(f"   Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create placeholder model: {e}")
            return False
    
    def validate_model(self, model_path: Optional[Path] = None) -> bool:
        """Validate TensorFlow Lite model structure and performance."""
        if not TF_AVAILABLE:
            print("⚠️ TensorFlow not available, skipping model validation")
            return True
        
        if model_path is None:
            model_path = self.assets_dir / "efficientnet_food_classifier.tflite"
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        print(f"🔍 Validating model: {model_path}")
        
        try:
            # Load interpreter
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"✅ Model loaded successfully")
            print(f"   Input shape: {input_details[0]['shape']}")
            print(f"   Output shape: {output_details[0]['shape']}")
            print(f"   Input dtype: {input_details[0]['dtype']}")
            print(f"   Output dtype: {output_details[0]['dtype']}")
            
            # Test inference
            input_shape = input_details[0]['shape']
            test_input = np.random.random(input_shape).astype(input_details[0]['dtype'])
            
            # Time inference
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            inference_time = (time.time() - start_time) * 1000
            
            print(f"✅ Test inference completed in {inference_time:.1f}ms")
            print(f"   Output range: [{output_data.min():.4f}, {output_data.max():.4f}]")
            
            # Check if output is normalized (should sum to ~1 for softmax)
            if len(output_data.shape) > 1:
                output_sum = np.sum(output_data[0])
                print(f"   Output sum: {output_sum:.4f} (should be ~1.0 for softmax)")
            
            return True
            
        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            return False
    
    def prepare_assets(self) -> bool:
        """Prepare all assets for Flutter deployment."""
        print("📦 Preparing assets for Flutter deployment...")
        
        # Check comprehensive food labels
        labels_file = self.assets_dir / "comprehensive_food_labels.txt"
        if not labels_file.exists():
            print(f"❌ Comprehensive food labels not found: {labels_file}")
            return False
        
        # Count labels
        with open(labels_file, 'r') as f:
            lines = f.readlines()
            labels = [line.strip() for line in lines 
                     if line.strip() and not line.strip().startswith('#')]
        
        print(f"✅ Found {len(labels)} food labels")
        
        # Update model config if needed
        if len(labels) != self.model_config['num_classes']:
            print(f"📝 Updating model config: {self.model_config['num_classes']} -> {len(labels)} classes")
            self.model_config['num_classes'] = len(labels)
        
        # Check model file
        model_file = self.assets_dir / "efficientnet_food_classifier.tflite"
        if not model_file.exists():
            print("⚠️ TensorFlow Lite model not found, creating placeholder...")
            if not self.create_placeholder_model():
                return False
        
        # Validate pubspec.yaml has correct assets
        pubspec_file = self.project_root / "pubspec.yaml"
        if pubspec_file.exists():
            with open(pubspec_file, 'r') as f:
                pubspec_content = f.read()
                
            required_assets = [
                "assets/models/efficientnet_food_classifier.tflite",
                "assets/models/comprehensive_food_labels.txt"
            ]
            
            missing_assets = []
            for asset in required_assets:
                if asset not in pubspec_content:
                    missing_assets.append(asset)
            
            if missing_assets:
                print("⚠️ Missing assets in pubspec.yaml:")
                for asset in missing_assets:
                    print(f"   - {asset}")
                print("Please add these to the flutter.assets section in pubspec.yaml")
        
        return True
    
    def test_flutter_build(self) -> bool:
        """Test Flutter build process."""
        print("🔨 Testing Flutter build...")
        
        try:
            # Test debug build
            result = subprocess.run(
                ['flutter', 'build', 'ios', '--debug', '--no-pub'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Flutter iOS debug build successful")
                return True
            else:
                print("❌ Flutter build failed:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Flutter build timed out")
            return False
        except Exception as e:
            print(f"❌ Flutter build error: {e}")
            return False
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report."""
        print("📊 Generating deployment report...")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_config': self.model_config,
            'assets': {},
            'performance': {},
            'recommendations': []
        }
        
        # Check assets
        model_file = self.assets_dir / "efficientnet_food_classifier.tflite"
        labels_file = self.assets_dir / "comprehensive_food_labels.txt"
        
        if model_file.exists():
            model_size = model_file.stat().st_size
            report['assets']['model_size_mb'] = model_size / 1024 / 1024
            report['assets']['model_exists'] = True
        else:
            report['assets']['model_exists'] = False
        
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                lines = f.readlines()
                labels = [line.strip() for line in lines 
                         if line.strip() and not line.strip().startswith('#')]
            report['assets']['num_labels'] = len(labels)
            report['assets']['labels_exist'] = True
        else:
            report['assets']['labels_exist'] = False
        
        # Performance estimates
        if TF_AVAILABLE and model_file.exists():
            try:
                interpreter = tf.lite.Interpreter(model_path=str(model_file))
                interpreter.allocate_tensors()
                
                # Estimate inference time (very rough)
                input_details = interpreter.get_input_details()
                test_input = np.random.random(input_details[0]['shape']).astype(input_details[0]['dtype'])
                
                times = []
                for _ in range(10):
                    start = time.time()
                    interpreter.set_tensor(input_details[0]['index'], test_input)
                    interpreter.invoke()
                    times.append((time.time() - start) * 1000)
                
                report['performance']['avg_inference_time_ms'] = np.mean(times)
                report['performance']['min_inference_time_ms'] = np.min(times)
                report['performance']['max_inference_time_ms'] = np.max(times)
                
            except Exception as e:
                report['performance']['error'] = str(e)
        
        # Recommendations
        if report['assets'].get('model_size_mb', 0) > 50:
            report['recommendations'].append("Consider model compression - current size > 50MB")
        
        if report['performance'].get('avg_inference_time_ms', 0) > 500:
            report['recommendations'].append("Consider model optimization - inference time > 500ms")
        
        if not report['assets'].get('model_exists'):
            report['recommendations'].append("Create production model for best accuracy")
        
        # Save report
        report_file = self.project_root / "deployment_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Deployment report saved: {report_file}")
        return report
    
    def run_full_deployment(self) -> bool:
        """Run the complete deployment pipeline."""
        print("🚀 Starting FoodGenius deployment pipeline...\n")
        
        steps = [
            ("Environment Validation", self.validate_environment),
            ("Asset Preparation", self.prepare_assets),
            ("Model Validation", self.validate_model),
            ("Flutter Build Test", self.test_flutter_build),
        ]
        
        for step_name, step_func in steps:
            print(f"{'='*50}")
            print(f"Step: {step_name}")
            print(f"{'='*50}")
            
            if not step_func():
                print(f"❌ Deployment failed at: {step_name}")
                return False
            
            print(f"✅ {step_name} completed successfully\n")
        
        # Generate final report
        print(f"{'='*50}")
        print("Generating Deployment Report")
        print(f"{'='*50}")
        
        report = self.generate_deployment_report()
        
        # Print summary
        print("\n" + "="*50)
        print("🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Model Size: {report['assets'].get('model_size_mb', 0):.1f} MB")
        print(f"Food Categories: {report['assets'].get('num_labels', 0)}")
        
        if 'avg_inference_time_ms' in report['performance']:
            print(f"Avg Inference Time: {report['performance']['avg_inference_time_ms']:.1f}ms")
        
        if report['recommendations']:
            print("\n📋 Recommendations:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
        
        print("\n🎯 Next Steps:")
        print("   1. Test the app on real devices")
        print("   2. Collect user feedback on predictions")
        print("   3. Consider training a production model with real data")
        print("   4. Optimize model size and performance as needed")
        
        return True


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Deploy FoodGenius mobile app")
    parser.add_argument("--project_root", default=".", 
                       help="Root directory of Flutter project")
    parser.add_argument("--create_model", action="store_true",
                       help="Force creation of new placeholder model")
    parser.add_argument("--skip_build", action="store_true",
                       help="Skip Flutter build test")
    
    args = parser.parse_args()
    
    # Create deployer
    deployer = FoodGeniusDeployer(project_root=args.project_root)
    
    # Force model creation if requested
    if args.create_model:
        deployer.create_placeholder_model()
        return
    
    # Run deployment
    success = deployer.run_full_deployment()
    
    if success:
        print("\n✅ Deployment completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Deployment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
