#!/usr/bin/env python3
"""
Simple Food Classification Training Script
==========================================

This script trains a basic food classification model using the collected dataset.

For better training performance with small datasets, consider consolidating similar 
food categories using the consolidate_food_categories.py script before training.
This groups similar foods (e.g., different pasta types, citrus fruits) to increase 
samples per category.

Usage Examples:
1. Basic training:
   python scripts/train_simple_model.py --dataset_dir ml_experiments/datasets/my_dataset --epochs 20

2. Training with consolidated categories:
   python scripts/consolidate_food_categories.py --input_dir ml_experiments/datasets/my_dataset --output_dir ml_experiments/datasets/my_dataset_consolidated --execute
   python scripts/train_simple_model.py --dataset_dir ml_experiments/datasets/my_dataset_consolidated --epochs 20

3. Hyperparameter tuning:
   python scripts/train_simple_model.py --dataset_dir ml_experiments/datasets/my_dataset --tune --epochs 20
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Ensure we can run from either root directory or scripts directory
if Path.cwd().name == 'scripts':
    os.chdir('..')  # Change to parent directory (project root)

class SimpleFoodTrainer:
    def tune_hyperparameters(self, epochs=5, max_trials=10):
        """Simple grid search for hyperparameter tuning."""
        print("\n🔍 Starting hyperparameter tuning (grid search)...")
        # Define search space
        learning_rates = [0.01,0.001, 0.0005, 0.0001]
        batch_sizes = [8, 16, 32]
        dropouts = [0.2, 0.3, 0.4]
        best_acc = 0
        best_config = None
        best_history = None
        best_model = None
        img_size = self.supported_architectures[self.architecture]['input_size']
        train_gen, val_gen, test_gen = self.create_data_generators(img_size, batch_size=8)
        trial = 0
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for dropout in dropouts:
                    if trial >= max_trials:
                        break
                    print(f"\nTrial {trial+1}: lr={lr}, batch_size={batch_size}, dropout={dropout}")
                    # Create model
                    model, base_model = self.create_model(img_size)
                    # Adjust dropout in classification head
                    for layer in model.layers:
                        if isinstance(layer, tf.keras.layers.Dropout):
                            layer.rate = dropout
                    # for MAC M1, use keras.optimizers.legacy.Adam otherwise keras.optimizers.Adam
                    
                    model.compile(
                        optimizer=keras.optimizers.legacy.Adam(  # Prova SGD invece di Adam!
                            learning_rate=lr
                        ),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    # Train briefly
                    history = model.fit(
                        train_gen,
                        epochs=epochs,
                        validation_data=val_gen,
                        verbose=0
                    )
                    val_acc = max(history.history['val_accuracy'])
                    print(f"  -> val_accuracy: {val_acc:.4f}")
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_config = {'learning_rate': lr, 'batch_size': batch_size, 'dropout': dropout}
                        best_history = history
                        best_model = model
                    trial += 1
        print(f"\n🏆 Best hyperparameters: {best_config} (val_accuracy={best_acc:.4f})")
        return best_config, best_model, best_history
    """Food classification model trainer with multiple architecture support."""
    
    def __init__(self, dataset_dir: str = "small_training_test", model_dir: str = "models", 
                 architecture: str = "mobilenet_v2"):
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.architecture = architecture.lower()
        
        self.train_dir = self.dataset_dir / "train"
        self.val_dir = self.dataset_dir / "val"  # Changed back to "val"
        self.test_dir = self.dataset_dir / "test"
        
        # Supported architectures with their optimal input sizes and configurations
        self.supported_architectures = {
            'mobilenet_v2': {'input_size': (224, 224), 'efficient': True, 'mobile_friendly': True},
            'efficientnet_v2_b0': {'input_size': (224, 224), 'efficient': True, 'mobile_friendly': True},
            'efficientnet_v2_b1': {'input_size': (240, 240), 'efficient': True, 'mobile_friendly': True},
            'efficientnet_v2_b2': {'input_size': (260, 260), 'efficient': True, 'mobile_friendly': True},
            'efficientnet_v2_b3': {'input_size': (300, 300), 'efficient': True, 'mobile_friendly': False},
            'efficientnet_v2_s': {'input_size': (384, 384), 'efficient': True, 'mobile_friendly': False},
            'resnet50': {'input_size': (224, 224), 'efficient': False, 'mobile_friendly': True},
            'resnet101': {'input_size': (224, 224), 'efficient': False, 'mobile_friendly': False},
            'densenet121': {'input_size': (224, 224), 'efficient': False, 'mobile_friendly': True},
            'xception': {'input_size': (299, 299), 'efficient': False, 'mobile_friendly': False},
            'inception_v3': {'input_size': (299, 299), 'efficient': False, 'mobile_friendly': False},
        }
        
        if self.architecture not in self.supported_architectures:
            available = ', '.join(self.supported_architectures.keys())
            raise ValueError(f"Unsupported architecture '{self.architecture}'. "
                           f"Available: {available}")
        
        # Load label mapping
        label_file = self.dataset_dir / "label_mapping.json"
        if label_file.exists():
            with open(label_file, 'r') as f:
                self.label_mapping = json.load(f)
            self.num_classes = len(self.label_mapping)
        else:
            print("No label mapping found, will auto-detect classes")
            self.label_mapping = None
            self.num_classes = None
        
        print(f"🏗️ Architecture: {self.architecture}")
        print(f"📐 Input size: {self.supported_architectures[self.architecture]['input_size']}")
        print(f"📱 Mobile friendly: {self.supported_architectures[self.architecture]['mobile_friendly']}")
    
    def create_data_generators(self, img_size=None, batch_size=8):
        """Create data generators for training with architecture-specific input size and strong augmentation for small datasets."""
        # Use architecture-specific input size if not provided
        if img_size is None:
            img_size = self.supported_architectures[self.architecture]['input_size']

        # More aggressive data augmentation for small datasets
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=30.0,
            fill_mode='nearest'
        )

        # Only rescaling for validation/test
        val_datagen = ImageDataGenerator(rescale=1./255)

        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )

        val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        test_generator = val_datagen.flow_from_directory(
            self.test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        self.num_classes = train_generator.num_classes
        self.class_names = list(train_generator.class_indices.keys())

        print(f"📊 Dataset Info:")
        print(f"  Classes: {self.num_classes}")
        print(f"  Class names: {self.class_names}")
        print(f"  Train samples: {train_generator.samples}")
        print(f"  Validation samples: {val_generator.samples}")
        print(f"  Test samples: {test_generator.samples}")

        return train_generator, val_generator, test_generator
    
    def create_model(self, img_size=None):
        """Create a model based on selected architecture using transfer learning."""
        
        # Use architecture-specific input size if not provided
        if img_size is None:
            img_size = self.supported_architectures[self.architecture]['input_size']
        
        # Clear any cached models to avoid weight conflicts
        self.clear_tf_cache()
        
        print(f"🧠 Loading {self.architecture} with ImageNet pretrained weights...")
        
        try:
            # Create base model based on selected architecture
            print(f"🧠 Loading {self.architecture} with ImageNet pretrained weights...")
            base_model = self._create_base_model(img_size)
            
            print(f"✅ Successfully loaded {self.architecture} with ImageNet weights")
            model_name = self.architecture
            
        except Exception as e:
            print(f"❌ {self.architecture} creation failed: {e}")
            raise RuntimeError(f"Could not create {self.architecture} model architecture.")
        
        # Freeze the base model initially for transfer learning
        base_model.trainable = False
        
        print(f"🧠 Using {model_name} with {len(base_model.layers)} frozen layers")
        
        # Create the complete model with architecture-specific configuration
        model = self._build_classification_head(base_model, img_size)
        
        # Compile with transfer learning optimized settings
        learning_rate = self._get_optimal_learning_rate()
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']  # Removed top_5_accuracy for compatibility
        )
        
        return model, base_model
    
    def _create_base_model(self, img_size):
        """Create base model for the selected architecture."""
        input_shape = (*img_size, 3)
        
        if self.architecture == 'mobilenet_v2':
            return tf.keras.applications.MobileNetV2(
                weights='imagenet', include_top=False, input_shape=input_shape)
        
        elif self.architecture == 'efficientnet_v2_b0':
            return tf.keras.applications.EfficientNetV2B0(
                weights='imagenet', include_top=False, input_shape=input_shape)
        
        elif self.architecture == 'efficientnet_v2_b1':
            return tf.keras.applications.EfficientNetV2B1(
                weights='imagenet', include_top=False, input_shape=input_shape)
        
        elif self.architecture == 'efficientnet_v2_b2':
            return tf.keras.applications.EfficientNetV2B2(
                weights='imagenet', include_top=False, input_shape=input_shape)
        
        elif self.architecture == 'efficientnet_v2_b3':
            return tf.keras.applications.EfficientNetV2B3(
                weights='imagenet', include_top=False, input_shape=input_shape)
        
        elif self.architecture == 'efficientnet_v2_s':
            return tf.keras.applications.EfficientNetV2S(
                weights='imagenet', include_top=False, input_shape=input_shape)
        
        elif self.architecture == 'resnet50':
            return tf.keras.applications.ResNet50(
                weights='imagenet', include_top=False, input_shape=input_shape)
        
        elif self.architecture == 'resnet101':
            return tf.keras.applications.ResNet101(
                weights='imagenet', include_top=False, input_shape=input_shape)
        
        elif self.architecture == 'densenet121':
            return tf.keras.applications.DenseNet121(
                weights='imagenet', include_top=False, input_shape=input_shape)
        
        elif self.architecture == 'xception':
            return tf.keras.applications.Xception(
                weights='imagenet', include_top=False, input_shape=input_shape)
        
        elif self.architecture == 'inception_v3':
            return tf.keras.applications.InceptionV3(
                weights='imagenet', include_top=False, input_shape=input_shape)
        
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
    
    def _create_base_model_no_weights(self, img_size):
        """Create base model without pre-trained weights as fallback."""
        input_shape = (*img_size, 3)
        
        if self.architecture == 'efficientnet_v2_b0':
            return tf.keras.applications.EfficientNetV2B0(
                weights=None, include_top=False, input_shape=input_shape)
        
        elif self.architecture == 'efficientnet_v2_b1':
            return tf.keras.applications.EfficientNetV2B1(
                weights=None, include_top=False, input_shape=input_shape)
        
        elif self.architecture == 'efficientnet_v2_b2':
            return tf.keras.applications.EfficientNetV2B2(
                weights=None, include_top=False, input_shape=input_shape)
        
        elif self.architecture == 'efficientnet_v2_b3':
            return tf.keras.applications.EfficientNetV2B3(
                weights=None, include_top=False, input_shape=input_shape)
        
        elif self.architecture == 'efficientnet_v2_s':
            return tf.keras.applications.EfficientNetV2S(
                weights=None, include_top=False, input_shape=input_shape)
        
        else:
            raise ValueError(f"No fallback available for architecture: {self.architecture}")
    
    def _build_classification_head(self, base_model, img_size):
        """Build classification head optimized for the selected architecture."""
        
        # Architecture-specific head configurations
        head_configs = {
            'mobilenet_v2': {'dense_units': [128], 'dropout_rates': [0.2, 0.3]},
            'efficientnet_v2_b0': {'dense_units': [256], 'dropout_rates': [0.3, 0.2]},
            'efficientnet_v2_b1': {'dense_units': [256], 'dropout_rates': [0.3, 0.2]},
            'efficientnet_v2_b2': {'dense_units': [512], 'dropout_rates': [0.3, 0.2]},
            'efficientnet_v2_b3': {'dense_units': [512], 'dropout_rates': [0.4, 0.3]},
            'efficientnet_v2_s': {'dense_units': [1024], 'dropout_rates': [0.4, 0.3]},
            'resnet50': {'dense_units': [512], 'dropout_rates': [0.3, 0.2]},
            'resnet101': {'dense_units': [1024], 'dropout_rates': [0.4, 0.3]},
            'densenet121': {'dense_units': [512], 'dropout_rates': [0.3, 0.2]},
            'xception': {'dense_units': [1024], 'dropout_rates': [0.4, 0.3]},
            'inception_v3': {'dense_units': [1024], 'dropout_rates': [0.4, 0.3]},
        }
        
        config = head_configs[self.architecture]
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(config['dropout_rates'][0])
        ])
        
        # Add dense layers based on configuration
        for units in config['dense_units']:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(config['dropout_rates'][1]))
        
        # Final classification layer
        model.add(layers.Dense(self.num_classes, activation='softmax', name='predictions'))
        
        return model
    
    def _get_optimal_learning_rate(self):
        """Get optimal learning rate for the selected architecture."""
        # Architecture-specific learning rates optimized for transfer learning
        lr_configs = {
            'mobilenet_v2': 0.01,
            'efficientnet_v2_b0': 0.001,
            'efficientnet_v2_b1': 0.0008,
            'efficientnet_v2_b2': 0.0005,
            'efficientnet_v2_b3': 0.0003,
            'efficientnet_v2_s': 0.0003,
            'resnet50': 0.001,
            'resnet101': 0.0005,
            'densenet121': 0.001,
            'xception': 0.0005,
            'inception_v3': 0.0005,
        }
        return lr_configs[self.architecture]
    
    def fine_tune_model(self, model, base_model, train_gen, val_gen, epochs=5):
        """Fine-tune the model by unfreezing some layers of the base model."""
        
        print(f"\n🔧 Fine-tuning: Unfreezing top layers of {self.architecture}...")
        
        # Unfreeze the top layers of the base model for fine-tuning
        base_model.trainable = True
        
        # Architecture-specific fine-tuning strategies
        fine_tune_configs = {
            'mobilenet_v2': {'unfreeze_from': 0.8, 'lr_divisor': 10},
            'efficientnet_v2_b0': {'unfreeze_from': 0.8, 'lr_divisor': 10},
            'efficientnet_v2_b1': {'unfreeze_from': 0.75, 'lr_divisor': 10},
            'efficientnet_v2_b2': {'unfreeze_from': 0.75, 'lr_divisor': 10},
            'efficientnet_v2_b3': {'unfreeze_from': 0.7, 'lr_divisor': 20},
            'efficientnet_v2_s': {'unfreeze_from': 0.7, 'lr_divisor': 20},
            'resnet50': {'unfreeze_from': 0.8, 'lr_divisor': 10},
            'resnet101': {'unfreeze_from': 0.7, 'lr_divisor': 20},
            'densenet121': {'unfreeze_from': 0.8, 'lr_divisor': 10},
            'xception': {'unfreeze_from': 0.7, 'lr_divisor': 20},
            'inception_v3': {'unfreeze_from': 0.7, 'lr_divisor': 20},
        }
        
        config = fine_tune_configs[self.architecture]
        
        # Fine-tune from this layer onwards
        fine_tune_at = int(len(base_model.layers) * config['unfreeze_from'])
        
        # Freeze all the layers before fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        unfrozen_layers = sum(1 for layer in base_model.layers[fine_tune_at:] if layer.trainable)
        print(f"🔓 Unfrozen {unfrozen_layers} layers (from layer {fine_tune_at})")
        
        # Recompile with a much lower learning rate for fine-tuning
        base_lr = self._get_optimal_learning_rate()
        fine_tune_lr = base_lr / config['lr_divisor']
        
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(  # Prova SGD invece di Adam!
                learning_rate=fine_tune_lr
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']  # Removed top_5_accuracy for compatibility
        )
        
        print(f"📉 Fine-tuning learning rate: {fine_tune_lr}")
        
        # Callbacks for fine-tuning
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=3, 
                restore_best_weights=True,
                monitor='val_accuracy',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,          # dimezza il learning rate
                patience=5,          # aspetta 5 epoche senza miglioramenti
                verbose=1,           # stampa quando riduce
                mode='min',
                min_lr=1e-7,        # non scendere sotto questo
                cooldown=2          # aspetta 2 epoche dopo ogni riduzione
            )
        ]
        
        print(f"🎯 Fine-tuning for {epochs} epochs...")
        
        # Fine-tune the model
        fine_tune_history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return fine_tune_history

    def train_model(self, epochs: int = 20, batch_size: int = 32):
        """Train the food classification model using transfer learning."""
        
        print(f"🏋️  Starting {self.architecture.upper()} transfer learning for {epochs} epochs...")
        
        # Clear TensorFlow cache to prevent weight loading issues
        self.clear_tf_cache()
        
        # Get architecture-specific input size
        img_size = self.supported_architectures[self.architecture]['input_size']
        
        # Create data generators
        train_gen, val_gen, test_gen = self.create_data_generators(img_size, batch_size)
        
        # Create model with current architecture base
        model, base_model = self.create_model(img_size)
        
        print(f"\n🏗️  {self.architecture.upper()} Model Architecture:")
        model.summary()
        
        # Split training into two phases for better transfer learning
        initial_epochs = min(max(1, epochs // 2), epochs - 1)  # First phase: frozen base (at least 1, but leave at least 1 for fine-tuning)
        if epochs <= 2:
            initial_epochs = epochs  # For very short training, use all epochs for phase 1
            fine_tune_epochs = 0
        else:
            fine_tune_epochs = epochs - initial_epochs  # Second phase: fine-tuning
        
        print(f"\n📋 Training Plan:")
        print(f"  Phase 1: {initial_epochs} epochs with frozen {self.architecture.upper()}")
        print(f"  Phase 2: {fine_tune_epochs} epochs with fine-tuning")
        
        # Phase 1: Train with frozen base model
        print(f"\n🎯 PHASE 1: Training classifier head ({initial_epochs} epochs)")
        
        callbacks_phase1 = [
            keras.callbacks.EarlyStopping(
                patience=9, 
                restore_best_weights=True,
                monitor='val_accuracy',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,          # dimezza il learning rate
                patience=5,          # aspetta 5 epoche senza miglioramenti
                verbose=1,           # stampa quando riduce
                mode='min',
                min_lr=1e-7,        # non scendere sotto questo
                cooldown=2          # aspetta 2 epoche dopo ogni riduzione
            )
        ]
        
        # Train the model with frozen base
        history_phase1 = model.fit(
            train_gen,
            epochs=initial_epochs,
            validation_data=val_gen,
            callbacks=callbacks_phase1,
            verbose=1
        )
        
        # Phase 2: Fine-tune the model
        print(f"\n🎯 PHASE 2: Fine-tuning {self.architecture.upper()} ({fine_tune_epochs} epochs)")
        
        if fine_tune_epochs > 0:
            history_phase2 = self.fine_tune_model(
                model, base_model, train_gen, val_gen, fine_tune_epochs
            )
            
            # Combine histories for plotting
            combined_history = {
                'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
                'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
                'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
                'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
            }
            
            # Create a mock history object for plotting
            class CombinedHistory:
                def __init__(self, history_dict):
                    self.history = history_dict
            
            history = CombinedHistory(combined_history)
        else:
            history = history_phase1
        
        # Evaluate on test set
        print(f"\n🧪 Evaluating final model on test set...")
        test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
        print(f"📊 Final test accuracy: {test_accuracy:.3f}")
        
        # Print model summary
        trainable_params = sum(p.numel() for p in model.trainable_weights if hasattr(p, 'numel'))
        total_params = sum(p.numel() for p in model.weights if hasattr(p, 'numel'))
        print(f"📈 Model statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Save the model with architecture name
        model_path = self.model_dir / f"food_classifier_{self.architecture}.keras"  # Use .keras format
        model.save(model_path)
        print(f"💾 {self.architecture} model saved to: {model_path}")
        
        # Convert to TensorFlow Lite
        self.convert_to_tflite(model)
        
        # Save training history
        history_path = self.model_dir / f"training_history_{self.architecture}.json"
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            hist_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
            json.dump(hist_dict, f, indent=2)
        
        return model, history, test_accuracy
    
    def plot_training_history(self, history):
        """Plot training history for transfer learning."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title(f'{self.architecture} Transfer Learning - Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add phase separator if two-phase training
        if len(history.history['accuracy']) > 5:  # Assume two-phase if more than 5 epochs
            phase1_end = len(history.history['accuracy']) // 2
            ax1.axvline(x=phase1_end - 0.5, color='red', linestyle='--', alpha=0.7, 
                       label='Fine-tuning starts')
            ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title(f'{self.architecture} Transfer Learning - Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add phase separator if two-phase training
        if len(history.history['loss']) > 5:
            phase1_end = len(history.history['loss']) // 2
            ax2.axvline(x=phase1_end - 0.5, color='red', linestyle='--', alpha=0.7,
                       label='Fine-tuning starts')
            ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.model_dir / f"training_plot_{self.architecture}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training plot saved to: {plot_path}")
        
        return fig
    
    def convert_to_tflite(self, model):
        """Convert Keras model to TensorFlow Lite format for mobile deployment."""
        
        print(f"\n🔄 Converting {self.architecture} model to TensorFlow Lite...")
        
        try:
            # Create TensorFlow Lite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Basic optimization (quantization)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Convert the model
            tflite_model = converter.convert()
            
            # Save the TFLite model with architecture name
            tflite_path = self.model_dir / f"food_classifier_{self.architecture}.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Get model size info
            original_size = os.path.getsize(self.model_dir / f"food_classifier_{self.architecture}.keras") / (1024 * 1024)
            tflite_size = len(tflite_model) / (1024 * 1024)
            compression_ratio = original_size / tflite_size
            
            print(f"✅ TensorFlow Lite model saved to: {tflite_path}")
            print(f"📊 Model size comparison:")
            print(f"   Original (.keras): {original_size:.2f} MB")
            print(f"   TFLite (.tflite): {tflite_size:.2f} MB")
            print(f"   Compression ratio: {compression_ratio:.1f}x")
            
            # Test the TFLite model
            self.test_tflite_model(tflite_path)
            
            return tflite_path
            
        except Exception as e:
            print(f"❌ Failed to convert to TensorFlow Lite: {e}")
            return None
    
    def test_tflite_model(self, tflite_path):
        """Test the TensorFlow Lite model to ensure it works correctly."""
        
        print(f"\n🧪 Testing TensorFlow Lite model...")
        
        try:
            # Load the TFLite model
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            
            # Get input and output tensors
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"✅ TFLite model loaded successfully!")
            print(f"📊 Model info:")
            print(f"   Input shape: {input_details[0]['shape']}")
            print(f"   Input dtype: {input_details[0]['dtype']}")
            print(f"   Output shape: {output_details[0]['shape']}")
            print(f"   Output dtype: {output_details[0]['dtype']}")
            
            # Create a dummy input to test inference
            input_shape = input_details[0]['shape']
            dummy_input = np.random.random(input_shape).astype(input_details[0]['dtype'])
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            print(f"✅ Test inference successful!")
            print(f"   Output shape: {output_data.shape}")
            print(f"   Output sample: {output_data[0][:5]}...")  # Show first 5 values
            
            return True
            
        except Exception as e:
            print(f"❌ TFLite model test failed: {e}")
            return False

    def clear_tf_cache(self):
        """Clear TensorFlow cache to resolve weight loading issues."""
        print("🧹 Clearing TensorFlow cache and corrupted weights...")
        tf.keras.backend.clear_session()
        
        # Try to force garbage collection
        import gc
        gc.collect()
        
        # Clear specific EfficientNet cache files
        try:
            import shutil
            cache_dir = os.path.expanduser('~/.keras')
            if os.path.exists(cache_dir):
                # Look for EfficientNet model files
                models_dir = os.path.join(cache_dir, 'models')
                if os.path.exists(models_dir):
                    for file in os.listdir(models_dir):
                        if 'efficientnet' in file.lower():
                            file_path = os.path.join(models_dir, file)
                            try:
                                os.remove(file_path)
                                print(f"🗑️ Removed corrupted cached file: {file}")
                            except:
                                pass
                    
                    # Also remove any partial downloads or corrupted files
                    for file in os.listdir(models_dir):
                        if file.endswith('.tmp') or 'efficientnet' in file.lower():
                            file_path = os.path.join(models_dir, file)
                            try:
                                os.remove(file_path)
                                print(f"🗑️ Removed temporary file: {file}")
                            except:
                                pass
                                
        except Exception as e:
            print(f"Note: Could not clear cache completely: {e}")
            
        print("✅ Cache cleared successfully")
    
    # ...existing code...

def main():
    """Main function to run the multi-architecture food classification training."""
    parser = argparse.ArgumentParser(description='Train food classification model with multiple architecture support')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning before training')
    parser.add_argument('--dataset_dir', type=str, default='small_training_test',
                      help='Path to the dataset directory')
    parser.add_argument('--model_dir', type=str, default='models',
                      help='Directory to save trained models')
    parser.add_argument('--architecture', type=str, default='mobilenet_v2',
                      choices=['mobilenet_v2', 'efficientnet_v2_b0', 'efficientnet_v2_b1', 'efficientnet_v2_b2', 
                              'efficientnet_v2_b3', 'efficientnet_v2_s',
                              'resnet50', 'resnet101', 'densenet121', 'xception', 'inception_v3'],
                      help='Model architecture to use for transfer learning')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Total number of training epochs (split between base training and fine-tuning)')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training (will be adjusted for architecture if needed)')
    parser.add_argument('--img_size', type=int, nargs=2, default=None,
                      help='Image size (height width) - will use architecture optimal size if not specified')
    parser.add_argument('--list_architectures', action='store_true',
                      help='List all available architectures with their specifications')
    
    args = parser.parse_args()
    
    # List architectures if requested
    if args.list_architectures:
        print("\n�️ Available Architectures for Food Classification:")
        print("=" * 60)
        
        architectures_info = {
            'mobilenet_v2': {'size': '224x224', 'mobile': '✅', 'speed': 'Fast', 'accuracy': 'Good'},
            'efficientnet_v2_b0': {'size': '224x224', 'mobile': '✅', 'speed': 'Fast', 'accuracy': 'Better'},
            'efficientnet_v2_b1': {'size': '240x240', 'mobile': '✅', 'speed': 'Fast', 'accuracy': 'Better'},
            'efficientnet_v2_b2': {'size': '260x260', 'mobile': '✅', 'speed': 'Medium', 'accuracy': 'Better'},
            'efficientnet_v2_b3': {'size': '300x300', 'mobile': '⚠️', 'speed': 'Medium', 'accuracy': 'Best'},
            'efficientnet_v2_s': {'size': '384x384', 'mobile': '❌', 'speed': 'Slow', 'accuracy': 'Best'},
            'resnet50': {'size': '224x224', 'mobile': '✅', 'speed': 'Medium', 'accuracy': 'Good'},
            'resnet101': {'size': '224x224', 'mobile': '❌', 'speed': 'Slow', 'accuracy': 'Better'},
            'densenet121': {'size': '224x224', 'mobile': '✅', 'speed': 'Medium', 'accuracy': 'Good'},
            'xception': {'size': '299x299', 'mobile': '❌', 'speed': 'Slow', 'accuracy': 'Best'},
            'inception_v3': {'size': '299x299', 'mobile': '❌', 'speed': 'Slow', 'accuracy': 'Best'},
        }
        
        for arch, info in architectures_info.items():
            print(f"{arch:20} | {info['size']:8} | Mobile: {info['mobile']:2} | Speed: {info['speed']:6} | Accuracy: {info['accuracy']}")
        
        print("\n📱 Mobile friendly models are recommended for mobile deployment")
        print("🎯 For best results with food images, try: efficientnet_v2_b1 or efficientnet_v2_b2")
        print("⚡ For fastest training/inference: mobilenet_v2 or efficientnet_v2_b0")
        print("🏆 For best accuracy (but slower): efficientnet_v2_b3 or xception")
        return
    
    print(f"🍎 FoodGenius - Multi-Architecture Transfer Learning Pipeline")
    print("=" * 60)
    print(f"🧠 Architecture: {args.architecture}")
    print(f"🎯 Transfer learning approach: freeze -> fine-tune")
    print("=" * 60)
    
    # Check if dataset exists
    dataset_dir = args.dataset_dir
    if not Path(dataset_dir).exists():
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        print("Please run the collection script first.")
        return
    
    # Create trainer with selected architecture
    trainer = SimpleFoodTrainer(dataset_dir, args.model_dir, args.architecture)
    img_size = tuple(args.img_size) if args.img_size else None
    batch_size = args.batch_size
    if args.architecture in ['efficientnet_v2_s', 'xception', 'inception_v3', 'resnet101']:
        batch_size = min(batch_size, 8)
        print(f"📉 Reduced batch size to {batch_size} for {args.architecture}")

    # Hyperparameter tuning step
    if args.tune:
        best_config, _, _ = trainer.tune_hyperparameters(epochs=3, max_trials=12)
        # Use best hyperparameters for final training
        print(f"\n� Training final model with best hyperparameters: {best_config}")
        batch_size = best_config['batch_size']
        # Optionally, you could also set learning rate and dropout in create_model

    # Train the model with selected architecture
    model, history, test_accuracy = trainer.train_model(
        epochs=args.epochs,
        batch_size=batch_size
    )

    # Plot training history
    trainer.plot_training_history(history)

    print(f"\n🎉 {args.architecture} Transfer Learning Complete!")
    print(f"📊 Final test accuracy: {test_accuracy:.3f}")
    print(f"🧠 Model: {args.architecture} + Custom Classification Head")
    print(f"📁 Results saved in: {args.model_dir}/")
    print(f"📱 TensorFlow Lite model optimized for mobile deployment!")
    print(f"🚀 Ready for Flutter integration in FoodGenius app!")

    arch_config = trainer.supported_architectures[args.architecture]
    if not arch_config['mobile_friendly']:
        print(f"⚠️  Note: {args.architecture} may be too large for mobile deployment")
        print(f"💡 Consider using a mobile-friendly architecture for production")

    return test_accuracy

if __name__ == "__main__":
    main()
