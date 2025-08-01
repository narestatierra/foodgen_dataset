#!/usr/bin/env python3
"""
Advanced Large-Scale Food Recognition Training Configuration
==========================================================

This file contains optimized configurations for training food recognition models
on thousands of food categories with state-of-the-art architectures.
"""

import tensorflow as tf
from typing import Dict, Any, List, Tuple
import math

class ModelArchitectures:
    """State-of-the-art model architectures for large-scale food recognition."""
    
    @staticmethod
    def efficientnet_v2_l(num_classes: int, input_shape: Tuple[int, int, int] = (380, 380, 3)) -> tf.keras.Model:
        """
        EfficientNet-V2-L for high-accuracy food recognition.
        Optimized for 2000+ food categories.
        """
        inputs = tf.keras.Input(shape=input_shape)
        
        # Use EfficientNet-V2-L as backbone
        backbone = tf.keras.applications.EfficientNetV2L(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling='avg'
        )
        
        # Add custom classification head
        x = backbone.output
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Multi-scale feature fusion
        x = tf.keras.layers.Dense(2048, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Final classification layer
        outputs = tf.keras.layers.Dense(
            num_classes, 
            activation='softmax',
            name='predictions'
        )(x)
        
        model = tf.keras.Model(inputs, outputs, name='efficientnet_v2_l_food')
        return model
    
    @staticmethod
    def vision_transformer_large(num_classes: int, input_shape: Tuple[int, int, int] = (384, 384, 3)) -> tf.keras.Model:
        """
        Vision Transformer Large for food recognition.
        Excellent for fine-grained food classification.
        """
        import tensorflow_addons as tfa
        
        # Vision Transformer configuration
        patch_size = 16
        num_patches = (input_shape[0] // patch_size) ** 2
        projection_dim = 1024
        num_heads = 16
        transformer_units = [projection_dim * 2, projection_dim]
        transformer_layers = 24
        
        inputs = tf.keras.Input(shape=input_shape)
        
        # Create patches
        patches = Patches(patch_size)(inputs)
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
        
        # Create multiple layers of the Transformer block
        for _ in range(transformer_layers):
            # Layer normalization 1
            x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            
            # Create a multi-head attention layer
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            
            # Skip connection 1
            x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
            
            # Layer normalization 2
            x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
            
            # MLP
            x3 = tf.keras.layers.Dense(transformer_units[0], activation=tfa.activations.gelu)(x3)
            x3 = tf.keras.layers.Dropout(0.1)(x3)
            x3 = tf.keras.layers.Dense(transformer_units[1])(x3)
            x3 = tf.keras.layers.Dropout(0.1)(x3)
            
            # Skip connection 2
            encoded_patches = tf.keras.layers.Add()([x3, x2])
        
        # Layer normalization and global average pooling
        representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = tf.keras.layers.GlobalAveragePooling1D()(representation)
        representation = tf.keras.layers.Dropout(0.3)(representation)
        
        # Classify outputs
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(representation)
        
        model = tf.keras.Model(inputs, outputs, name='vision_transformer_large_food')
        return model
    
    @staticmethod
    def hybrid_cnn_transformer(num_classes: int, input_shape: Tuple[int, int, int] = (384, 384, 3)) -> tf.keras.Model:
        """
        Hybrid CNN-Transformer architecture combining the best of both worlds.
        Uses CNN for low-level features and Transformer for high-level relationships.
        """
        inputs = tf.keras.Input(shape=input_shape)
        
        # CNN backbone for feature extraction
        backbone = tf.keras.applications.EfficientNetV2B3(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        
        # Get features from multiple scales
        features = backbone.output  # Shape: (batch, H, W, C)
        
        # Reshape for transformer
        batch_size = tf.shape(features)[0]
        h, w, c = features.shape[1], features.shape[2], features.shape[3]
        
        # Flatten spatial dimensions
        features_flat = tf.reshape(features, [batch_size, h * w, c])
        
        # Add positional encoding
        seq_len = h * w
        projection_dim = c
        
        # Transformer layers
        for _ in range(6):  # 6 transformer layers
            # Multi-head attention
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=8, key_dim=projection_dim // 8, dropout=0.1
            )(features_flat, features_flat)
            
            # Skip connection and layer norm
            features_flat = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                features_flat + attn_output
            )
            
            # Feed forward network
            ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(projection_dim * 2, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(projection_dim),
                tf.keras.layers.Dropout(0.1)
            ])
            
            ffn_output = ffn(features_flat)
            features_flat = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                features_flat + ffn_output
            )
        
        # Global average pooling and classification
        pooled = tf.keras.layers.GlobalAveragePooling1D()(features_flat)
        pooled = tf.keras.layers.Dropout(0.3)(pooled)
        
        # Classification head
        x = tf.keras.layers.Dense(1024, activation='relu')(pooled)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name='hybrid_cnn_transformer_food')
        return model


class TrainingConfigurations:
    """Optimized training configurations for different scenarios."""
    
    @staticmethod
    def large_scale_config(num_classes: int) -> Dict[str, Any]:
        """Configuration for training on 2000+ food categories."""
        return {
            'batch_size': 32,  # Adjust based on GPU memory
            'learning_rate': 1e-4,  # Lower LR for stability with many classes
            'epochs': 100,
            'warmup_epochs': 5,
            'weight_decay': 1e-4,
            'label_smoothing': 0.1,  # Helps with many classes
            'mixup_alpha': 0.2,
            'cutmix_alpha': 1.0,
            'optimizer': 'adamw',
            'scheduler': 'cosine_with_warmup',
            'gradient_clip_norm': 1.0,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5,
            'augmentation_strength': 'strong',
            'class_balancing': True,
            'focal_loss_alpha': 0.25,
            'focal_loss_gamma': 2.0
        }
    
    @staticmethod
    def mobile_optimized_config() -> Dict[str, Any]:
        """Configuration optimized for mobile deployment."""
        return {
            'input_size': (224, 224),  # Smaller for mobile
            'quantization': True,
            'pruning': True,
            'distillation': True,
            'teacher_model': 'efficientnet_v2_l',
            'student_model': 'efficientnet_b0',
            'distillation_temperature': 4.0,
            'distillation_alpha': 0.7
        }


class DataAugmentation:
    """Advanced data augmentation for food images."""
    
    @staticmethod
    def get_augmentation_pipeline(config: Dict[str, Any]) -> tf.keras.Sequential:
        """Create augmentation pipeline based on configuration."""
        augmentations = []
        
        # Basic augmentations
        augmentations.extend([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ])
        
        if config.get('augmentation_strength') == 'strong':
            augmentations.extend([
                tf.keras.layers.RandomTranslation(0.1, 0.1),
                # Add more advanced augmentations
            ])
        
        return tf.keras.Sequential(augmentations)


class LossFunctions:
    """Custom loss functions for large-scale food classification."""
    
    @staticmethod
    def focal_loss(alpha: float = 0.25, gamma: float = 2.0):
        """Focal loss for handling class imbalance in large datasets."""
        def focal_loss_fn(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Calculate cross entropy
            ce = -y_true * tf.math.log(y_pred)
            
            # Calculate focal weight
            focal_weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
            
            # Apply focal weight
            focal_loss = focal_weight * ce
            
            return tf.reduce_sum(focal_loss, axis=1)
        
        return focal_loss_fn
    
    @staticmethod
    def label_smoothing_loss(smoothing: float = 0.1):
        """Label smoothing loss for better generalization."""
        def smoothing_loss(y_true, y_pred):
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            smooth_positives = 1.0 - smoothing
            smooth_negatives = smoothing / num_classes
            
            smoothed_labels = y_true * smooth_positives + smooth_negatives
            
            return tf.keras.losses.categorical_crossentropy(
                smoothed_labels, y_pred
            )
        
        return smoothing_loss


class MetricsAndCallbacks:
    """Custom metrics and callbacks for monitoring training."""
    
    @staticmethod
    def get_callbacks(config: Dict[str, Any], model_name: str) -> List[tf.keras.callbacks.Callback]:
        """Get list of callbacks for training."""
        callbacks = []
        
        # Model checkpointing
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                f"checkpoints/{model_name}_best.h5",
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        )
        
        # Early stopping
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.get('early_stopping_patience', 10),
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Learning rate scheduling
        if config.get('scheduler') == 'cosine_with_warmup':
            callbacks.append(CosineDecayWithWarmup(
                warmup_epochs=config.get('warmup_epochs', 5),
                total_epochs=config.get('epochs', 100),
                base_lr=config.get('learning_rate', 1e-4)
            ))
        
        # TensorBoard logging
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=f"logs/{model_name}",
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        )
        
        return callbacks


class CosineDecayWithWarmup(tf.keras.callbacks.Callback):
    """Cosine decay learning rate schedule with warmup."""
    
    def __init__(self, warmup_epochs: int, total_epochs: int, base_lr: float):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay phase
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)


# Helper classes for Vision Transformer
class Patches(tf.keras.layers.Layer):
    """Extract patches from images."""
    
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(tf.keras.layers.Layer):
    """Encode patches with position embedding."""
    
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
