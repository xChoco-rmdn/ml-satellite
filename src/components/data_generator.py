import tensorflow as tf
import numpy as np
from src.logger import logger
from src.exception import CustomException
import sys

class SatelliteDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=4, prefetch_factor=2, augment=True):
        """
        Initialize the data generator
        Args:
            X: Input sequences (batch, time_steps, height, width, channels)
            y: Target sequences (batch, time_steps, height, width, channels)
            batch_size: Batch size for training
            prefetch_factor: Number of batches to prefetch
            augment: Whether to apply data augmentation
        """
        try:
            self.X = X
            self.y = y
            self.batch_size = batch_size
            self.prefetch_factor = prefetch_factor
            self.augment = augment
            self.n_samples = len(X)
            
            # Calculate number of batches
            self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
            
            # Initialize augmentation parameters
            self.rotation_range = 20
            self.width_shift_range = 0.1
            self.height_shift_range = 0.1
            self.brightness_range = [0.8, 1.2]
            
            logger.info(f"Initialized SatelliteDataGenerator with {self.n_samples} samples")
            logger.info(f"Batch size: {self.batch_size}, Number of batches: {self.n_batches}")
            
        except Exception as e:
            logger.error("Error initializing SatelliteDataGenerator")
            raise CustomException(e, sys)
    
    def __len__(self):
        """Return number of batches"""
        return self.n_batches
    
    def __getitem__(self, idx):
        """Get a batch of data"""
        try:
            # Calculate batch indices
            start_idx = idx * self.batch_size
            end_idx = min((idx + 1) * self.batch_size, self.n_samples)
            
            # Get batch data
            batch_X = self.X[start_idx:end_idx]
            batch_y = self.y[start_idx:end_idx]
            
            # Apply augmentation if enabled
            if self.augment:
                batch_X, batch_y = self._augment_batch(batch_X, batch_y)
            
            return batch_X, batch_y
            
        except Exception as e:
            logger.error(f"Error getting batch {idx}")
            raise CustomException(e, sys)
    
    def _augment_batch(self, X, y):
        """Apply data augmentation to a batch"""
        try:
            augmented_X = []
            augmented_y = []
            
            for i in range(len(X)):
                # Get single sample
                x = X[i]
                y_sample = y[i]
                
                # Random rotation
                angle = np.random.uniform(-self.rotation_range, self.rotation_range)
                x = tf.image.rot90(x, k=int(angle/90))
                y_sample = tf.image.rot90(y_sample, k=int(angle/90))
                
                # Random shifts
                shift_x = int(np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[1])
                shift_y = int(np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[0])
                x = tf.image.translate(x, [shift_x, shift_y])
                y_sample = tf.image.translate(y_sample, [shift_x, shift_y])
                
                # Random brightness
                brightness = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
                x = x * brightness
                
                augmented_X.append(x)
                augmented_y.append(y_sample)
            
            return np.array(augmented_X), np.array(augmented_y)
            
        except Exception as e:
            logger.error("Error in data augmentation")
            raise CustomException(e, sys)
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        # Shuffle the data
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices] 