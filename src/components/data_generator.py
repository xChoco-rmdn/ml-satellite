import tensorflow as tf
import numpy as np
from src.logger import logger
from src.exception import CustomException
import sys
import scipy.ndimage

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
            
            # Convert data to float32 to reduce memory usage
            self.X = self.X.astype(np.float32)
            self.y = self.y.astype(np.float32)
            
            # Log initialization info once
            logger.info(f"Data generator initialized - samples: {self.n_samples}, batch_size: {self.batch_size}")
            
        except Exception as e:
            logger.error("Error initializing data generator")
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
            batch_X = self.X[start_idx:end_idx].copy()  # Create a copy to avoid memory issues
            batch_y = self.y[start_idx:end_idx].copy()
            
            # Apply augmentation if enabled
            if self.augment:
                batch_X, batch_y = self._augment_batch(batch_X, batch_y)
            
            return batch_X, batch_y
            
        except Exception as e:
            logger.error(f"Error in batch {idx}")
            raise CustomException(e, sys)
    
    def _augment_batch(self, X, y):
        """Apply data augmentation to a batch"""
        try:
            augmented_X = []
            augmented_y = []
            
            for i in range(len(X)):
                # Original data
                augmented_X.append(X[i])
                augmented_y.append(y[i])
                
                # Random rotation
                angle = np.random.uniform(-self.rotation_range, self.rotation_range)
                rotated_X = scipy.ndimage.rotate(X[i], angle, axes=(1, 2), reshape=False)
                rotated_y = scipy.ndimage.rotate(y[i], angle, axes=(1, 2), reshape=False)
                augmented_X.append(rotated_X)
                augmented_y.append(rotated_y)
                
                # Random flip
                if np.random.random() > 0.5:
                    flipped_X = np.flip(X[i], axis=1)
                    flipped_y = np.flip(y[i], axis=1)
                    augmented_X.append(flipped_X)
                    augmented_y.append(flipped_y)
                
                # Random brightness adjustment
                brightness_factor = np.random.uniform(*self.brightness_range)
                brightened_X = X[i] * brightness_factor
                augmented_X.append(brightened_X)
                augmented_y.append(y[i])
            
            # Convert to numpy arrays and ensure float32
            augmented_X = np.array(augmented_X, dtype=np.float32)
            augmented_y = np.array(augmented_y, dtype=np.float32)
            
            return augmented_X, augmented_y
            
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