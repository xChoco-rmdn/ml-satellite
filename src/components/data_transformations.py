import os
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import numpy as np
from src.logger import logger
from src.exception import CustomException
import sys
import tensorflow as tf
import cv2
from scipy import ndimage
import scipy.ndimage

@dataclass
class DataTransformationConfig:
    sequence_length: int = 6  # Number of time steps in each sequence
    prediction_horizon: int = 6  # Number of time steps to predict
    overlap_size: int = 2  # Number of overlapping frames between sequences
    standardization_strategy: str = 'global'  # 'global' or 'sequence'
    spatial_crop_size: tuple = (256, 256)  # Size to crop images to (height, width)
    
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.scaler = StandardScaler()
        self.normalization_stats = None
        
    def create_sequences(self, data, sequence_length=6):
        """
        Create input-output sequences for training
        Args:
            data: Input data of shape (batch, time_steps, height, width, channels)
            sequence_length: Length of input sequence
        Returns:
            X: Input sequences
            y: Target sequences
        """
        try:
            # Get number of samples and time steps
            n_samples = data.shape[0]
            n_timesteps = data.shape[1]
            
            # Calculate number of sequences
            n_sequences = n_samples * (n_timesteps - sequence_length)
            
            # Initialize arrays
            X = np.zeros((n_sequences, sequence_length, data.shape[2], data.shape[3], data.shape[4]))
            y = np.zeros((n_sequences, sequence_length, data.shape[2], data.shape[3], data.shape[4]))
            
            # Create sequences
            seq_idx = 0
            for i in range(n_samples):
                for j in range(n_timesteps - sequence_length):
                    X[seq_idx] = data[i, j:j+sequence_length]
                    y[seq_idx] = data[i, j+1:j+sequence_length+1]
                    seq_idx += 1
            
            logger.info(f"Created {n_sequences} sequences")
            logger.info(f"Input shape: {X.shape}, Target shape: {y.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error("Error in creating sequences")
            raise CustomException(e, sys)
    
    def clean_data(self, data):
        """
        Enhanced data cleaning with advanced techniques. Handles 3D, 4D, or 5D arrays.
        """
        try:
            # If data is 5D (batch, time, height, width, channel)
            if data.ndim == 5:
                for b in range(data.shape[0]):
                    for c in range(data.shape[-1]):
                        data[b, :, :, :, c] = self.clean_data(data[b, :, :, :, c])
                return data
            # If data is 4D (batch, time, height, width)
            if data.ndim == 4:
                for b in range(data.shape[0]):
                    data[b] = self.clean_data(data[b])
                return data
            # Now data is 3D (time, height, width)
            data = np.where(data < 0, np.nan, data)
            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    pixel_timeline = data[:, i, j]
                    mask = np.isnan(pixel_timeline)
                    if np.any(mask) and not np.all(mask):
                        valid_indices = np.where(~mask)[0]
                        valid_values = pixel_timeline[valid_indices]
                        interp_values = np.interp(
                            np.where(mask)[0],
                            valid_indices,
                            valid_values,
                            left=valid_values[0],
                            right=valid_values[-1]
                        )
                        data[mask, i, j] = interp_values
            # Temporal smoothing
            data = ndimage.gaussian_filter1d(data, sigma=1, axis=0)
            # Remove outliers using z-score
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            data = np.where(z_scores > 3, np.nan, data)
            # Final interpolation for any remaining NaN values
            data = self._interpolate_remaining_nans(data)
            return data
        except Exception as e:
            logger.error("Error in cleaning data")
            raise CustomException(e, sys)
    
    def _interpolate_remaining_nans(self, data):
        """
        Interpolate any remaining NaN values using spatial and temporal information
        """
        try:
            # Create a mask of NaN values
            nan_mask = np.isnan(data)
            
            if not np.any(nan_mask):
                return data
            
            # For each time step
            for t in range(data.shape[0]):
                # Get the current frame
                frame = data[t]
                mask = nan_mask[t]
                
                if not np.any(mask):
                    continue
                
                # Check if frame has any variation
                frame_range = frame.max() - frame.min()
                if frame_range == 0:
                    # If all values are the same, use that value for inpainting
                    frame_uint8 = np.full_like(frame, 128, dtype=np.uint8)
                else:
                    # Normalize to [0, 255] range
                    frame_uint8 = ((frame - frame.min()) * (255.0 / frame_range)).astype(np.uint8)
                
                mask_uint8 = mask.astype(np.uint8)
                
                # Apply inpainting
                filled_frame = cv2.inpaint(frame_uint8, mask_uint8, 3, cv2.INPAINT_TELEA)
                
                # Convert back to original scale
                if frame_range == 0:
                    filled_frame = np.full_like(filled_frame, frame[0, 0], dtype=float)
                else:
                    filled_frame = (filled_frame.astype(float) * frame_range / 255.0) + frame.min()
                
                # Update the data
                data[t] = np.where(mask, filled_frame, frame)
            
            return data
            
        except Exception as e:
            logger.error("Error in interpolating remaining NaN values")
            raise CustomException(e, sys)
    
    def normalize_data(self, data):
        """Normalize data using z-score normalization."""
        try:
            # Handle NaN values before normalization
            data = np.nan_to_num(data, nan=0.0)
            
            # Calculate mean and std along spatial dimensions
            mean = np.mean(data, axis=(1, 2), keepdims=True)
            std = np.std(data, axis=(1, 2), keepdims=True)
            
            # Avoid division by zero
            std = np.where(std == 0, 1.0, std)
            
            # Normalize
            normalized_data = (data - mean) / std
            
            return normalized_data
        except Exception as e:
            raise CustomException(e, sys)
    
    def augment_data(self, data, labels):
        """Apply data augmentation techniques."""
        try:
            augmented_data = []
            augmented_labels = []
            
            for i in range(len(data)):
                # Original data
                augmented_data.append(data[i])
                augmented_labels.append(labels[i])
                
                # Random rotation
                angle = np.random.uniform(-15, 15)
                rotated_data = scipy.ndimage.rotate(data[i], angle, axes=(1, 2), reshape=False)
                rotated_labels = scipy.ndimage.rotate(labels[i], angle, axes=(1, 2), reshape=False)
                augmented_data.append(rotated_data)
                augmented_labels.append(rotated_labels)
                
                # Random flip
                if np.random.random() > 0.5:
                    flipped_data = np.flip(data[i], axis=1)
                    flipped_labels = np.flip(labels[i], axis=1)
                    augmented_data.append(flipped_data)
                    augmented_labels.append(flipped_labels)
                
                # Random brightness adjustment
                brightness_factor = np.random.uniform(0.8, 1.2)
                brightened_data = data[i] * brightness_factor
                augmented_data.append(brightened_data)
                augmented_labels.append(labels[i])
            
            return np.array(augmented_data), np.array(augmented_labels)
        except Exception as e:
            raise CustomException(e, sys)
    
    def spatial_cropping(self, data, center_crop=True):
        """
        Crop the spatial dimensions of the data
        Args:
            data: numpy array of shape (time, height, width) or (batch, time, height, width)
            center_crop: whether to crop from center or randomly
        Returns:
            cropped_data: cropped numpy array
        """
        try:
            target_height, target_width = self.config.spatial_crop_size
            
            if len(data.shape) == 3:
                time, height, width = data.shape
                if center_crop:
                    start_h = (height - target_height) // 2
                    start_w = (width - target_width) // 2
                else:
                    start_h = np.random.randint(0, height - target_height + 1)
                    start_w = np.random.randint(0, width - target_width + 1)
                
                cropped_data = data[:, 
                                  start_h:start_h + target_height,
                                  start_w:start_w + target_width]
            else:
                batch, time, height, width = data.shape
                if center_crop:
                    start_h = (height - target_height) // 2
                    start_w = (width - target_width) // 2
                else:
                    start_h = np.random.randint(0, height - target_height + 1)
                    start_w = np.random.randint(0, width - target_width + 1)
                
                cropped_data = data[:, :,
                                  start_h:start_h + target_height,
                                  start_w:start_w + target_width]
            
            return cropped_data
            
        except Exception as e:
            logger.error("Error in spatial cropping")
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, data_path):
        """Complete data transformation pipeline."""
        try:
            # Load and clean data
            data = self.clean_data(data_path)
            
            # Normalize data
            normalized_data = self.normalize_data(data)
            
            # Create sequences
            X, y = self.create_sequences(normalized_data)
            
            # Split into train and test sets
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Augment training data
            X_train_aug, y_train_aug = self.augment_data(X_train, y_train)
            
            return X_train_aug, y_train_aug, X_test, y_test
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Example usage
    try:
        # Load some sample data
        train_data = np.load("data/train/sample_train_data.npy")
        test_data = np.load("data/test/sample_test_data.npy")
        
        # Initialize transformation
        transform = DataTransformation()
        
        # Transform data
        (X_train, y_train), (X_test, y_test) = transform.initiate_data_transformation(
            train_data
        )
        
        print("Transformation successful!")
        print(f"Training input shape: {X_train.shape}")
        print(f"Training target shape: {y_train.shape}")
        print(f"Testing input shape: {X_test.shape}")
        print(f"Testing target shape: {y_test.shape}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
