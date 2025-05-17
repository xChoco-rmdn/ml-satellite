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
            
            # Validation and logging
            nan_count_X = np.isnan(X).sum()
            inf_count_X = np.isinf(X).sum()
            nan_count_y = np.isnan(y).sum()
            inf_count_y = np.isinf(y).sum()
            logger.info(f"create_sequences: X nan_count={nan_count_X}, inf_count={inf_count_X}, y nan_count={nan_count_y}, inf_count={inf_count_y}")
            if nan_count_X > 0 or inf_count_X > 0 or nan_count_y > 0 or inf_count_y > 0:
                raise CustomException("NaN or inf values in sequences", sys)
            
            return X, y
            
        except Exception as e:
            logger.error("Error in creating sequences")
            raise CustomException(e, sys)
    
    def clean_data(self, data):
        """
        Enhanced data cleaning with advanced techniques. Handles 3D, 4D, or 5D arrays.
        Ensures data has shape (batch, time, height, width, channels).
        Adds validation and logging after cleaning.
        """
        try:
            cleaned = None
            if data.ndim == 3:
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
                data = ndimage.gaussian_filter1d(data, sigma=1, axis=0)
                z_scores = np.abs((data - np.mean(data)) / np.std(data))
                data = np.where(z_scores > 3, np.nan, data)
                data = self._interpolate_remaining_nans(data)
                cleaned = data
            elif data.ndim == 4:
                if data.shape[0] < 10:
                    cleaned = [self.clean_data(data[..., c]) for c in range(data.shape[-1])]
                    cleaned = np.stack(cleaned, axis=-1)
                else:
                    cleaned = [self.clean_data(data[b]) for b in range(data.shape[0])]
                    cleaned = np.stack(cleaned, axis=0)
            elif data.ndim == 5:
                cleaned = []
                for b in range(data.shape[0]):
                    cleaned_channels = [self.clean_data(data[b, :, :, :, c]) for c in range(data.shape[-1])]
                    cleaned.append(np.stack(cleaned_channels, axis=-1))
                cleaned = np.stack(cleaned, axis=0)
            else:
                raise CustomException(f"Unexpected data shape in clean_data: {data.shape}", sys)
            # Validation and logging
            nan_count = np.isnan(cleaned).sum()
            inf_count = np.isinf(cleaned).sum()
            if nan_count > 0 or inf_count > 0:
                logger.warning(f"Data cleaning issues - shape={cleaned.shape}, nan_count={nan_count}, inf_count={inf_count}, min={np.nanmin(cleaned):.4f}, max={np.nanmax(cleaned):.4f}")
            return cleaned
        except Exception as e:
            logger.error("Error in cleaning data")
            raise CustomException(e, sys)
    
    def _interpolate_remaining_nans(self, data):
        """
        Interpolate any remaining NaN values using spatial and temporal information
        with enhanced debugging and validation
        """
        try:
            nan_mask = np.isnan(data)
            if not np.any(nan_mask):
                return data
                
            nan_count_before = np.isnan(data).sum()
            for t in range(data.shape[0]):
                frame = data[t]
                mask = nan_mask[t]
                if not np.any(mask):
                    continue
                frame_min = np.nanmin(frame)
                frame_max = np.nanmax(frame)
                frame_range = frame_max - frame_min
                if frame_range == 0 or np.isnan(frame_range) or frame_range < 1e-10:
                    frame_uint8 = np.full_like(frame, 128, dtype=np.uint8)
                else:
                    frame_normalized = (frame - frame_min) * (255.0 / frame_range)
                    if np.isnan(frame_normalized).any():
                        frame_normalized = np.nan_to_num(frame_normalized, nan=128.0)
                    frame_normalized = np.clip(frame_normalized, 0, 255)
                    frame_uint8 = frame_normalized.astype(np.uint8)
                mask_uint8 = mask.astype(np.uint8)
                try:
                    filled_frame = cv2.inpaint(frame_uint8, mask_uint8, 3, cv2.INPAINT_TELEA)
                except Exception as e:
                    filled_frame = frame_uint8.copy()
                    filled_frame[mask] = 128
                if frame_range == 0 or np.isnan(frame_range) or frame_range < 1e-10:
                    filled_frame = np.full_like(filled_frame, frame_min if not np.isnan(frame_min) else 0, dtype=float)
                else:
                    filled_frame = (filled_frame.astype(float) * frame_range / 255.0) + frame_min
                data[t] = np.where(mask, filled_frame, frame)
            
            nan_count_after = np.isnan(data).sum()
            inf_count_after = np.isinf(data).sum()
            if nan_count_after > 0 or inf_count_after > 0:
                logger.warning(f"Interpolation issues - nan_count: {nan_count_before} -> {nan_count_after}, inf_count: {inf_count_after}")
                data = np.nan_to_num(data, nan=0.0)
            return data
        except Exception as e:
            logger.error(f"Error in interpolating remaining NaN values: {str(e)}")
            raise CustomException(e, sys)
    
    def normalize_data(self, data):
        """Normalize data using per-pixel z-score normalization with robust handling for all-NaN and zero-std pixels."""
        try:
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            if nan_count > 0 or inf_count > 0:
                logger.warning(f"Normalization input issues - nan_count={nan_count}, inf_count={inf_count}")
                
            global_mean = np.nanmean(data)
            if len(data.shape) == 3:
                all_nan_mask = np.all(np.isnan(data), axis=0, keepdims=True)
                data = np.where(all_nan_mask, global_mean, data)
                pixel_means = np.nanmean(data, axis=0, keepdims=True)
                data = np.where(np.isnan(data), pixel_means, data)
            else:
                all_nan_mask = np.all(np.isnan(data), axis=1, keepdims=True)
                data = np.where(all_nan_mask, global_mean, data)
                pixel_means = np.nanmean(data, axis=1, keepdims=True)
                data = np.where(np.isnan(data), pixel_means, data)
                
            if len(data.shape) == 3:
                mean = np.mean(data, axis=0, keepdims=True)
                std = np.std(data, axis=0, keepdims=True)
            else:
                mean = np.mean(data, axis=1, keepdims=True)
                std = np.std(data, axis=1, keepdims=True)
                
            std = np.where(std < 1e-6, 1.0, std)
            
            # log statistics if there are potential issues
            if std.min() < 1e-6 or std.max() > 100:
                logger.warning(f"Normalization stats - mean range=({mean.min():.4f}, {mean.max():.4f}), std range=({std.min():.4f}, {std.max():.4f})")
                
            normalized_data = (data - mean) / std
            clip_threshold = 3.0
            normalized_data = np.clip(normalized_data, -clip_threshold, clip_threshold)
            
            nan_count = np.isnan(normalized_data).sum()
            inf_count = np.isinf(normalized_data).sum()
            if nan_count > 0 or inf_count > 0:
                logger.warning(f"Normalization output issues - nan_count={nan_count}, inf_count={inf_count}")
                normalized_data = np.nan_to_num(normalized_data, nan=0.0, posinf=0.0, neginf=0.0)
                
            if np.isnan(normalized_data).any() or np.isinf(normalized_data).any():
                raise CustomException("Normalization failed: NaN or inf values remain in the data", sys)
                
            return normalized_data
        except Exception as e:
            logger.error(f"Error in per-pixel normalization: {str(e)}")
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
            
            augmented_data = np.array(augmented_data)
            augmented_labels = np.array(augmented_labels)
            
            # Validation and logging
            nan_count_data = np.isnan(augmented_data).sum()
            inf_count_data = np.isinf(augmented_data).sum()
            nan_count_labels = np.isnan(augmented_labels).sum()
            inf_count_labels = np.isinf(augmented_labels).sum()
            logger.info(f"augment_data: data nan_count={nan_count_data}, inf_count={inf_count_data}, labels nan_count={nan_count_labels}, inf_count={inf_count_labels}")
            if nan_count_data > 0 or inf_count_data > 0 or nan_count_labels > 0 or inf_count_labels > 0:
                raise CustomException("NaN or inf values in augmented data", sys)
            
            return augmented_data, augmented_labels
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
