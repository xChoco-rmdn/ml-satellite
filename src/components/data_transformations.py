import os
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import numpy as np
from src.logger import logger
from src.exception import CustomException
import sys

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
        
    def create_sequences(self, data):
        """
        Create sequences from the input data for training/testing
        Args:
            data: numpy array of shape (time, height, width)
        Returns:
            X: Input sequences
            y: Target sequences
        """
        try:
            total_samples = len(data)
            sequence_length = self.config.sequence_length
            prediction_horizon = self.config.prediction_horizon
            stride = sequence_length - self.config.overlap_size
            
            # Calculate number of sequences
            n_sequences = (total_samples - sequence_length - prediction_horizon) // stride + 1
            
            X = []
            y = []
            
            for i in range(n_sequences):
                start_idx = i * stride
                end_idx = start_idx + sequence_length
                target_idx = end_idx + prediction_horizon
                
                if target_idx <= total_samples:
                    X.append(data[start_idx:end_idx])
                    y.append(data[end_idx:target_idx])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error("Error in creating sequences")
            raise CustomException(e, sys)
    
    def clean_data(self, data):
        """
        Clean the satellite data by handling missing values and artifacts
        Args:
            data: numpy array of satellite data
        Returns:
            cleaned_data: numpy array of cleaned data
        """
        try:
            # Replace invalid values (like -999) with NaN
            data = np.where(data < 0, np.nan, data)
            
            # Handle missing values by interpolation
            # For each pixel location over time
            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    pixel_timeline = data[:, i, j]
                    mask = np.isnan(pixel_timeline)
                    
                    if np.any(mask) and not np.all(mask):
                        # Get indices of valid values
                        valid_indices = np.where(~mask)[0]
                        # Get valid values
                        valid_values = pixel_timeline[valid_indices]
                        # Create interpolator
                        interp_values = np.interp(
                            np.where(mask)[0],  # indices to interpolate
                            valid_indices,       # known indices
                            valid_values         # known values
                        )
                        # Fill in interpolated values
                        data[mask, i, j] = interp_values
            
            return data
            
        except Exception as e:
            logger.error("Error in cleaning data")
            raise CustomException(e, sys)
    
    def normalize_data(self, data, fit=False):
        """
        Normalize the data using the specified strategy
        Args:
            data: numpy array of shape (time, height, width) or (batch, time, height, width)
            fit: whether to fit the scaler or just transform
        Returns:
            normalized_data: normalized numpy array
        """
        try:
            original_shape = data.shape
            
            if len(original_shape) == 3:
                # Reshape to 2D for StandardScaler (time, height*width)
                reshaped_data = data.reshape(original_shape[0], -1)
            else:
                # Reshape to 2D for StandardScaler (batch*time, height*width)
                reshaped_data = data.reshape(-1, original_shape[2] * original_shape[3])
            
            if fit:
                normalized_data = self.scaler.fit_transform(reshaped_data)
            else:
                normalized_data = self.scaler.transform(reshaped_data)
            
            # Reshape back to original shape
            normalized_data = normalized_data.reshape(original_shape)
            
            return normalized_data
            
        except Exception as e:
            logger.error("Error in normalizing data")
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
    
    def initiate_data_transformation(self, train_array, test_array):
        """
        Main method to transform the data
        Args:
            train_array: training data array
            test_array: testing data array
        Returns:
            Transformed training and testing data
        """
        try:
            logger.info("Started data transformation")
            
            # 1. Clean the data
            logger.info("Cleaning training data")
            train_clean = self.clean_data(train_array)
            logger.info("Cleaning testing data")
            test_clean = self.clean_data(test_array)
            
            # 2. Normalize the data
            logger.info("Normalizing data")
            train_normalized = self.normalize_data(train_clean, fit=True)
            test_normalized = self.normalize_data(test_clean, fit=False)
            
            # 3. Spatial cropping
            logger.info("Applying spatial cropping")
            train_cropped = self.spatial_cropping(train_normalized, center_crop=True)
            test_cropped = self.spatial_cropping(test_normalized, center_crop=True)
            
            # 4. Create sequences
            logger.info("Creating sequences")
            X_train, y_train = self.create_sequences(train_cropped)
            X_test, y_test = self.create_sequences(test_cropped)
            
            logger.info("Data transformation completed")
            logger.info(f"Training sequences shape: {X_train.shape}")
            logger.info(f"Testing sequences shape: {X_test.shape}")
            
            return (X_train, y_train), (X_test, y_test)
            
        except Exception as e:
            logger.error("Error in data transformation")
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
            train_data, test_data
        )
        
        print("Transformation successful!")
        print(f"Training input shape: {X_train.shape}")
        print(f"Training target shape: {y_train.shape}")
        print(f"Testing input shape: {X_test.shape}")
        print(f"Testing target shape: {y_test.shape}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
