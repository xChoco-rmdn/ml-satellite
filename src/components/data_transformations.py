import os
import sys
import numpy as np
from src.logger import logger
from src.exception import CustomException
from src.utils import save_numpy_array_data, load_numpy_array_data
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import cv2

@dataclass
class DataTransformationConfig:
    preprocessed_data_path: str = os.path.join('data', 'processed', 'preprocessed')
    window_size: int = 4  # 4-hour prediction window
    sequence_length: int = 6  # Number of previous frames to use for prediction

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        os.makedirs(self.data_transformation_config.preprocessed_data_path, exist_ok=True)
        
    def get_data_transformer(self):
        """
        Returns the data transformer object
        """
        try:
            scaler = StandardScaler()
            return scaler
            
        except Exception as e:
            raise CustomException(e, sys)
            
    def create_sequences(self, data):
        """
        Create sequences for training from the data
        """
        try:
            sequences = []
            targets = []
            
            for i in range(len(data) - self.data_transformation_config.sequence_length - self.data_transformation_config.window_size):
                # Get sequence of frames
                seq = data[i:i + self.data_transformation_config.sequence_length]
                # Get target frame
                target = data[i + self.data_transformation_config.sequence_length + self.data_transformation_config.window_size]
                
                sequences.append(seq)
                targets.append(target)
                
            return np.array(sequences), np.array(targets)
            
        except Exception as e:
            raise CustomException(e, sys)
            
    def normalize_data(self, data, scaler=None):
        """
        Normalize the satellite data
        """
        try:
            original_shape = data.shape
            flattened_data = data.reshape(-1, 1)
            
            if scaler is None:
                scaler = StandardScaler()
                normalized_data = scaler.fit_transform(flattened_data)
            else:
                normalized_data = scaler.transform(flattened_data)
                
            normalized_data = normalized_data.reshape(original_shape)
            
            return normalized_data, scaler
            
        except Exception as e:
            raise CustomException(e, sys)
            
    def apply_cloud_mask(self, data, threshold=235):
        """
        Apply cloud mask based on brightness temperature threshold
        """
        try:
            # Create binary mask where 1 indicates cloud presence
            cloud_mask = (data < threshold).astype(np.float32)
            return cloud_mask
            
        except Exception as e:
            raise CustomException(e, sys)
            
    def initiate_data_transformation(self, data_path):
        """
        Initiate the data transformation process
        """
        try:
            logger.info("Started data transformation")
            
            # Load the data
            data = load_numpy_array_data(data_path)
            
            # Apply cloud mask
            cloud_mask = self.apply_cloud_mask(data)
            
            # Normalize the data
            normalized_data, scaler = self.normalize_data(data)
            
            # Create sequences for training
            X, y = self.create_sequences(normalized_data)
            
            # Save the transformed data
            transformed_data_path = os.path.join(
                self.data_transformation_config.preprocessed_data_path,
                'transformed_data.npz'
            )
            
            np.savez(
                transformed_data_path,
                X=X,
                y=y,
                cloud_mask=cloud_mask
            )
            
            logger.info(f"Data transformation completed. Data saved to {transformed_data_path}")
            
            return transformed_data_path, scaler
            
        except Exception as e:
            logger.error("Error in data transformation")
            raise CustomException(e, sys) 