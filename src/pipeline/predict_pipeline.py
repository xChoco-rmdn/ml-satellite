import os
import sys
import numpy as np
from src.logger import logger
from src.exception import CustomException
from src.utils import load_object
from tensorflow.keras.models import load_model
from src.components.data_ingestion import DataIngestion
from src.components.data_transformations import DataTransformation
from datetime import datetime, timedelta

class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join('models', 'saved_models', 'final_model.h5')
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        
    def load_model(self):
        """
        Load the trained model
        """
        try:
            if not os.path.exists(self.model_path):
                raise CustomException(f"Model file not found at {self.model_path}", sys)
                
            model = load_model(self.model_path)
            return model
            
        except Exception as e:
            raise CustomException(e, sys)
            
    def predict(self, current_time=None, region=None):
        """
        Make predictions for the next 4 hours
        """
        try:
            logger.info("Starting prediction pipeline")
            
            # Set default time if not provided
            if current_time is None:
                current_time = datetime.now()
                
            # Get the last 6 hours of data for sequence input
            start_time = current_time - timedelta(hours=6)
            
            # Get satellite data
            satellite_files = self.data_ingestion.get_himawari_data(
                start_time=start_time,
                end_time=current_time,
                region=region
            )
            
            if not satellite_files:
                raise CustomException("No satellite data files found for prediction", sys)
                
            # Process the data
            raw_data_path = self.data_ingestion.initiate_data_ingestion(satellite_files)
            
            # Transform the data
            data = np.load(raw_data_path)
            normalized_data, _ = self.data_transformation.normalize_data(data)
            
            # Prepare sequence for prediction
            sequence = normalized_data[-6:]  # Last 6 frames
            sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
            
            # Load model and make prediction
            model = self.load_model()
            prediction = model.predict(sequence)
            
            # Inverse transform the prediction if needed
            # This would depend on how the data was normalized during training
            
            logger.info("Prediction completed successfully")
            
            return prediction
            
        except Exception as e:
            logger.error("Error in prediction pipeline")
            raise CustomException(e, sys)
            
    def get_prediction_timestamps(self, current_time=None):
        """
        Get timestamps for the prediction horizon
        """
        if current_time is None:
            current_time = datetime.now()
            
        prediction_times = [
            current_time + timedelta(hours=i)
            for i in range(1, 5)  # 4-hour prediction horizon
        ]
        
        return prediction_times 