import os
import sys
from src.logger import logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformations import DataTransformation
from src.components.model_trainer import ModelTrainer
from datetime import datetime, timedelta

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        
    def initiate_training(self, start_time=None, end_time=None, region=None):
        """
        Initiate the complete training pipeline
        """
        try:
            logger.info("Starting training pipeline")
            
            # Set default time range if not provided
            if start_time is None:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=30)  # Default to last 30 days
                
            # Step 1: Data Ingestion
            logger.info("Starting data ingestion")
            satellite_files = self.data_ingestion.get_himawari_data(
                start_time=start_time,
                end_time=end_time,
                region=region
            )
            
            if not satellite_files:
                raise CustomException("No satellite data files found", sys)
                
            raw_data_path = self.data_ingestion.initiate_data_ingestion(satellite_files)
            logger.info("Data ingestion completed")
            
            # Step 2: Data Transformation
            logger.info("Starting data transformation")
            transformed_data_path, _ = self.data_transformation.initiate_data_transformation(raw_data_path)
            logger.info("Data transformation completed")
            
            # Step 3: Model Training
            logger.info("Starting model training")
            model, metrics = self.model_trainer.initiate_model_trainer(transformed_data_path)
            logger.info("Model training completed")
            
            return model, metrics
            
        except Exception as e:
            logger.error("Error in training pipeline")
            raise CustomException(e, sys) 