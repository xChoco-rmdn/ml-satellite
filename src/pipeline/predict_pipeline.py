import os
import sys
import numpy as np
import tensorflow as tf
from src.logger import logger
from src.exception import CustomException
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        try:
            logger.info("Initializing prediction pipeline")
            self.model = tf.keras.models.load_model('artifacts/final_model.h5')
            self.data_transformer = load_object('artifacts/data_transformer.pkl')
            logger.info("Model and transformer loaded successfully")
        except Exception as e:
            logger.error("Error initializing prediction pipeline")
            raise CustomException(e, sys)
    
    def transform_input(self, input_sequence):
        """
        Transform input sequence for prediction
        Args:
            input_sequence: numpy array of shape (time_steps, height, width)
        Returns:
            transformed_sequence: numpy array ready for model input
        """
        try:
            # Add channel dimension if needed
            if len(input_sequence.shape) == 3:
                input_sequence = np.expand_dims(input_sequence, axis=-1)
            
            # Add batch dimension if needed
            if len(input_sequence.shape) == 4:
                input_sequence = np.expand_dims(input_sequence, axis=0)
            
            return input_sequence
            
        except Exception as e:
            logger.error("Error in transforming input")
            raise CustomException(e, sys)
    
    def inverse_transform_output(self, model_output):
        """
        Transform model output back to original scale
        Args:
            model_output: numpy array of model predictions
        Returns:
            original_scale_output: numpy array in original scale
        """
        try:
            # Remove batch dimension if present
            if len(model_output.shape) == 5:
                model_output = model_output[0]
            
            # Remove channel dimension if present
            if model_output.shape[-1] == 1:
                model_output = model_output[..., 0]
            
            return model_output
            
        except Exception as e:
            logger.error("Error in inverse transforming output")
            raise CustomException(e, sys)
    
    def predict(self, input_sequence):
        """
        Make predictions using the trained model
        Args:
            input_sequence: numpy array of shape (time_steps, height, width)
        Returns:
            predictions: numpy array of predictions
        """
        try:
            # Transform input
            transformed_input = self.transform_input(input_sequence)
            
            # Make prediction
            logger.info("Making prediction...")
            prediction = self.model.predict(transformed_input)
            
            # Inverse transform
            final_prediction = self.inverse_transform_output(prediction)
            
            logger.info(f"Prediction shape: {final_prediction.shape}")
            return final_prediction
            
        except Exception as e:
            logger.error("Error in making prediction")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Load a sample sequence for testing
        sample_input = np.load("data/processed/samples/sample_input_sequence.npy")
        
        # Initialize predictor
        predictor = PredictionPipeline()
        
        # Make prediction
        prediction = predictor.predict(sample_input)
        
        logger.info("Prediction completed successfully!")
        logger.info(f"Input shape: {sample_input.shape}")
        logger.info(f"Output shape: {prediction.shape}")
        
    except Exception as e:
        logger.error(f"Error in prediction test: {str(e)}")
        raise e
