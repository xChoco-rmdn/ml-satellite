import os
import numpy as np
from src.components.data_transformations import DataTransformation
from src.logger import logger

def test_transformation():
    try:
        # Load the data
        logger.info("Loading training and test data...")
        train_data = np.load("data/train/train_data_20250401_0000_to_20250425_0100.npy")
        test_data = np.load("data/test/test_data_20250425_0110_to_20250501_0000.npy")
        
        logger.info(f"Original training data shape: {train_data.shape}")
        logger.info(f"Original test data shape: {test_data.shape}")
        
        # Initialize transformation
        transform = DataTransformation()
        
        # Transform training data
        logger.info("Starting data transformation for training data...")
        X_train, y_train, _, _ = transform.initiate_data_transformation(train_data)
        
        # Transform test data
        logger.info("Starting data transformation for test data...")
        X_test, y_test, _, _ = transform.initiate_data_transformation(test_data)
        
        # Print results
        logger.info("\nTransformation Results:")
        logger.info(f"Training input shape (batch, time_steps, height, width): {X_train.shape}")
        logger.info(f"Training target shape (batch, prediction_steps, height, width): {y_train.shape}")
        logger.info(f"Test input shape (batch, time_steps, height, width): {X_test.shape}")
        logger.info(f"Test target shape (batch, prediction_steps, height, width): {y_test.shape}")
        
        # Basic validation
        logger.info("\nValidation Checks:")
        logger.info(f"Number of training sequences: {len(X_train)}")
        logger.info(f"Number of test sequences: {len(X_test)}")
        logger.info(f"Input sequence length: {X_train.shape[1]}")
        logger.info(f"Prediction sequence length: {y_train.shape[1]}")
        logger.info(f"Spatial dimensions after cropping: {X_train.shape[2]}x{X_train.shape[3]}")
        
        # Check for NaN values
        logger.info("\nData Quality Checks:")
        logger.info(f"Training input contains NaN: {np.isnan(X_train).any()}")
        logger.info(f"Training target contains NaN: {np.isnan(y_train).any()}")
        logger.info(f"Test input contains NaN: {np.isnan(X_test).any()}")
        logger.info(f"Test target contains NaN: {np.isnan(y_test).any()}")
        
        # Save sample sequences for visualization if needed
        sample_dir = "data/processed/samples"
        os.makedirs(sample_dir, exist_ok=True)
        
        np.save(f"{sample_dir}/sample_input_sequence.npy", X_train[0])
        np.save(f"{sample_dir}/sample_target_sequence.npy", y_train[0])
        
        logger.info("\nTest completed successfully!")
        logger.info(f"Sample sequences saved in {sample_dir}")
        
    except Exception as e:
        logger.error(f"Error during transformation test: {str(e)}")
        raise e

if __name__ == "__main__":
    test_transformation() 