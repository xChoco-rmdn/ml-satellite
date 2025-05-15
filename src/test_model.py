import os
import numpy as np
from src.components.model import CloudNowcastingModel
from src.logger import logger

def test_model():
    try:
        # Load the preprocessed data
        logger.info("Loading preprocessed data...")
        sample_dir = "data/processed/samples"
        X_sample = np.load(f"{sample_dir}/sample_input_sequence.npy")
        y_sample = np.load(f"{sample_dir}/sample_target_sequence.npy")
        
        # Add channel dimension if needed
        if len(X_sample.shape) == 3:
            X_sample = np.expand_dims(X_sample, axis=-1)
        if len(y_sample.shape) == 3:
            y_sample = np.expand_dims(y_sample, axis=-1)
        
        # Add batch dimension for testing
        X_sample = np.expand_dims(X_sample, axis=0)
        y_sample = np.expand_dims(y_sample, axis=0)
        
        logger.info(f"Input shape: {X_sample.shape}")
        logger.info(f"Target shape: {y_sample.shape}")
        
        # Initialize model
        model_builder = CloudNowcastingModel()
        model = model_builder.build_model()
        
        # Print model summary
        model.summary()
        
        # Test forward pass
        logger.info("\nTesting forward pass...")
        test_output = model.predict(X_sample)
        logger.info(f"Output shape: {test_output.shape}")
        
        # Test training for one epoch
        logger.info("\nTesting training for one epoch...")
        history = model_builder.train_model(
            model,
            train_data=(X_sample, y_sample),
            valid_data=None,
            callbacks=None
        )
        
        logger.info("\nModel test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model test: {str(e)}")
        raise e

if __name__ == "__main__":
    test_model() 