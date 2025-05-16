import os
import numpy as np
import tensorflow as tf
from src.components.model import CloudNowcastingModel
from src.components.data_transformations import DataTransformation
from src.logger import logger
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from src.exception import CustomException

def generate_test_data(batch_size=2, time_steps=6, height=256, width=256):
    """Generate synthetic test data with realistic cloud patterns."""
    try:
        # Generate base cloud patterns
        x = np.linspace(-5, 5, width)
        y = np.linspace(-5, 5, height)
        X, Y = np.meshgrid(x, y)
        
        # Create multiple time steps with evolving patterns
        data = []
        for t in range(time_steps):
            # Generate evolving cloud pattern
            pattern = np.sin(X + t*0.5) * np.cos(Y + t*0.3)
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
            
            # Add some noise
            noise = np.random.normal(0, 0.1, (height, width))
            pattern = pattern + noise
            
            # Clip to valid range
            pattern = np.clip(pattern, 0, 1)
            
            # Add channel dimension
            pattern = pattern[..., np.newaxis]
            data.append(pattern)
        
        # Stack time steps
        data = np.stack(data, axis=0)
        
        # Create batch
        data = np.stack([data] * batch_size, axis=0)
        
        return data
    except Exception as e:
        raise CustomException(e, sys)

def test_model_architecture():
    """Test the model architecture."""
    try:
        print("Testing enhanced model architecture...")
        model_builder = CloudNowcastingModel()
        model = model_builder.build_model()
        
        # Generate test input
        test_input = generate_test_data()
        test_output = model.predict(test_input)
        
        print(f"Test input shape: {test_input.shape}")
        print(f"Test output shape: {test_output.shape}")
        
        return model
    except Exception as e:
        raise CustomException(e, sys)

def test_data_transformation():
    """Test the data transformation pipeline."""
    try:
        print("Testing data transformation pipeline...")
        transformer = DataTransformation()
        
        # Generate test data
        data = generate_test_data(batch_size=4)
        
        # Clean data
        cleaned_data = transformer.clean_data(data)
        print(f"Cleaned data shape: {cleaned_data.shape}")
        
        # Normalize data
        normalized_data = transformer.normalize_data(cleaned_data)
        print(f"Normalized data mean: {np.mean(normalized_data)}, std: {np.std(normalized_data)}")
        
        # Create sequences
        X, y = transformer.create_sequences(normalized_data)
        print(f"Sequence shapes - X: {X.shape}, y: {y.shape}")
        
        # Augment data
        X_aug, y_aug = transformer.augment_data(X, y)
        print(f"Augmented data shape: {X_aug.shape}")
        
        return transformer
    except Exception as e:
        raise CustomException(e, sys)

def test_training_pipeline(model_builder, transformer):
    """Test the training pipeline."""
    try:
        print("Testing training pipeline...")
        
        # Generate training data with more time steps
        train_data = generate_test_data(batch_size=16, time_steps=16)
        
        # Transform data
        X_train, y_train, X_test, y_test = transformer.initiate_data_transformation(train_data)
        
        print(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
        print(f"Test data shapes - X: {X_test.shape}, y: {y_test.shape}")
        
        # Build model
        model = model_builder.build_model()
        
        # Train model
        history = model_builder.train_model(
            model,
            (X_train, y_train),
            (X_test, y_test)
        )
        
        return history
    except Exception as e:
        raise CustomException(e, sys)

def test_prediction_pipeline(model_builder):
    """Test the prediction pipeline."""
    try:
        print("Testing prediction pipeline...")
        
        # Generate test data
        test_data = generate_test_data(batch_size=2)
        
        # Make predictions
        predictions = model_builder.predict(test_data)
        
        print(f"Prediction shape: {predictions.shape}")
        
        return predictions
    except Exception as e:
        raise CustomException(e, sys)

def main():
    """Run all tests."""
    try:
        # Test model architecture
        model_builder = CloudNowcastingModel()
        model = test_model_architecture()
        
        # Test data transformation
        transformer = test_data_transformation()
        
        # Test training pipeline
        history = test_training_pipeline(model_builder, transformer)
        
        # Test prediction pipeline
        predictions = test_prediction_pipeline(model_builder)
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Error in main test execution: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 