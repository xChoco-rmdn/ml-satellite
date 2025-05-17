"""
Satellite Image Preprocessing Script

This script handles all preprocessing steps for satellite imagery data including:
- Data loading and validation
- Image normalization
- Data augmentation
- Feature extraction
- Data splitting

Usage:
    python 1_preprocessing.py

The script will:
1. Load raw satellite data
2. Process and transform the data
3. Save processed data to data/processed/
4. Save train/test splits to data/train/ and data/test/
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.components.data_ingestion import DataIngestion
from src.components.data_transformations import DataTransformation
from src.logger import logger
from src.exception import CustomException

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data/raw',
        'data/processed',
        'data/train',
        'data/test',
        'artifacts',
        'logs'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def reshape_data_for_sequences_xy_nonoverlap(data, sequence_length=6):
    """Create X, y pairs for sequence prediction using non-overlapping windows."""
    n_samples = (len(data) - sequence_length) // sequence_length
    X = np.zeros((n_samples, sequence_length, data.shape[1], data.shape[2], 1))
    y = np.zeros((n_samples, sequence_length, data.shape[1], data.shape[2], 1))
    for i in range(n_samples):
        start = i * sequence_length
        end = start + sequence_length
        X[i] = data[start:end, :, :, np.newaxis]
        y[i] = data[start+1:end+1, :, :, np.newaxis]
    return X, y

def center_crop(data, target_height=256, target_width=256):
    """Crop the center of each frame in the data to the target size."""
    cropped = []
    for frame in data:
        h, w = frame.shape
        start_h = (h - target_height) // 2
        start_w = (w - target_width) // 2
        cropped.append(frame[start_h:start_h+target_height, start_w:start_w+target_width])
    return np.stack(cropped)

def process_data_in_batches(data, data_transformation, batch_size=10):
    """Process data in smaller batches to prevent memory issues"""
    n_samples = len(data)
    processed_batches = []
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = data[i:end_idx]
        logger.info(f"Processing batch {i//batch_size + 1} of {(n_samples + batch_size - 1)//batch_size}")
        try:
            # Clean and normalize batch
            cleaned_batch = data_transformation.clean_data(batch)
            normalized_batch = data_transformation.normalize_data(cleaned_batch)
            processed_batches.append(normalized_batch)
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
            continue
    
    return np.concatenate(processed_batches, axis=0)

def main():
    try:
        logger.info("Starting preprocessing pipeline")
        setup_directories()

        # Initialize components
        data_ingestion = DataIngestion()
        data_transformation = DataTransformation()

        # Get list of satellite files
        raw_data_path = os.path.join('data', 'raw')
        satellite_files = [os.path.join(raw_data_path, f) for f in os.listdir(raw_data_path) if f.endswith('.nc')]
        if not satellite_files:
            raise CustomException("No satellite files found in data/raw directory", sys)
        logger.info(f"Found {len(satellite_files)} satellite files")

        # Process and ingest data
        logger.info("Starting data ingestion...")
        ingestion_result = data_ingestion.initiate_data_ingestion(satellite_files)

        # Load the processed data
        train_data = np.load(ingestion_result['train_file_path'])
        test_data = np.load(ingestion_result['test_file_path'])
        logger.info(f"Loaded training data shape: {train_data.shape}")
        logger.info(f"Loaded test data shape: {test_data.shape}")

        # Process data in batches
        logger.info("Processing data in batches...")
        train_data = process_data_in_batches(train_data, data_transformation)
        test_data = process_data_in_batches(test_data, data_transformation)

        # Crop data
        logger.info("Cropping data...")
        train_data = center_crop(train_data, 256, 256)
        test_data = center_crop(test_data, 256, 256)

        # Create sequences
        logger.info("Creating sequences...")
        sequence_length = data_transformation.config.sequence_length
        
        X_train, y_train = reshape_data_for_sequences_xy_nonoverlap(train_data, sequence_length)
        X_test, y_test = reshape_data_for_sequences_xy_nonoverlap(test_data, sequence_length)

        # Save transformed data
        logger.info("Saving transformed data...")
        np.save('data/processed/X_train.npy', X_train)
        np.save('data/processed/y_train.npy', y_train)
        np.save('data/processed/X_test.npy', X_test)
        np.save('data/processed/y_test.npy', y_test)

        logger.info("Preprocessing completed successfully!")
        logger.info(f"Transformed data shapes:")
        logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    except Exception as e:
        logger.error("Error in preprocessing pipeline")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main() 