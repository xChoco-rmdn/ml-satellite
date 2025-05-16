"""
Satellite Image Prediction and Evaluation Script

This script handles model evaluation and prediction including:
- Model loading
- Making predictions
- Performance metrics calculation
- Visualization of results
- Error analysis

Usage:
    python 3_prediction_evaluation.py

The script will:
1. Load the trained model
2. Load test data
3. Generate predictions
4. Calculate and display metrics
5. Save visualizations to artifacts/
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.components.model import CloudNowcastingModel
from src.utils import evaluate_model, load_object
from src.logger import logger
from src.exception import CustomException

def plot_predictions(y_true, y_pred, save_path, num_samples=5):
    """Plot and save prediction visualizations"""
    plt.figure(figsize=(15, 3 * num_samples))
    
    for i in range(num_samples):
        # Plot ground truth
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(y_true[i, 0, :, :, 0], cmap='viridis')
        plt.title(f'Ground Truth - Sample {i+1}')
        plt.colorbar()
        
        # Plot prediction
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(y_pred[i, 0, :, :, 0], cmap='viridis')
        plt.title(f'Prediction - Sample {i+1}')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_error_heatmap(y_true, y_pred, save_path):
    """Plot and save error heatmap"""
    error = np.abs(y_true - y_pred)
    mean_error = np.mean(error, axis=(0, 1))  # Average over samples and time
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_error[:, :, 0], cmap='hot', cbar_kws={'label': 'Mean Absolute Error'})
    plt.title('Mean Absolute Error Heatmap')
    plt.savefig(save_path)
    plt.close()

def main():
    try:
        logger.info("Starting prediction and evaluation pipeline")
        
        # Create artifacts directory if it doesn't exist
        os.makedirs('artifacts', exist_ok=True)
        
        # Load test data
        logger.info("Loading test data...")
        X_test = np.load('data/processed/X_test.npy')
        y_test = np.load('data/processed/y_test.npy')
        
        logger.info(f"Test data shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Load trained model
        logger.info("Loading trained model...")
        model = tf.keras.models.load_model('artifacts/final_model.h5')
        
        # Load data transformer
        logger.info("Loading data transformer...")
        data_transformer = load_object('artifacts/data_transformer.pkl')
        
        # Generate predictions
        logger.info("Generating predictions...")
        y_pred, y_pred_std = model.predict(X_test)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = evaluate_model(y_test, y_pred)
        
        # Log metrics
        logger.info("Evaluation metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot predictions
        plot_predictions(
            y_test,
            y_pred,
            os.path.join('artifacts', f'predictions_{timestamp}.png')
        )
        
        # Plot error heatmap
        plot_error_heatmap(
            y_test,
            y_pred,
            os.path.join('artifacts', f'error_heatmap_{timestamp}.png')
        )
        
        # Save predictions and metrics
        np.save(os.path.join('artifacts', f'predictions_{timestamp}.npy'), y_pred)
        np.save(os.path.join('artifacts', f'prediction_std_{timestamp}.npy'), y_pred_std)
        
        logger.info("Prediction and evaluation completed successfully!")
        
    except Exception as e:
        logger.error("Error in prediction and evaluation pipeline")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main() 