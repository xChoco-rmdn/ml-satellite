"""
Satellite Image Training Script with Distributed Training

This script implements distributed training across multiple GPUs using TensorFlow's
MirroredStrategy for satellite nowcasting with ConvLSTM.

Usage:
    python 2_training.py

The script will:
1. Detect available GPUs and set up distributed training
2. Load and preprocess data
3. Train the model across GPUs
4. Save checkpoints and visualizations
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Suppress all TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logging
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Disable TensorFlow debugging tools
tf.debugging.disable_check_numerics()
tf.debugging.disable_traceback_filtering()

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.pipeline.train_pipeline import TrainPipeline
from src.logger import logger
from src.exception import CustomException

def plot_training_history(history, save_path):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_distributed_dataset(X, y, batch_size, strategy):
    """Create a distributed dataset for training"""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return strategy.experimental_distribute_dataset(dataset)

def main():
    try:
        logger.info("Starting training pipeline")
        os.makedirs('artifacts', exist_ok=True)
        
        # Initialize pipeline with small batch size for CPU
        trainer = TrainPipeline(batch_size=2)
        
        try:
            # Use the pipeline's training method which handles strategy setup
            metrics, history = trainer.initiate_training()
            
            # Plot and save training history
            plot_training_history(
                history,
                os.path.join('artifacts', f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            )
            
            logger.info("Training pipeline completed successfully!")
            return metrics, history
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)
            
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    # Configure GPU memory growth
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {str(e)}")
        else:
            logger.warning("No GPU devices found, using CPU")
            
    except Exception as e:
        logger.error(f"Error configuring GPU: {str(e)}")
    
    main() 