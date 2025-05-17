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
        logger.info("Starting distributed training pipeline")
        os.makedirs('artifacts', exist_ok=True)
        
        # Configure distributed training
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            strategy = tf.distribute.MirroredStrategy()
            logger.info(f'Using MirroredStrategy with {len(gpus)} GPU(s)')
        else:
            strategy = tf.distribute.get_strategy()
            logger.info('No GPUs found, using default strategy (CPU)')
        
        logger.info(f'Number of devices: {strategy.num_replicas_in_sync}')
        
        # Calculate global batch size
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            BATCH_SIZE_PER_REPLICA = 4
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
            logger.info(f'Using GPU batch size: {GLOBAL_BATCH_SIZE}')
        else:
            # Smaller batch size for CPU training
            BATCH_SIZE_PER_REPLICA = 2
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA
            logger.info(f'Using CPU batch size: {GLOBAL_BATCH_SIZE}')
        
        try:
            with strategy.scope():
                 # Initialize pipeline with distributed batch size
                 trainer = TrainPipeline(batch_size=GLOBAL_BATCH_SIZE)
                 
                 # Load preprocessed data
                 logger.info("Loading preprocessed data...")
                 X_train = np.load('data/processed/X_train.npy', mmap_mode='r').astype('float32')
                 y_train = np.load('data/processed/y_train.npy', mmap_mode='r').astype('float32')
                 X_test = np.load('data/processed/X_test.npy', mmap_mode='r').astype('float32')
                 y_test = np.load('data/processed/y_test.npy', mmap_mode='r').astype('float32')
                 
                 # Create validation split
                 val_split = 0.1
                 val_size = int(len(X_train) * val_split)
                 X_val = X_train[-val_size:]
                 y_val = y_train[-val_size:]
                 X_train = X_train[:-val_size]
                 y_train = y_train[:-val_size]
                 
                 logger.info(f"Split data shapes:")
                 logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
                 logger.info(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
                 logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

                 # Enable mixed precision training
                 policy = tf.keras.mixed_precision.Policy('mixed_float16')
                 tf.keras.mixed_precision.set_global_policy(policy)
                 
                 # Build and compile model
                 logger.info("Building model...")
                 model = trainer.model_trainer.build_model()
                 model.summary()
                 
                 # Create distributed datasets
                 train_dataset = create_distributed_dataset(X_train, y_train, GLOBAL_BATCH_SIZE, strategy)
                 val_dataset = create_distributed_dataset(X_val, y_val, GLOBAL_BATCH_SIZE, strategy)
                 test_dataset = create_distributed_dataset(X_test, y_test, GLOBAL_BATCH_SIZE, strategy)
                 
                 # Callbacks for distributed training
                 callbacks = [
                     tf.keras.callbacks.ModelCheckpoint(
                         'artifacts/best_model.h5',
                         save_best_only=True,
                         monitor='val_loss',
                         mode='min'
                     ),
                     tf.keras.callbacks.EarlyStopping(
                         patience=20,
                         monitor='val_loss',
                         restore_best_weights=True
                     ),
                     tf.keras.callbacks.ReduceLROnPlateau(
                         factor=0.2,
                         patience=10,
                         monitor='val_loss',
                         min_lr=1e-8
                     ),
                     # Add TensorBoard callback for monitoring
                     tf.keras.callbacks.TensorBoard(
                         log_dir='logs/fit',
                         histogram_freq=1,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch'
                     )
                 ]
                 
                 # Clear GPU memory
                 tf.keras.backend.clear_session()
                 
                 # Train model with distributed strategy
                 logger.info("Starting distributed model training...")
                 history = model.fit(
                     train_dataset,
                     validation_data=val_dataset,
                     epochs=50,
                     callbacks=callbacks,
                     verbose=1
                 )
                 
                 # Evaluate on test set
                 logger.info("Evaluating model on test set...")
                 test_metrics = model.evaluate(test_dataset, verbose=1)
                 metrics = dict(zip(model.metrics_names, test_metrics))
                 
                 logger.info("Test Set Metrics:")
                 for metric_name, value in metrics.items():
                     logger.info(f"{metric_name}: {value:.4f}")
                     
                 # Save final model
                 logger.info("Saving final model...")
                 model.save('artifacts/final_model.h5')
                 
                 # Plot and save training history
                 plot_training_history(
                     history,
                     os.path.join('artifacts', f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                 )
                 
                 logger.info("Distributed training pipeline completed successfully!")
                 return metrics, history
        except Exception as e:
             logger.error(f"Error in training pipeline (MirroredStrategy scope): {str(e)}")
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
                # Set memory growth for all GPUs
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