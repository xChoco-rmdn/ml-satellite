"""
Satellite Image Training Script

This script handles the model training process including:
- Model architecture definition
- Training loop implementation
- Hyperparameter tuning
- Model checkpointing
- Training visualization

Usage:
    python 2_training.py

The script will:
1. Load preprocessed data
2. Initialize and train the model
3. Save the trained model to artifacts/
4. Generate training visualizations
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

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

def main():
    try:
        logger.info("Starting training pipeline")
        os.makedirs('artifacts', exist_ok=True)
        
        # Initialize pipeline with smaller batch size
        trainer = TrainPipeline(batch_size=2)
        
        # Setup training strategy first
        strategy = trainer.setup_training_strategy()
        
        # Load preprocessed data
        logger.info("Loading preprocessed data...")
        X_train = np.load('data/processed/X_train.npy')
        y_train = np.load('data/processed/y_train.npy')
        X_test = np.load('data/processed/X_test.npy')
        y_test = np.load('data/processed/y_test.npy')
        
        logger.info(f"Loaded data shapes:")
        logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

        # Create validation split from training data
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

        # Build and compile model within strategy scope
        with strategy.scope():
            logger.info("Building model...")
            model = trainer.model_trainer.build_model()
            
            # Use mixed precision optimizer with reduced learning rate
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=1e-5,  # Reduced learning rate
                weight_decay=1e-6,   # Reduced weight decay
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
        # Callbacks with adjusted parameters
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'artifacts/best_model.h5',
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=15,  # Increased patience
                monitor='val_loss',
                restore_best_weights=True,
                min_delta=1e-4
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=8,  # Increased patience
                monitor='val_loss',
                min_lr=1e-7
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='logs/fit',
                histogram_freq=1,
                update_freq='epoch'
            )
        ]
        
        # Train model with adjusted parameters
        logger.info("Starting model training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=2,  # Reduced batch size
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        test_metrics = model.evaluate(X_test, y_test, verbose=1)
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
        
        logger.info("Training pipeline completed successfully!")
        return metrics, history
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    # Configure GPU memory growth and prevent plugin registration warnings
    try:
        # Disable TensorFlow logging for plugin registration
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Set memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
                
                # Set visible devices to use all available GPUs
                tf.config.set_visible_devices(gpus, 'GPU')
                
                # Verify GPU is being used
                logger.info("GPU devices configured:")
                for gpu in gpus:
                    logger.info(f"  - {gpu.name}")
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {str(e)}")
        else:
            logger.warning("No GPU devices found, using CPU")
            
    except Exception as e:
        logger.error(f"Error configuring GPU: {str(e)}")
    
    main() 