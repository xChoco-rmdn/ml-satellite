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
        
        # Initialize pipeline
        trainer = TrainPipeline(batch_size=4)
        
        # Setup training strategy first
        strategy = trainer.setup_training_strategy()
        
        # Load preprocessed data directly
        logger.info("Loading preprocessed data from data/processed/ ...")
        X_train = np.load('data/processed/X_train.npy')
        y_train = np.load('data/processed/y_train.npy')
        X_test = np.load('data/processed/X_test.npy')
        y_test = np.load('data/processed/y_test.npy')
        logger.info(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")

        # Create validation split from training data
        val_split = 0.1
        val_size = int(len(X_train) * val_split)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        logger.info(f"Split: X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")

        # Build and compile model within strategy scope
        with strategy.scope():
            model = trainer.model_trainer.build_model()
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'artifacts/best_model.h5',
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                monitor='val_loss',
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                monitor='val_loss',
                min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='logs/fit',
                histogram_freq=1,
                update_freq='epoch',
                profile_batch='100,120'
            )
        ]
        
        logger.info("Training model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=4,
            callbacks=callbacks
        )
        
        logger.info("Evaluating model on test set...")
        test_metrics = model.evaluate(X_test, y_test, verbose=1)
        metrics = dict(zip(model.metrics_names, test_metrics))
        logger.info("Test Set Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
            
        logger.info("Saving final model...")
        model.save('artifacts/final_model.h5')
        logger.info("Training pipeline completed successfully!")
        
        plot_training_history(
            history,
            os.path.join('artifacts', f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        )
        return metrics, history
        
    except Exception as e:
        logger.error("Error in training pipeline")
        raise CustomException(e, sys)

if __name__ == "__main__":
    # Set memory growth for GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            logger.warning(f"Memory growth setting failed: {str(e)}")
    
    main() 