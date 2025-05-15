import os
import sys
import numpy as np
from src.components.data_transformations import DataTransformation
from src.components.model import CloudNowcastingModel
from src.components.data_generator import SatelliteDataGenerator
from src.logger import logger
from src.exception import CustomException
from src.utils import save_object, evaluate_model
import tensorflow as tf

class TrainPipeline:
    def __init__(self, batch_size=4):
        self.data_transform = DataTransformation()
        self.model_trainer = CloudNowcastingModel()
        self.batch_size = batch_size
        
        # Enable mixed precision training
        self.policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(self.policy)
        
    def setup_training_strategy(self):
        """Setup distributed training strategy if multiple GPUs are available"""
        try:
            # Get all available GPUs
            gpus = tf.config.list_physical_devices('GPU')
            
            if len(gpus) > 1:
                # Use MirroredStrategy for multi-GPU training
                strategy = tf.distribute.MirroredStrategy()
                logger.info(f"Using MirroredStrategy with {len(gpus)} GPUs")
            else:
                # Use default strategy for single GPU or CPU
                strategy = tf.distribute.get_strategy()
                if gpus:
                    logger.info("Using single GPU")
                else:
                    logger.info("Using CPU")
            
            return strategy
            
        except Exception as e:
            logger.error("Error setting up training strategy")
            raise CustomException(e, sys)
        
    def initiate_training(self):
        try:
            logger.info("Starting training pipeline")
            
            # Create necessary directories
            os.makedirs('artifacts', exist_ok=True)
            os.makedirs('logs/fit', exist_ok=True)
            
            # Load training and test data
            logger.info("Loading data...")
            train_data = np.load("data/train/train_data_20250401_0000_to_20250425_0100.npy")
            test_data = np.load("data/test/test_data_20250425_0110_to_20250501_0000.npy")
            
            logger.info(f"Training data shape: {train_data.shape}")
            logger.info(f"Test data shape: {test_data.shape}")
            
            # Transform data
            logger.info("Transforming data...")
            (X_train, y_train), (X_test, y_test) = self.data_transform.initiate_data_transformation(
                train_data, test_data
            )
            
            # Create validation split from training data
            val_split = 0.1
            val_size = int(len(X_train) * val_split)
            
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train = X_train[:-val_size]
            y_train = y_train[:-val_size]
            
            logger.info(f"Final data shapes:")
            logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            logger.info(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
            logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
            
            # Setup training strategy
            strategy = self.setup_training_strategy()
            
            # Create optimized data generators with prefetching
            train_generator = SatelliteDataGenerator(
                X_train, y_train,
                batch_size=self.batch_size,
                prefetch_factor=2
            )
            val_generator = SatelliteDataGenerator(
                X_val, y_val,
                batch_size=self.batch_size,
                prefetch_factor=2
            )
            test_generator = SatelliteDataGenerator(
                X_test, y_test,
                batch_size=self.batch_size,
                prefetch_factor=2
            )
            
            # Build and compile model within strategy scope
            with strategy.scope():
                logger.info("Building model...")
                model = self.model_trainer.build_model()
                
                # Use mixed precision optimizer
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.model_trainer.config.learning_rate,
                    epsilon=1e-4  # Increased epsilon for mixed precision training
                )
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
                
                model.compile(
                    optimizer=optimizer,
                    loss='mse',
                    metrics=['mae']
                )
            
            # Create callbacks with reduced logging frequency
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
                    profile_batch='100,120'  # Profile performance for 20 batches
                )
            ]
            
            # Enable TensorFlow performance optimizations
            tf.config.optimizer.set_jit(True)  # Enable XLA
            
            # Train model with optimized settings
            logger.info("Training model...")
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=50,
                callbacks=callbacks
            )
            
            # Evaluate on test set
            logger.info("Evaluating model on test set...")
            test_metrics = model.evaluate(
                test_generator,
                verbose=1
            )
            metrics = dict(zip(model.metrics_names, test_metrics))
            
            logger.info("Test Set Metrics:")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
            
            # Save the final model
            logger.info("Saving final model...")
            model.save('artifacts/final_model.h5')
            
            # Save the data transformer for inference
            logger.info("Saving data transformer...")
            save_object(
                file_path='artifacts/data_transformer.pkl',
                obj=self.data_transform
            )
            
            logger.info("Training pipeline completed successfully!")
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
    
    trainer = TrainPipeline(batch_size=4)  # Reduced batch size for memory efficiency
    metrics, history = trainer.initiate_training()
