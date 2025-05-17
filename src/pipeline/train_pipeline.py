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
import logging

class TrainPipeline:
    def __init__(self, batch_size=1):  # Reduced default batch size
        # Set environment variables to suppress TensorFlow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL only
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
        
        # Set GPU environment variables before any TF operations
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 1:
            # Set environment variables for better multi-GPU performance
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
            os.environ['TF_GPU_THREAD_COUNT'] = '1'
            os.environ['TF_USE_CUDNN'] = '1'
            os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'
            
            # Configure memory growth for all GPUs before TF initialization
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Set memory limit to 90% of available memory
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=12500)]  # ~12.5GB for T4
                )
        
        # Configure TensorFlow logging
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        
        # Disable TensorFlow debug logs
        tf.debugging.disable_traceback_filtering()
        tf.debugging.disable_check_numerics()
        
        # Configure Python logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('tensorflow.python').setLevel(logging.ERROR)
        logging.getLogger('tensorflow.python.ops').setLevel(logging.ERROR)
        
        # Create necessary directories
        os.makedirs('artifacts', exist_ok=True)
        os.makedirs('logs/fit', exist_ok=True)
        
        # Setup distributed training strategy
        self.strategy = self.setup_training_strategy()
        
        # Initialize components within strategy scope
        with self.strategy.scope():
            self.data_transform = DataTransformation()
            self.model_trainer = CloudNowcastingModel()
            # Scale batch size based on number of GPUs but keep it small
            self.batch_size = min(batch_size * self.strategy.num_replicas_in_sync, 4)
            logger.info(f"Using batch size of {self.batch_size} across {self.strategy.num_replicas_in_sync} GPUs")
            
            # Enable mixed precision training
            self.policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(self.policy)
        
    def setup_training_strategy(self):
        """Setup distributed training strategy if multiple GPUs are available"""
        try:
            # Get all available GPUs
            gpus = tf.config.list_physical_devices('GPU')
            logger.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            
            if len(gpus) > 1:
                # Use MirroredStrategy for multi-GPU training with explicit devices
                strategy = tf.distribute.MirroredStrategy(
                    devices=[f"/GPU:{i}" for i in range(len(gpus))],
                    cross_device_ops=tf.distribute.NcclAllReduce()  # Use NCCL for better GPU communication
                )
                logger.info(f"Using MirroredStrategy with {len(gpus)} GPUs: {[gpu.name for gpu in gpus]}")
            elif len(gpus) == 1:
                strategy = tf.distribute.OneDeviceStrategy(f"/GPU:0")
                logger.info(f"Using single GPU: {gpus[0].name}")
            else:
                strategy = tf.distribute.get_strategy()
                logger.info("No GPUs found, using CPU")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error setting up training strategy: {str(e)}")
            raise CustomException(e, sys)
    
    def validate_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Validate data shapes and values"""
        try:
            # Check shapes
            assert len(X_train.shape) == 5, f"Expected 5D input, got shape {X_train.shape}"
            assert len(y_train.shape) == 5, f"Expected 5D target, got shape {y_train.shape}"
            
            # Check for NaN values
            assert not np.isnan(X_train).any(), "NaN values found in training input"
            assert not np.isnan(y_train).any(), "NaN values found in training target"
            assert not np.isnan(X_val).any(), "NaN values found in validation input"
            assert not np.isnan(y_val).any(), "NaN values found in validation target"
            assert not np.isnan(X_test).any(), "NaN values found in test input"
            assert not np.isnan(y_test).any(), "NaN values found in test target"
            
            # Check value ranges
            assert X_train.min() >= 0 and X_train.max() <= 1, "Training input values outside [0,1] range"
            assert y_train.min() >= 0 and y_train.max() <= 1, "Training target values outside [0,1] range"
            
            logger.info("Data validation passed successfully")
            
        except AssertionError as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise CustomException(e, sys)
        
    def initiate_training(self):
        try:
            logger.info("Starting training pipeline")
            
            # Load training and test data
            train_data = np.load("data/train/train_data_20250401_0000_to_20250425_0100.npy")
            test_data = np.load("data/test/test_data_20250425_0110_to_20250501_0000.npy")
            
            # Transform data
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
            
            # Validate data
            self.validate_data(X_train, y_train, X_val, y_val, X_test, y_test)
            
            # Log data shapes once
            logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            # Create distributed datasets within strategy scope
            with self.strategy.scope():
                # Create optimized data generators with distributed strategy
                train_generator = SatelliteDataGenerator(
                    X_train, y_train,
                    batch_size=max(1, self.batch_size // self.strategy.num_replicas_in_sync),
                    prefetch_factor=1,  # Reduced prefetch factor
                    augment=True
                )
                val_generator = SatelliteDataGenerator(
                    X_val, y_val,
                    batch_size=max(1, self.batch_size // self.strategy.num_replicas_in_sync),
                    prefetch_factor=1,
                    augment=False
                )
                test_generator = SatelliteDataGenerator(
                    X_test, y_test,
                    batch_size=max(1, self.batch_size // self.strategy.num_replicas_in_sync),
                    prefetch_factor=1,
                    augment=False
                )
                
                # Build and compile model within strategy scope
                model = self.model_trainer.build_model()
                
                # Use a more memory-efficient optimizer
                optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=1e-4,
                    weight_decay=1e-5,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7,
                    clipnorm=1.0,
                    # Enable gradient accumulation
                    gradient_accumulation_steps=4
                )
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
                
                model.compile(
                    optimizer=optimizer,
                    loss='mse',
                    metrics=['mae'],
                    jit_compile=False  # Disable XLA compilation to save memory
                )
            
            # Create callbacks
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    'artifacts/best_model.h5',
                    save_best_only=True,
                    monitor='val_loss',
                    save_weights_only=True,
                    verbose=0
                ),
                tf.keras.callbacks.EarlyStopping(
                    patience=10,
                    monitor='val_loss',
                    restore_best_weights=True,
                    verbose=0
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.5,
                    patience=5,
                    monitor='val_loss',
                    min_lr=1e-6,
                    verbose=0
                ),
                # Add memory monitoring callback
                tf.keras.callbacks.TensorBoard(
                    log_dir='logs/fit',
                    histogram_freq=1,
                    write_graph=False,  # Disable graph writing to save memory
                    write_images=False,  # Disable image writing to save memory
                    update_freq='epoch',
                    profile_batch=0  # Disable profiling to save memory
                )
            ]
            
            # Custom callback for memory monitoring
            class MemoryMonitorCallback(tf.keras.callbacks.Callback):
                def on_epoch_begin(self, epoch, logs=None):
                    if epoch == 0:
                        logger.info("GPU Memory Usage at start of training:")
                        for gpu in tf.config.list_physical_devices('GPU'):
                            logger.info(f"GPU: {gpu.name}")
                            try:
                                tf.config.experimental.get_memory_info(gpu.name)
                            except:
                                pass
            
            callbacks.append(MemoryMonitorCallback())
            
            # Train model with distributed strategy
            logger.info("Starting model training...")
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=50,
                callbacks=callbacks,
                workers=1,  # Reduced workers
                use_multiprocessing=False,  # Disable multiprocessing to save memory
                verbose=1
            )
            
            # Evaluate on test set
            logger.info("Evaluating model...")
            test_metrics = model.evaluate(
                test_generator,
                verbose=0
            )
            metrics = dict(zip(model.metrics_names, test_metrics))
            
            # Log final metrics in a single line
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"Final metrics - {metrics_str}")
            
            # Save model and transformer
            model.save('artifacts/final_model.h5', save_format='h5')
            save_object(
                file_path='artifacts/data_transformer.pkl',
                obj=self.data_transform
            )
            
            logger.info("Training completed successfully!")
            return metrics, history
            
        except Exception as e:
            logger.error("Error in training pipeline")
            raise CustomException(e, sys)

if __name__ == "__main__":
    trainer = TrainPipeline(batch_size=4)  # Reduced batch size for memory efficiency
    metrics, history = trainer.initiate_training()
