import os
import sys
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import layers, models
from src.logger import logger
from src.exception import CustomException

@dataclass
class ModelConfig:
    # Model Architecture Parameters
    filters: list = None
    kernel_sizes: list = None
    activation: str = 'tanh'
    recurrent_activation: str = 'hard_sigmoid'
    input_shape: tuple = (6, 256, 256, 1)  # (time_steps, height, width, channels)
    num_prediction_steps: int = 6
    
    # Training Parameters
    batch_size: int = 4
    epochs: int = 50
    learning_rate: float = 0.001
    
    # Model Paths
    model_path: str = os.path.join('artifacts', 'model.h5')
    
    def __post_init__(self):
        # Default architecture if not specified
        if self.filters is None:
            self.filters = [32, 64, 64, 32]  # Modified filter sizes
        if self.kernel_sizes is None:
            self.kernel_sizes = [(3, 3), (3, 3), (3, 3), (3, 3)]  # Smaller kernels

class CloudNowcastingModel:
    def __init__(self):
        self.config = ModelConfig()
        
    def build_model(self):
        """
        Build the ConvLSTM model architecture with spatial downsampling
        Returns:
            tf.keras.Model: Compiled model
        """
        try:
            # Input layer
            inputs = layers.Input(shape=self.config.input_shape)
            
            # Initial spatial downsampling using average pooling
            x = layers.TimeDistributed(
                layers.AveragePooling2D(pool_size=(2, 2))
            )(inputs)  # Now 128x128
            
            # First ConvLSTM layer
            x = layers.ConvLSTM2D(
                filters=self.config.filters[0],
                kernel_size=self.config.kernel_sizes[0],
                padding='same',
                activation=self.config.activation,
                recurrent_activation=self.config.recurrent_activation,
                return_sequences=True
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(x)  # Now 64x64
            
            # Second ConvLSTM layer
            x = layers.ConvLSTM2D(
                filters=self.config.filters[1],
                kernel_size=self.config.kernel_sizes[1],
                padding='same',
                activation=self.config.activation,
                recurrent_activation=self.config.recurrent_activation,
                return_sequences=True
            )(x)
            x = layers.BatchNormalization()(x)
            
            # Third ConvLSTM layer
            x = layers.ConvLSTM2D(
                filters=self.config.filters[2],
                kernel_size=self.config.kernel_sizes[2],
                padding='same',
                activation=self.config.activation,
                recurrent_activation=self.config.recurrent_activation,
                return_sequences=True
            )(x)
            x = layers.BatchNormalization()(x)
            
            # Upsampling path
            x = layers.TimeDistributed(
                layers.UpSampling2D(size=(2, 2))
            )(x)  # Now 128x128
            
            # Final ConvLSTM layer
            x = layers.ConvLSTM2D(
                filters=self.config.filters[3],
                kernel_size=self.config.kernel_sizes[3],
                padding='same',
                activation=self.config.activation,
                recurrent_activation=self.config.recurrent_activation,
                return_sequences=True
            )(x)
            x = layers.BatchNormalization()(x)
            
            # Final upsampling to original resolution
            x = layers.TimeDistributed(
                layers.UpSampling2D(size=(2, 2))
            )(x)  # Now 256x256
            
            # Final convolution for output
            outputs = layers.TimeDistributed(
                layers.Conv2D(
                    filters=1,
                    kernel_size=(3, 3),
                    padding='same',
                    activation='linear'
                )
            )(x)
            
            # Create and compile model
            model = models.Model(inputs=inputs, outputs=outputs)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
            
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            logger.info("Model built successfully")
            logger.info(f"Input shape: {self.config.input_shape}")
            logger.info(f"Output shape: {model.output_shape}")
            
            # Print model summary with parameter count
            total_params = model.count_params()
            logger.info(f"Total parameters: {total_params:,}")
            
            return model
            
        except Exception as e:
            logger.error("Error in building model")
            raise CustomException(e, sys)
    
    def train_model(self, model, train_data, valid_data=None, callbacks=None):
        """
        Train the model
        Args:
            model: tf.keras.Model
            train_data: tuple of (X_train, y_train)
            valid_data: tuple of (X_val, y_val)
            callbacks: list of keras callbacks
        Returns:
            history: Training history
        """
        try:
            X_train, y_train = train_data
            
            # Create default callbacks if none provided
            if callbacks is None:
                callbacks = [
                    tf.keras.callbacks.ModelCheckpoint(
                        self.config.model_path,
                        save_best_only=True,
                        monitor='val_loss' if valid_data else 'loss'
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        patience=10,
                        monitor='val_loss' if valid_data else 'loss'
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        factor=0.5,
                        patience=5,
                        monitor='val_loss' if valid_data else 'loss'
                    )
                ]
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_data=valid_data,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Model training completed")
            return history
            
        except Exception as e:
            logger.error("Error in training model")
            raise CustomException(e, sys)
    
    def predict(self, model, X):
        """
        Make predictions using the trained model
        Args:
            model: trained tf.keras.Model
            X: input data
        Returns:
            predictions: model predictions
        """
        try:
            predictions = model.predict(X)
            return predictions
            
        except Exception as e:
            logger.error("Error in making predictions")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Test model creation
    model_builder = CloudNowcastingModel()
    model = model_builder.build_model()
    model.summary() 