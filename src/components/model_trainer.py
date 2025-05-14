import os
import sys
import numpy as np
from src.logger import logger
from src.exception import CustomException
from src.utils import save_object, evaluate_model
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('models', 'saved_models')
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    patience: int = 10

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        os.makedirs(self.model_trainer_config.trained_model_path, exist_ok=True)
        
    def build_convlstm_model(self, input_shape):
        """
        Build ConvLSTM model for cloud nowcasting
        """
        try:
            model = Sequential([
                # First ConvLSTM layer
                ConvLSTM2D(
                    filters=64,
                    kernel_size=(3, 3),
                    padding='same',
                    return_sequences=True,
                    activation='relu',
                    input_shape=input_shape
                ),
                BatchNormalization(),
                
                # Second ConvLSTM layer
                ConvLSTM2D(
                    filters=64,
                    kernel_size=(3, 3),
                    padding='same',
                    return_sequences=False,
                    activation='relu'
                ),
                BatchNormalization(),
                
                # Output Conv layer
                Conv2D(
                    filters=1,
                    kernel_size=(3, 3),
                    activation='linear',
                    padding='same'
                )
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            raise CustomException(e, sys)
            
    def train_model(self, X_train, y_train):
        """
        Train the model with the provided data
        """
        try:
            logger.info("Started model training")
            
            # Get input shape from training data
            input_shape = (X_train.shape[1], *X_train.shape[2:])
            
            # Build model
            model = self.build_convlstm_model(input_shape)
            
            # Setup callbacks
            checkpoint_path = os.path.join(
                self.model_trainer_config.trained_model_path,
                'model_checkpoint.h5'
            )
            
            callbacks = [
                ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.model_trainer_config.patience,
                    restore_best_weights=True
                )
            ]
            
            # Train model
            history = model.fit(
                X_train,
                y_train,
                batch_size=self.model_trainer_config.batch_size,
                epochs=self.model_trainer_config.epochs,
                validation_split=self.model_trainer_config.validation_split,
                callbacks=callbacks
            )
            
            # Save the final model
            final_model_path = os.path.join(
                self.model_trainer_config.trained_model_path,
                'final_model.h5'
            )
            model.save(final_model_path)
            
            logger.info(f"Model training completed. Model saved to {final_model_path}")
            
            return model, history
            
        except Exception as e:
            logger.error("Error in model training")
            raise CustomException(e, sys)
            
    def initiate_model_trainer(self, train_data_path):
        """
        Initiate the model training process
        """
        try:
            # Load training data
            data = np.load(train_data_path)
            X_train = data['X']
            y_train = data['y']
            
            # Train model
            model, history = self.train_model(X_train, y_train)
            
            # Evaluate on validation set
            val_idx = int(len(X_train) * (1 - self.model_trainer_config.validation_split))
            X_val = X_train[val_idx:]
            y_val = y_train[val_idx:]
            
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            metrics = evaluate_model(y_val, y_pred)
            
            logger.info(f"Model evaluation metrics: {metrics}")
            
            return model, metrics
            
        except Exception as e:
            raise CustomException(e, sys) 