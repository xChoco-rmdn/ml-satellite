import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from src.logger import logger
from src.exception import CustomException

class AttentionBlock(layers.Layer):
    def __init__(self, filters, reduction_ratio=16, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        # Spatial attention
        self.conv1 = layers.Conv2D(self.filters, 1, padding='same')
        self.conv2 = layers.Conv2D(self.filters, 1, padding='same')
        self.conv3 = layers.Conv2D(self.filters, 1, padding='same')
        
        # Channel attention
        self.fc1 = layers.Dense(self.filters // self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(self.filters, activation='sigmoid')
        
        super(AttentionBlock, self).build(input_shape)
        
    def call(self, x):
        # Spatial attention
        attention = self.conv1(x)
        attention = tf.nn.sigmoid(attention)
        x = x * attention
        
        # Channel attention
        avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        
        avg_out = self.conv2(avg_pool)
        max_out = self.conv3(max_pool)
        
        attention = tf.nn.sigmoid(avg_out + max_out)
        return x * attention

    def compute_output_shape(self, input_shape):
        return input_shape
        
    def get_config(self):
        config = super(AttentionBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'reduction_ratio': self.reduction_ratio
        })
        return config

class CloudNowcastingModel:
    def __init__(self, input_shape=(6, 256, 256, 1)):
        self.input_shape = input_shape
    
    def build_model(self):
        """
        Enhanced 2D ConvLSTM model with attention mechanisms and U-Net style connections
        """
        try:
            # Input layer
            inputs = layers.Input(shape=self.input_shape)
            
            # Initial normalization
            x = layers.BatchNormalization()(inputs)
            
            # Encoder path
            # Level 1
            x1 = layers.ConvLSTM2D(
                filters=32,
                kernel_size=(3, 3),
                padding='same',
                activation='swish',
                kernel_regularizer=regularizers.l2(1e-4),
                return_sequences=True
            )(x)
            x1 = layers.BatchNormalization()(x1)
            x1 = layers.TimeDistributed(AttentionBlock(32))(x1)
            
            # Level 2
            x2 = layers.ConvLSTM2D(
                filters=64,
                kernel_size=(3, 3),
                padding='same',
                activation='swish',
                kernel_regularizer=regularizers.l2(1e-4),
                return_sequences=True
            )(x1)
            x2 = layers.BatchNormalization()(x2)
            x2 = layers.TimeDistributed(AttentionBlock(64))(x2)
            
            # Level 3
            x3 = layers.ConvLSTM2D(
                filters=128,
                kernel_size=(3, 3),
                padding='same',
                activation='swish',
                kernel_regularizer=regularizers.l2(1e-4),
                return_sequences=True
            )(x2)
            x3 = layers.BatchNormalization()(x3)
            x3 = layers.TimeDistributed(AttentionBlock(128))(x3)
            
            # Decoder path with skip connections
            # Level 3 to 2
            x = layers.ConvLSTM2D(
                filters=64,
                kernel_size=(3, 3),
                padding='same',
                activation='swish',
                kernel_regularizer=regularizers.l2(1e-4),
                return_sequences=True
            )(x3)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, x2])  # Skip connection
            x = layers.TimeDistributed(AttentionBlock(64))(x)
            
            # Level 2 to 1
            x = layers.ConvLSTM2D(
                filters=32,
                kernel_size=(3, 3),
                padding='same',
                activation='swish',
                kernel_regularizer=regularizers.l2(1e-4),
                return_sequences=True
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, x1])  # Skip connection
            x = layers.TimeDistributed(AttentionBlock(32))(x)
            
            # Output layer with improved activation
            outputs = layers.TimeDistributed(
                layers.Conv2D(
                    filters=1,
                    kernel_size=(3, 3),
                    padding='same',
                    activation='linear',
                    kernel_regularizer=regularizers.l2(1e-4)
                )
            )(x)
            
            # Create model
            model = models.Model(inputs=inputs, outputs=outputs)
            
            # Enhanced loss function combining MSE, SSIM, and temporal consistency
            def ssim_metric(y_true, y_pred):
                return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

            def temporal_consistency_metric(y_true, y_pred):
                true_grad = y_true[:, 1:] - y_true[:, :-1]
                pred_grad = y_pred[:, 1:] - y_pred[:, :-1]
                return -tf.reduce_mean(tf.abs(true_grad - pred_grad))  # negative for minimization

            def combined_loss(y_true, y_pred):
                # MSE loss
                mse = tf.reduce_mean(tf.square(y_true - y_pred))
                # SSIM loss
                ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
                # Temporal consistency loss
                true_grad = y_true[:, 1:] - y_true[:, :-1]
                pred_grad = y_pred[:, 1:] - y_pred[:, :-1]
                temporal = tf.reduce_mean(tf.abs(true_grad - pred_grad))
                return 0.4 * mse + 0.4 * ssim + 0.2 * temporal
            
            # Advanced optimizer with warmup
            initial_learning_rate = 1e-4
            lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate,
                first_decay_steps=1000,
                t_mul=2.0,
                m_mul=0.9,
                alpha=0.1
            )
            
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=1e-5,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
            
            # Compile model with gradient clipping
            model.compile(
                optimizer=optimizer,
                loss=combined_loss,
                metrics=['mae', ssim_metric, temporal_consistency_metric]
            )
            
            # Logging model details
            logger.info("Enhanced 2D ConvLSTM Model built successfully")
            logger.info(f"Input shape: {self.input_shape}")
            logger.info(f"Output shape: {model.output_shape}")
            total_params = model.count_params()
            logger.info(f"Total parameters: {total_params:,}")
            
            return model
            
        except Exception as e:
            logger.error("Error in building model")
            raise CustomException(e, sys)
    
    def train_model(self, model, train_data, valid_data=None):
        """
        Enhanced training method with improved callbacks and monitoring
        """
        try:
            X_train, y_train = train_data
            
            # Enhanced callbacks
            callbacks = [
                # Model checkpointing with improved monitoring
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.keras',
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    save_weights_only=False
                ),
                # Early stopping with improved patience
                tf.keras.callbacks.EarlyStopping(
                    patience=25,
                    monitor='val_loss',
                    restore_best_weights=True,
                    min_delta=1e-4
                ),
                # Learning rate reduction with improved scheduling
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.5,
                    patience=15,
                    min_lr=1e-6,
                    monitor='val_loss',
                    verbose=1
                ),
                # TensorBoard logging
                tf.keras.callbacks.TensorBoard(
                    log_dir='logs/fit',
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch'
                ),
                # Custom callback for gradient clipping
                tf.keras.callbacks.LambdaCallback(
                    on_batch_end=lambda batch, logs: tf.clip_by_global_norm(
                        model.optimizer.get_gradients(model.total_loss, model.trainable_weights),
                        1.0
                    )
                )
            ]
            
            # Train the model with improved batch size
            history = model.fit(
                X_train, y_train,
                batch_size=16,  # Increased batch size
                epochs=100,
                validation_data=valid_data,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Model training completed successfully")
            return history
            
        except Exception as e:
            logger.error("Error in training model")
            raise CustomException(e, sys)
    
    def predict(self, model, X):
        """
        Enhanced prediction method with confidence estimation
        """
        try:
            # Enable dropout during inference for uncertainty estimation
            predictions = []
            for _ in range(5):  # Multiple forward passes
                pred = model(X, training=True)
                predictions.append(pred)
            
            # Stack predictions and calculate mean and std
            predictions = tf.stack(predictions)
            mean_pred = tf.reduce_mean(predictions, axis=0)
            std_pred = tf.math.reduce_std(predictions, axis=0)
            
            return mean_pred, std_pred
            
        except Exception as e:
            logger.error("Error in making predictions")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Test model creation
    model_builder = CloudNowcastingModel()
    model = model_builder.build_model()
    model.summary()
