import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from src.logger import logger
from src.exception import CustomException
import seaborn as sns
from datetime import datetime

class ModelEvaluator:
    def __init__(self, model, data_transformer):
        self.model = model
        self.data_transformer = data_transformer
        self.metrics = {}
        self.visualization_path = os.path.join('artifacts', 'visualizations')
        os.makedirs(self.visualization_path, exist_ok=True)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate various performance metrics"""
        try:
            # Reshape if needed
            y_true = y_true.reshape(-1)
            y_pred = y_pred.reshape(-1)
            
            self.metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            
            logger.info("Model Performance Metrics:")
            for metric_name, value in self.metrics.items():
                logger.info(f"{metric_name.upper()}: {value:.4f}")
                
            return self.metrics
            
        except Exception as e:
            logger.error("Error in calculating metrics")
            raise CustomException(e, sys)
    
    def plot_training_history(self, history):
        """Plot training and validation metrics over epochs"""
        try:
            plt.figure(figsize=(12, 4))
            
            # Plot loss
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot MAE
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualization_path, 'training_history.png'))
            plt.close()
            
            logger.info(f"Training history plot saved to {self.visualization_path}")
            
        except Exception as e:
            logger.error("Error in plotting training history")
            raise CustomException(e, sys)
    
    def plot_prediction_examples(self, X_test, y_test, num_examples=3):
        """Plot example predictions vs ground truth"""
        try:
            predictions = self.model.predict(X_test[:num_examples])
            
            for i in range(num_examples):
                plt.figure(figsize=(15, 5))
                
                # Plot input sequence
                for t in range(X_test.shape[1]):
                    plt.subplot(2, X_test.shape[1], t + 1)
                    plt.imshow(X_test[i, t, :, :, 0], cmap='jet')
                    plt.title(f'Input t-{X_test.shape[1]-t}')
                    plt.axis('off')
                
                # Plot ground truth vs predictions
                for t in range(predictions.shape[1]):
                    plt.subplot(2, predictions.shape[1], predictions.shape[1] + t + 1)
                    
                    # Create difference plot
                    diff = predictions[i, t, :, :, 0] - y_test[i, t, :, :, 0]
                    plt.imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
                    plt.title(f'Pred vs Truth t+{t+1}')
                    plt.colorbar()
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.visualization_path, f'prediction_example_{i}.png'))
                plt.close()
            
            logger.info(f"Prediction examples saved to {self.visualization_path}")
            
        except Exception as e:
            logger.error("Error in plotting prediction examples")
            raise CustomException(e, sys)
    
    def plot_error_distribution(self, y_true, y_pred):
        """Plot error distribution"""
        try:
            errors = (y_pred - y_true).reshape(-1)
            
            plt.figure(figsize=(10, 6))
            sns.histplot(errors, kde=True)
            plt.title('Prediction Error Distribution')
            plt.xlabel('Error')
            plt.ylabel('Count')
            plt.savefig(os.path.join(self.visualization_path, 'error_distribution.png'))
            plt.close()
            
            logger.info(f"Error distribution plot saved to {self.visualization_path}")
            
        except Exception as e:
            logger.error("Error in plotting error distribution")
            raise CustomException(e, sys)
    
    def evaluate_model(self, test_generator, history):
        """Comprehensive model evaluation"""
        try:
            logger.info("Starting model evaluation...")
            
            # Get predictions
            y_pred = self.model.predict(test_generator)
            y_true = np.concatenate([y for _, y in test_generator], axis=0)
            
            # Calculate metrics
            self.calculate_metrics(y_true.reshape(-1), y_pred.reshape(-1))
            
            # Generate visualizations
            self.plot_training_history(history)
            self.plot_prediction_examples(
                next(iter(test_generator))[0],  # Get first batch of inputs
                next(iter(test_generator))[1]   # Get first batch of targets
            )
            self.plot_error_distribution(y_true, y_pred)
            
            # Save evaluation results
            self.save_evaluation_results()
            
            logger.info("Model evaluation completed successfully")
            return self.metrics
            
        except Exception as e:
            logger.error("Error in model evaluation")
            raise CustomException(e, sys)
    
    def save_evaluation_results(self):
        """Save evaluation results to a text file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.visualization_path, f'evaluation_results_{timestamp}.txt')
            
            with open(results_path, 'w') as f:
                f.write("Model Evaluation Results\n")
                f.write("======================\n\n")
                
                f.write("Performance Metrics:\n")
                f.write("-----------------\n")
                for metric_name, value in self.metrics.items():
                    f.write(f"{metric_name.upper()}: {value:.4f}\n")
                
            logger.info(f"Evaluation results saved to {results_path}")
            
        except Exception as e:
            logger.error("Error in saving evaluation results")
            raise CustomException(e, sys) 