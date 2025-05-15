# Phase 4: Model Deployment and Prediction Pipeline

## 1. Initial Setup and Imports

First, let's import necessary libraries and load our trained model.

```python
import os
import numpy as np
import tensorflow as tf
import xarray as xr
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import our configurations and processors
from phase1_setup_and_data_preparation import ProjectConfig
from phase2_data_preprocessing import DataProcessor, SatelliteDataNormalizer
from phase3_model_development import ModelConfig, build_convlstm_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize configurations
config = ProjectConfig()
model_config = ModelConfig()
```

## 2. Model Loading and Version Management

Implement model loading and version tracking.

```python
class ModelManager:
    def __init__(self, config, model_name="convlstm_model"):
        self.config = config
        self.model_name = model_name
        self.model_dir = Path(config.model_save_path) / model_name
        self.version_file = self.model_dir / "version_info.json"
        
    def load_latest_model(self):
        """Load the latest version of the model"""
        try:
            model_path = self.model_dir / "best_model.h5"
            if not model_path.exists():
                raise FileNotFoundError(f"No model found at {model_path}")
            
            logger.info(f"Loading model from {model_path}")
            model = tf.keras.models.load_model(str(model_path))
            
            # Load version info
            version_info = self.load_version_info()
            logger.info(f"Model version info: {version_info}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def load_version_info(self):
        """Load model version information"""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {"version": "unknown", "timestamp": "unknown"}
    
    def save_version_info(self, version_info):
        """Save model version information"""
        with open(self.version_file, 'w') as f:
            json.dump(version_info, f)

# Initialize model manager
model_manager = ModelManager(config)
model = model_manager.load_latest_model()
```

## 3. Prediction Pipeline

Create a prediction pipeline for real-time forecasting.

```python
class PredictionPipeline:
    def __init__(self, model, config, processor):
        self.model = model
        self.config = config
        self.processor = processor
        
    def prepare_input_sequence(self, data_files, sequence_length=6):
        """
        Prepare input sequence from raw data files
        
        Args:
            data_files: List of paths to input data files
            sequence_length: Length of input sequence
            
        Returns:
            np.array: Processed input sequence
        """
        processed_data = []
        
        # Process each file
        for file_path in data_files[-sequence_length:]:
            result = self.processor.process_file(file_path)
            if result is not None:
                processed_data.append(result['normalized_data']['IR1'])
        
        # Stack into sequence
        if len(processed_data) == sequence_length:
            return np.stack(processed_data)[np.newaxis, ...]
        else:
            raise ValueError(f"Not enough valid data files for sequence (got {len(processed_data)}, need {sequence_length})")
    
    def make_prediction(self, input_sequence):
        """
        Make prediction using the model
        
        Args:
            input_sequence: Processed input sequence
            
        Returns:
            np.array: Predicted sequence
        """
        try:
            # Make prediction
            prediction = self.model.predict(input_sequence)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def visualize_prediction(self, input_sequence, prediction):
        """
        Visualize input sequence and prediction
        """
        fig, axes = plt.subplots(2, 6, figsize=(20, 8))
        
        # Plot input sequence
        for i in range(input_sequence.shape[1]):
            axes[0, i].imshow(input_sequence[0, i, :, :, 0], cmap='RdBu_r')
            axes[0, i].set_title(f'Input T+{i}')
            axes[0, i].axis('off')
        
        # Plot prediction
        for i in range(prediction.shape[1]):
            axes[1, i].imshow(prediction[0, i, :, :, 0], cmap='RdBu_r')
            axes[1, i].set_title(f'Prediction T+{i}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
```

## 4. Real-time Data Processing

Implement real-time data processing and prediction.

```python
class RealTimeProcessor:
    def __init__(self, config, prediction_pipeline):
        self.config = config
        self.pipeline = prediction_pipeline
        self.data_buffer = []
        self.buffer_size = 6  # Input sequence length
        
    def update_buffer(self, new_file_path):
        """
        Update data buffer with new file
        
        Args:
            new_file_path: Path to new data file
        """
        # Add new file to buffer
        self.data_buffer.append(new_file_path)
        
        # Keep only the latest files needed for sequence
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
    
    def process_new_data(self, new_file_path):
        """
        Process new data and make prediction
        
        Args:
            new_file_path: Path to new data file
            
        Returns:
            tuple: (input_sequence, prediction)
        """
        try:
            # Update buffer
            self.update_buffer(new_file_path)
            
            # Check if we have enough data
            if len(self.data_buffer) < self.buffer_size:
                logger.warning(f"Not enough data in buffer (got {len(self.data_buffer)}, need {self.buffer_size})")
                return None, None
            
            # Prepare input sequence
            input_sequence = self.pipeline.prepare_input_sequence(self.data_buffer)
            
            # Make prediction
            prediction = self.pipeline.make_prediction(input_sequence)
            
            return input_sequence, prediction
            
        except Exception as e:
            logger.error(f"Error processing new data: {str(e)}")
            return None, None
```

## 5. Monitoring and Logging

Implement monitoring and logging functions.

```python
class PredictionMonitor:
    def __init__(self, config):
        self.config = config
        self.log_dir = Path("logs/predictions")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def log_prediction(self, timestamp, input_sequence, prediction, metrics):
        """
        Log prediction details and metrics
        """
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "input_shape": input_sequence.shape,
            "prediction_shape": prediction.shape,
            "metrics": metrics
        }
        
        # Save log entry
        log_file = self.log_dir / f"prediction_log_{timestamp.strftime('%Y%m%d')}.json"
        with open(log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
    
    def calculate_metrics(self, prediction):
        """Calculate basic prediction metrics"""
        return {
            "mean_value": float(np.mean(prediction)),
            "std_value": float(np.std(prediction)),
            "min_value": float(np.min(prediction)),
            "max_value": float(np.max(prediction))
        }
```

## 6. Integration Example

Put everything together in a working example.

```python
def main():
    # Initialize components
    processor = DataProcessor(config)
    model = model_manager.load_latest_model()
    
    if model is None:
        logger.error("Failed to load model")
        return
    
    # Initialize prediction pipeline
    prediction_pipeline = PredictionPipeline(model, config, processor)
    
    # Initialize real-time processor
    rt_processor = RealTimeProcessor(config, prediction_pipeline)
    
    # Initialize monitor
    monitor = PredictionMonitor(config)
    
    # Simulate real-time processing
    def process_new_file(file_path):
        # Process new data
        input_sequence, prediction = rt_processor.process_new_data(file_path)
        
        if input_sequence is not None and prediction is not None:
            # Calculate metrics
            metrics = monitor.calculate_metrics(prediction)
            
            # Log prediction
            monitor.log_prediction(
                timestamp=datetime.now(),
                input_sequence=input_sequence,
                prediction=prediction,
                metrics=metrics
            )
            
            # Visualize results
            prediction_pipeline.visualize_prediction(input_sequence, prediction)
            
            return True
        return False
    
    # Example usage with sample file
    sample_files = sorted(Path(config.raw_data_path).rglob("*.nc"))
    if sample_files:
        success = process_new_file(sample_files[0])
        print(f"Processing {'successful' if success else 'failed'}")

if __name__ == "__main__":
    main()
```

## 7. Summary and Next Steps

This notebook has implemented:
1. Model loading and version management
2. Real-time prediction pipeline
3. Data buffer management
4. Monitoring and logging system

Future improvements could include:
- API endpoint for predictions
- Real-time visualization dashboard
- Automated model retraining
- Performance optimization

```python
# Print deployment summary
print("\nDeployment Summary:")
print("-" * 50)
print(f"✓ Model loading system implemented")
print(f"✓ Prediction pipeline created")
print(f"✓ Real-time processing setup")
print(f"✓ Monitoring system integrated")
``` 