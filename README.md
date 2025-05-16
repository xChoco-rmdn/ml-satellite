# Cloud Nowcasting Project: Himawari Satellite Imagery Analysis

## Project Overview
- **Domain**: Meteorological Nowcasting
- **Focus**: Cloud Forecasting in Central Indonesia (NTB Region)
- **Satellite**: Himawari-8/9 Band 13 (10.4 µm Infrared)
- **Prediction Horizon**: 0-5 hours
- **Technology Stack**: Python 3.8+, TensorFlow, SatPy

## Project Structure
```
ml-satellite/
├── artifacts/              # Model artifacts and outputs
├── data/                  # Data directory
│   ├── raw/              # Raw Himawari satellite data
│   ├── processed/        # Preprocessed data
│   ├── train/           # Training dataset
│   └── test/            # Test dataset
├── logs/                 # Application logs
├── models/               # Saved model checkpoints
├── notebooks/           # Jupyter notebooks for analysis
└── src/                 # Source code
    ├── components/      # Core components
    │   ├── data_generator.py
    │   ├── data_ingestion.py
    │   ├── data_transformations.py
    │   └── model.py
    ├── pipeline/        # Processing pipelines
    │   ├── predict_pipeline.py
    │   └── train_pipeline.py
    ├── exception.py     # Custom exception handling
    ├── logger.py       # Logging configuration
    └── utils.py        # Utility functions
```

## Features
- Real-time cloud prediction using Himawari-8/9 satellite imagery
- ConvLSTM-based deep learning architecture
- Automated data ingestion and preprocessing pipeline
- Comprehensive model evaluation metrics
- Robust error handling and logging

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/xChoco-rmdn/ml-satellite
   cd ml-satellite
   ```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Pipeline
```python
from src.components.data_ingestion import DataIngestion
from src.components.data_transformations import DataTransformation

# Initialize data pipeline
data_ingestion = DataIngestion()
data_transformation = DataTransformation()

# Process data
raw_data = data_ingestion.initiate_data_ingestion()
processed_data = data_transformation.initiate_data_transformation(raw_data)
```

### Training
```python
from src.pipeline.train_pipeline import TrainPipeline

# Initialize and run training pipeline
pipeline = TrainPipeline()
metrics, history = pipeline.initiate_training()
```

### Prediction
```python
from src.pipeline.predict_pipeline import PredictionPipeline

# Make predictions
pipeline = PredictionPipeline()
predictions = pipeline.predict(input_sequence)
```

### Web Interface
Run the web application:
```bash
python application.py --model artifacts/best_model.h5 --port 5000
```

## Model Architecture
The project implements an enhanced ConvLSTM2D architecture with attention mechanisms:
- Input: Sequence of satellite images (Band 13 - 10.4 µm Infrared)
- Architecture: 
  - Multiple ConvLSTM2D layers with batch normalization
  - Attention blocks for feature refinement
  - U-Net style skip connections
  - TimeDistributed layers for temporal processing
- Output: Predicted cloud patterns for 0-5 hours ahead
- Loss Function: Combined loss (MSE + SSIM + Temporal Consistency)
- Optimizer: AdamW with cosine decay learning rate

## Performance Metrics

The model is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Structural Similarity Index (SSIM)
- Temporal Consistency Score

## Testing
Run the comprehensive test suite:
```bash
python src/test_enhanced_model.py
```

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments
- Japan Meteorological Agency (JMA) for Himawari-8/9 satellite data
- SatPy development team
- TensorFlow community
