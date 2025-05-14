# Cloud Nowcasting Project for NTB Region

## Overview
This project implements a machine learning-based cloud nowcasting system for the Nusa Tenggara Barat (NTB) region in Indonesia. The system uses Himawari-8/9 satellite imagery to predict cloud patterns and movement for the next 0-4 hours.

## Features
- Real-time cloud prediction using satellite imagery
- ConvLSTM-based deep learning model
- 4-hour prediction horizon
- Cloud mask generation
- Automated data pipeline for satellite data processing

## Project Structure
```
ml-project/
│
├── data/                    # Data directory
│   ├── raw/                # Raw satellite data
│   └── processed/          # Processed data
│
├── notebooks/              # Jupyter notebooks
│   └── exploration.ipynb   # Data exploration notebook
│
├── src/                    # Source code
│   ├── components/         # Core components
│   ├── pipeline/          # Training and prediction pipelines
│   ├── logger.py          # Logging configuration
│   ├── utils.py           # Utility functions
│   └── exception.py       # Custom exception handling
│
├── models/                 # Saved models
│   └── saved_models/      # Trained model checkpoints
│
├── requirements.txt        # Project dependencies
├── setup.py               # Package setup file
└── README.md              # Project documentation
```

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
1. Training the model:
   ```python
   from src.pipeline.train_pipeline import TrainPipeline
   
   pipeline = TrainPipeline()
   model, metrics = pipeline.initiate_training()
   ```

2. Making predictions:
   ```python
   from src.pipeline.predict_pipeline import PredictionPipeline
   
   pipeline = PredictionPipeline()
   predictions = pipeline.predict()
   ```

## Model Architecture
The project uses a Convolutional LSTM (ConvLSTM) architecture, which combines the spatial feature extraction capabilities of CNNs with the temporal modeling of LSTMs. The model consists of:
- Two ConvLSTM2D layers with batch normalization
- Final convolutional layer for prediction
- MSE loss function and Adam optimizer

## Data
The system uses Himawari-8/9 satellite data, specifically:
- Infrared channel (~11 μm) for cloud identification
- 10-minute temporal resolution
- Spatial coverage of the NTB region

## Performance Metrics
The model is evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Himawari-8/9 satellite data provided by the Japan Meteorological Agency (JMA)
- SatPy library for satellite data processing
- TensorFlow for deep learning implementation 
