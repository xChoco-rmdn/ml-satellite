
# Cloud Nowcasting Project: NTB Region Satellite-based Forecasting

## Project Overview
- **Domain**: Cloud Forecasting in Nusa Tenggara Barat (NTB), Indonesia
- **Forecast Horizon**: 0-4 hours ahead
- **Primary Data Source**: Satellite Imagery (Likely Himawari-8/9)
A machine learning project focused on nowcasting cloud forecasts using Himawari-8/9 satellite imagery, with emphasis on short-term (0-4 hours) cloud prediction for NTB region.

## Key Insights from Preliminary Research
1. Satellite Observation Potential
   - Himawari-8 weather satellite provides high-resolution real-time cloud observations
   - Infrared channel (~11 μm) is crucial for cloud identification

2. Machine Learning Approaches
   - Deep learning methods show promise in satellite image-based predictions
   - Potential techniques:
     * Recurrent Neural Networks (RNNs)
     * Dual-Attention Recurrent Neural Networks (DA-RNN)
     * Convolutional Neural Networks (CNNs)

## Unique Challenges in NTB Region
- Tropical climate characteristics
- Complex topographical variations
- Potential rapid cloud formation due to maritime environment

## Recommended Data Sources
1. Himawari-8 Satellite Imagery
2. Global Precipitation Measurement (GPM) IMERG data
3. Local meteorological station data for ground truth validation

## Potential Model Architectures
1. Spatio-temporal Deep Learning Models
2. Recurrent Neural Networks with Attention Mechanisms
3. Convolutional LSTM Networks

## Success Metrics
- Prediction Accuracy
- Temporal Resolution
- Spatial Precision
- Computational Efficiency

## Ethical and Practical Considerations
- Open-source model development
- Potential applications:
  * Local agriculture planning
  * Disaster preparedness
  * Climate research

## Project Structure
```
ml-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── components
|   |   ├── __init__.py
|   |   ├── data_ingestion.py
|   |   ├── data_transformations.py
|   |   └── model_trainer.py
│   |
│   ├── pipeline
|   |   ├── __init__.py
|   |   ├── predict_pipeline.py
|   |   └── train_pipeline.py
│   |
│   ├── __init__.py
│   ├── logger.py
│   ├── utils.py
│   └── exception.py
│
├── models/
│   └── saved_models/
│
├── venv
├── requirements.txt
├── .gitignore
├── README.md
└── setup.py

```