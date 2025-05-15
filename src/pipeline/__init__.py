"""
Pipeline module for the Cloud Nowcasting project.
Contains training and prediction pipelines.
"""

from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

__all__ = ['TrainPipeline', 'PredictionPipeline'] 