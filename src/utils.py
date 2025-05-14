import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logger
import dill
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yaml

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
            
    except Exception as e:
        raise CustomException(e, sys)

def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(y_true, y_pred):
    try:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse
        }
        
    except Exception as e:
        raise CustomException(e, sys)

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
            
    except Exception as e:
        raise CustomException(e, sys) 