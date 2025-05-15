from flask import Flask, request, jsonify, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import pandas as pd
import numpy as np  
from sklearn.preprocessing import StandardScaler
import os

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')



