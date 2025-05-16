import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.colors import Normalize
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
model_path = 'artifacts/final_model.h5'
sequence_length = 10

# Function to load the model
def load_trained_model(model_path):
    """Load the trained model"""
    try:
        model = load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

# Route for the home page
@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

# Function to generate a prediction visualization
def generate_prediction_plot(input_sequence, predictions):
    """Generate a visualization of the predictions"""
    # Create a figure to hold the plots
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle('Cloud Nowcasting Prediction', fontsize=16)
    
    # Get the last frame of input sequence for reference
    last_input = input_sequence[0, -1, :, :, 0]
    
    # Set common colormap and normalization
    vmin = min(np.min(input_sequence), np.min(predictions))
    vmax = max(np.max(input_sequence), np.max(predictions))
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot the last input frame (top row)
    for i in range(6):
        axes[0, i].imshow(last_input, cmap='viridis', norm=norm)
        axes[0, i].set_title(f'Last Input Frame')
        axes[0, i].axis('off')
    
    # Plot the predictions (bottom row)
    for i in range(6):
        axes[1, i].imshow(predictions[0, i, :, :, 0], cmap='viridis', norm=norm)
        axes[1, i].set_title(f'Prediction (t+{i+1})')
        axes[1, i].axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cbar_ax)
    cb.set_label('Brightness Temperature (K)')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return plot_data

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    """Generate predictions from input sequence"""
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Load the input sequence from the file
        input_data = np.load(file)
        
        # Check input shape and preprocess if needed
        if len(input_data.shape) == 3:  # (time_steps, height, width)
            # Reshape to (1, time_steps, height, width, 1)
            input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1], input_data.shape[2], 1)
            
        # Make sure we have the right sequence length
        if input_data.shape[1] != sequence_length:
            return jsonify({'error': f'Input sequence must have {sequence_length} time steps'}), 400
        
        # Generate prediction
        logger.info("Generating prediction...")
        predictions = model.predict(input_data)
        logger.info(f"Prediction shape: {predictions.shape}")
        
        # Generate visualization
        plot_data = generate_prediction_plot(input_data, predictions)
        
        # Create response
        response = {
            'success': True,
            'plot': plot_data,
            'message': 'Prediction generated successfully'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Route for API-based predictions (receiving and returning numpy arrays)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        # Get the input data as JSON
        input_json = request.get_json()
        
        if not input_json or 'data' not in input_json:
            return jsonify({'error': 'No data provided'}), 400
            
        # Convert the data from JSON to numpy array
        input_data = np.array(input_json['data'])
        
        # Check input shape and preprocess if needed
        if len(input_data.shape) == 3:  # (time_steps, height, width)
            # Reshape to (1, time_steps, height, width, 1)
            input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1], input_data.shape[2], 1)
            
        # Make sure we have the right sequence length
        if input_data.shape[1] != sequence_length:
            return jsonify({'error': f'Input sequence must have {sequence_length} time steps'}), 400
        
        # Generate prediction
        logger.info("Generating prediction...")
        predictions = model.predict(input_data)
        logger.info(f"Prediction shape: {predictions.shape}")
        
        # Convert predictions to list for JSON response
        predictions_list = predictions[0].tolist()
        
        # Create response
        response = {
            'success': True,
            'predictions': predictions_list,
            'shape': list(predictions[0].shape)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during API prediction: {e}")
        return jsonify({'error': str(e)}), 500

def create_template_directory():
    """Create template directory and basic HTML template if they don't exist"""
    os.makedirs('templates', exist_ok=True)
    
    # Create a basic HTML template
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cloud Nowcasting Prediction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
            }
            .container {
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 20px;
                margin-top: 20px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #2980b9;
            }
            #result {
                margin-top: 20px;
                display: none;
            }
            #prediction-image {
                max-width: 100%;
                margin-top: 10px;
            }
            .error {
                color: #e74c3c;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <h1>Cloud Nowcasting Prediction System</h1>
        
        <div class="container">
            <h2>Upload Input Sequence</h2>
            <p>Upload a Numpy file (.npy) containing a sequence of 10 consecutive satellite images.</p>
            
            <form id="prediction-form" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="file">Select file:</label>
                    <input type="file" id="file" name="file" accept=".npy" required>
                </div>
                
                <button type="submit">Generate Prediction</button>
            </form>
            
            <div id="result">
                <h2>Prediction Result</h2>
                <p id="result-message"></p>
                <div id="error-message" class="error"></div>
                <img id="prediction-image" src="" alt="Prediction visualization">
            </div>
        </div>
        
        <script>
            document.getElementById('prediction-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const form = document.getElementById('prediction-form');
                const formData = new FormData(form);
                const resultDiv = document.getElementById('result');
                const errorMessage = document.getElementById('error-message');
                const resultMessage = document.getElementById('result-message');
                const predictionImage = document.getElementById('prediction-image');
                
                errorMessage.textContent = '';
                resultMessage.textContent = 'Processing...';
                resultDiv.style.display = 'block';
                predictionImage.style.display = 'none';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        resultMessage.textContent = data.message;
                        predictionImage.src = 'data:image/png;base64,' + data.plot;
                        predictionImage.style.display = 'block';
                    } else {
                        errorMessage.textContent = data.error || 'An error occurred';
                        predictionImage.style.display = 'none';
                    }
                } catch (error) {
                    errorMessage.textContent = 'An error occurred during prediction';
                    predictionImage.style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)
    
    logger.info("Created templates directory and index.html")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cloud Nowcasting Prediction Application')
    parser.add_argument('--model', type=str, default='artifacts/best_model.h5',
                        help='Path to the trained model file')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the application on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Update model path from arguments
    model_path = args.model
    
    # Create templates directory and HTML file
    create_template_directory()
    
    # Load the model
    model = load_trained_model(model_path)
    
    # Run the application
    logger.info(f"Starting application on port {args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)



