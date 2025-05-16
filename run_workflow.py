#!/usr/bin/env python

import os
import sys
import argparse
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/workflow.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment for the workflow"""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('artifacts/evaluation', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models/saved_models', exist_ok=True)
    
    logger.info("Environment setup complete")

def check_data():
    """Check if data is available, create synthetic data if needed"""
    from src.components.data_ingestion import DataIngestion
    
    try:
        # Check or create sample data
        if not os.path.exists("data/train") or not os.path.exists("data/test"):
            logger.info("Data not found. Creating synthetic data...")
            
            # Import function from main notebook
            sys.path.append(os.getcwd())
            from notebooks.Satellite_Cloud_Nowcasting_Project_Notebook import check_sample_data
            
            # Create synthetic data
            check_sample_data()
            
        logger.info("Data check complete")
        return True
    except Exception as e:
        logger.error(f"Error checking/creating data: {e}")
        return False

def train_model():
    """Train the model using the training pipeline"""
    try:
        logger.info("Starting model training...")
        
        # Run the training pipeline as a subprocess
        cmd = [sys.executable, "src/pipeline/train_pipeline.py"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Training failed with error: {stderr.decode()}")
            return False
        
        logger.info("Model training completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return False

def evaluate_model():
    """Evaluate the trained model"""
    try:
        logger.info("Starting model evaluation...")
        
        # Run the evaluation script as a subprocess
        cmd = [
            sys.executable, 
            "src/pipeline/evaluate_model.py",
            "--model", "artifacts/best_model.h5",
            "--test_data", "data/test/test_data_20250425_0110_to_20250501_0000.npy",
            "--output_dir", "artifacts/evaluation"
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Evaluation failed with error: {stderr.decode()}")
            return False
        
        logger.info("Model evaluation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        return False

def start_application(port=5000, debug=False):
    """Start the web application for predictions"""
    try:
        logger.info(f"Starting web application on port {port}...")
        
        # Build the command
        cmd = [
            sys.executable,
            "application.py",
            "--model", "artifacts/best_model.h5",
            "--port", str(port)
        ]
        
        if debug:
            cmd.append("--debug")
        
        # Run the application as a subprocess
        process = subprocess.Popen(cmd)
        
        # Wait for the application to start
        time.sleep(2)
        
        logger.info(f"Web application started at http://localhost:{port}")
        
        return process
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        return None

def run_workflow(train=True, evaluate=True, deploy=True, port=5000, debug=False):
    """Run the complete workflow"""
    try:
        # Setup the environment
        setup_environment()
        
        # Check data
        if not check_data():
            logger.error("Data check failed, aborting workflow")
            return False
        
        # Train model if requested
        if train:
            if not train_model():
                logger.error("Model training failed, aborting workflow")
                return False
        
        # Evaluate model if requested
        if evaluate:
            if not evaluate_model():
                logger.error("Model evaluation failed, aborting workflow")
                return False
        
        # Deploy application if requested
        if deploy:
            app_process = start_application(port, debug)
            if app_process is None:
                logger.error("Application startup failed, aborting workflow")
                return False
            
            # Keep the script running while the application is running
            logger.info(f"Workflow completed successfully. Application running at http://localhost:{port}")
            logger.info("Press Ctrl+C to stop the application and exit")
            
            try:
                app_process.wait()
            except KeyboardInterrupt:
                logger.info("Stopping application...")
                app_process.terminate()
                logger.info("Application stopped")
        else:
            logger.info("Workflow completed successfully (without deployment)")
        
        return True
    except Exception as e:
        logger.error(f"Error in workflow: {e}")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the complete cloud nowcasting workflow')
    parser.add_argument('--skip-train', action='store_true', help='Skip the training step')
    parser.add_argument('--skip-eval', action='store_true', help='Skip the evaluation step')
    parser.add_argument('--skip-deploy', action='store_true', help='Skip the deployment step')
    parser.add_argument('--port', type=int, default=5000, help='Port for the web application')
    parser.add_argument('--debug', action='store_true', help='Run the application in debug mode')
    
    args = parser.parse_args()
    
    # Run the workflow
    run_workflow(
        train=not args.skip_train,
        evaluate=not args.skip_eval,
        deploy=not args.skip_deploy,
        port=args.port,
        debug=args.debug
    ) 