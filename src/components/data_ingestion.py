import os
import sys
from src.logger import logger
from src.exception import CustomException
import numpy as np
from datetime import datetime, timedelta
import satpy
from satpy import Scene
import glob
from dataclasses import dataclass
import re
import xarray as xr

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('data', 'train')
    test_data_path: str = os.path.join('data', 'test')
    raw_data_path: str = os.path.join('data', 'raw')
    processed_data_path: str = os.path.join('data', 'processed')
    channels: list = None
    time_range: tuple = None

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def extract_timestamp_from_filename(self, filename):
        """
        Extract timestamp from Himawari filename
        Expected format example: H09_B13_Indonesia_202504010010.nc
        """
        try:
            # Extract date and time from filename using regex
            pattern = r'H09_B13_Indonesia_(\d{8})(\d{4})'
            match = re.search(pattern, os.path.basename(filename))
            if match:
                date_str, time_str = match.groups()
                return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
            return None
        except Exception as e:
            logger.error(f"Error extracting timestamp from filename: {filename}")
            return None

    def group_files_by_timestamp(self, satellite_files):
        """
        Group files by their timestamp
        Returns: Dictionary with timestamp as key and list of files as value
        """
        files_by_time = {}
        for file in satellite_files:
            timestamp = self.extract_timestamp_from_filename(file)
            if timestamp:
                if timestamp not in files_by_time:
                    files_by_time[timestamp] = []
                files_by_time[timestamp].append(file)
        return files_by_time

    def process_single_timestamp(self, files, timestamp):
        """
        Process satellite data for a single timestamp
        """
        try:
            # Load the data directly using xarray since it's a NetCDF file
            
            # We expect only one file per timestamp
            if len(files) != 1:
                raise ValueError(f"Expected 1 file per timestamp, got {len(files)}")
                
            # Open the NetCDF file
            ds = xr.open_dataset(files[0])
            
            # Get the brightness temperature data
            ir_data = ds['tbb'].values
            
            # Format timestamp for filename
            timestamp_str = timestamp.strftime('%Y%m%d_%H%M')
            
            # Save the processed data
            processed_file_path = os.path.join(
                self.ingestion_config.processed_data_path,
                f"processed_data_{timestamp_str}.npy"
            )
            
            # Save processed data with timestamp
            np.save(processed_file_path, ir_data)
            
            return processed_file_path, ir_data
            
        except Exception as e:
            logger.error(f"Error processing data for timestamp {timestamp}")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self, satellite_files):
        """
        Process and save satellite data
        Args:
            satellite_files: List of satellite data file paths (should be for same channel/time period)
        Returns:
            Dictionary containing paths to processed, train and test data
        """
        try:
            logger.info("Started data ingestion")
            
            # Create directories if they don't exist
            os.makedirs(self.ingestion_config.raw_data_path, exist_ok=True)
            os.makedirs(self.ingestion_config.processed_data_path, exist_ok=True)
            os.makedirs(self.ingestion_config.train_data_path, exist_ok=True)
            os.makedirs(self.ingestion_config.test_data_path, exist_ok=True)
            
            # Group files by timestamp
            files_by_time = self.group_files_by_timestamp(satellite_files)
            
            if not files_by_time:
                raise ValueError("No valid files with timestamps found")
            
            # Sort timestamps
            timestamps = sorted(files_by_time.keys())
            
            # Process each timestamp and collect data
            processed_files = []
            all_data = []
            
            for timestamp in timestamps:
                files = files_by_time[timestamp]
                processed_file, data = self.process_single_timestamp(files, timestamp)
                processed_files.append(processed_file)
                all_data.append(data)
            
            # Convert to numpy array with shape (time, lat, lon)
            all_data = np.stack(all_data)
            
            # Split data temporally (80-20 split)
            split_idx = int(0.8 * len(timestamps))
            train_data = all_data[:split_idx]
            test_data = all_data[split_idx:]
            
            # Save train and test data with timestamp range
            start_train = timestamps[0].strftime('%Y%m%d_%H%M')
            end_train = timestamps[split_idx-1].strftime('%Y%m%d_%H%M')
            start_test = timestamps[split_idx].strftime('%Y%m%d_%H%M')
            end_test = timestamps[-1].strftime('%Y%m%d_%H%M')
            
            train_file_path = os.path.join(
                self.ingestion_config.train_data_path,
                f"train_data_{start_train}_to_{end_train}.npy"
            )
            test_file_path = os.path.join(
                self.ingestion_config.test_data_path,
                f"test_data_{start_test}_to_{end_test}.npy"
            )
            
            np.save(train_file_path, train_data)
            np.save(test_file_path, test_data)
            
            logger.info(f"Data ingestion completed. Files saved:")
            logger.info(f"Processed files: {len(processed_files)} files")
            logger.info(f"Train data: {train_file_path}")
            logger.info(f"Test data: {test_file_path}")
            
            return {
                "processed_files": processed_files,
                "train_file_path": train_file_path,
                "test_file_path": test_file_path
            }
            
        except Exception as e:
            logger.error("Error in data ingestion")
            raise CustomException(e, sys)
            
    def get_himawari_data(self, start_time, end_time, region=None):
        """
        Get Himawari-8 satellite data for specified time range and region
        """
        try:
            logger.info(f"Fetching Himawari data from {start_time} to {end_time}")
            
            # This is a placeholder for actual Himawari data fetching
            # In a real implementation, you would:
            # 1. Connect to Himawari data source
            # 2. Download data for specified time range and region
            # 3. Save raw data to raw_data_path
            # For now, we'll just log the intention, for future implementation
            
            logger.info("Note: This is a placeholder for Himawari data fetching")
            logger.info("Implement actual data fetching mechanism based on data source")
            
            return []
            
        except Exception as e:
            logger.error("Error in fetching Himawari data")
            raise CustomException(e, sys) 

# Test the data ingestion
if __name__ == "__main__":
    # Example usage with proper file pattern for Himawari-9 Band 13 data
    data_ingestion = DataIngestion()
    # Example files should be from Band 13 with different timestamps
    sample_files = glob.glob(os.path.join("data/raw", "H09_B13_Indonesia_*.nc"))
    if not sample_files:
        print("No files found matching the pattern. Please check if the files exist in data/raw directory.")
        sys.exit(1)
    data_ingestion.initiate_data_ingestion(satellite_files=sample_files)
    