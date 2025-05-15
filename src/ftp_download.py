import os
import ftplib
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HimawariDataFetcher:
    def __init__(self):
        # FTP server details
        self.host = "2020.90.199.64"
        self.user = "97260"
        self.password = "S2@Vt8Y9Z#"
        self.base_path = "/himawari6/netcdf/Indonesia/2025/"
        
        # Local storage
        self.local_base_path = Path("data/raw/himawari")
        self.local_base_path.mkdir(parents=True, exist_ok=True)

    def connect(self):
        """Establish FTP connection"""
        try:
            logger.info(f"Connecting to FTP server {self.host}")
            self.ftp = ftplib.FTP(self.host)
            self.ftp.login(self.user, self.password)
            logger.info("Successfully connected to FTP server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to FTP server: {str(e)}")
            return False

    def disconnect(self):
        """Close FTP connection"""
        try:
            self.ftp.quit()
            logger.info("Disconnected from FTP server")
        except:
            pass

    def download_data(self, start_date, end_date=None):
        """
        Download Himawari data for a specific date range
        
        Args:
            start_date (datetime): Start date
            end_date (datetime, optional): End date. Defaults to start_date
        """
        if end_date is None:
            end_date = start_date

        if not self.connect():
            return

        try:
            current_date = start_date
            while current_date <= end_date:
                self._download_date_data(current_date)
                current_date += timedelta(days=1)
        finally:
            self.disconnect()

    def _download_date_data(self, date):
        """Download data for a specific date"""
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        
        remote_path = f"{self.base_path}/{year}/{month}/{day}"
        local_path = self.local_base_path / year / month / day
        local_path.mkdir(parents=True, exist_ok=True)

        try:
            # List files in remote directory
            file_list = []
            self.ftp.retrlines(f'LIST {remote_path}', file_list.append)
            
            for file_info in file_list:
                # Parse file information
                parts = file_info.split()
                filename = parts[-1]
                
                if filename.endswith('.nc'):  # Download only NetCDF files
                    remote_file = f"{remote_path}/{filename}"
                    local_file = local_path / filename
                    
                    if not local_file.exists():
                        logger.info(f"Downloading {filename}")
                        with open(local_file, 'wb') as f:
                            self.ftp.retrbinary(f'RETR {remote_file}', f.write)
                        logger.info(f"Successfully downloaded {filename}")
                    else:
                        logger.info(f"File {filename} already exists locally")

        except Exception as e:
            logger.error(f"Error downloading data for {date.strftime('%Y-%m-%d')}: {str(e)}")

def main():
    # Example usage
    fetcher = HimawariDataFetcher()
    
    # Download data for a specific date range
    start_date = datetime(2025, 1, 1)  # Adjust as needed
    end_date = datetime(2025, 1, 2)    # Adjust as needed
    
    fetcher.download_data(start_date, end_date)

if __name__ == "__main__":
    main()