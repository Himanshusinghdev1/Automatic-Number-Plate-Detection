# src/AutomaticNumberPlateDetection/components/data_ingestion.py
import os
import kagglehub
import shutil
from pathlib import Path
from AutomaticNumberPlateDetection import logger
from AutomaticNumberPlateDetection.utils.common import get_size
from AutomaticNumberPlateDetection.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """Download dataset using KaggleHub"""
        try:
            logger.info(f"📥 Downloading dataset: {self.config.dataset_name}")
            
            # Download using KaggleHub
            downloaded_path = kagglehub.dataset_download(self.config.dataset_name)
            logger.info(f"✅ Dataset downloaded to: {downloaded_path}")
            
            # Copy to project structure
            if os.path.exists(self.config.unzip_dir):
                shutil.rmtree(self.config.unzip_dir)
            
            shutil.copytree(downloaded_path, self.config.unzip_dir)
            logger.info(f"📁 Dataset copied to: {self.config.unzip_dir}")
            
            return self.config.unzip_dir
            
        except Exception as e:
            logger.error(f"❌ Failed to download dataset: {str(e)}")
            raise e

    def extract_zip_file(self):
        """Extract zip file if needed (KaggleHub handles this automatically)"""
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        logger.info(f"📦 Dataset ready at: {unzip_path}")
