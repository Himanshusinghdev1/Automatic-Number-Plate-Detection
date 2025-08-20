import os
import yaml
from pathlib import Path
from typing import Any
import logging

# Setup logger
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')
logger = logging.getLogger(__name__)

def read_yaml(path_to_yaml: Path) -> dict:
    """Read YAML file and return its content"""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return content
    except Exception as e:
        logger.error(f"Error reading YAML file {path_to_yaml}: {e}")
        raise e

def create_directories(path_to_directories: list, verbose=True):
    """Create list of directories"""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

def get_size(path: Path) -> str:
    """Get size of file in KB"""
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"
