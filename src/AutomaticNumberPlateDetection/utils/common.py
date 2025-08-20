import os
import json
import yaml
import joblib
import base64
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from pathlib import Path
from typing import Any
from AutomaticNumberPlateDetection import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If YAML file is empty.
        Exception: For any other error.

    Returns:
        ConfigBox: ConfigBox object containing YAML data.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully from: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        logger.exception(e)
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """
    Creates a list of directories.

    Args:
        path_to_directories (list): List of paths for directories.
        verbose (bool, optional): If True, logs directory creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves a dictionary as a JSON file.

    Args:
        path (Path): Path to save JSON.
        data (dict): Data to save.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"JSON file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads data from a JSON file.

    Args:
        path (Path): Path to JSON file.

    Returns:
        ConfigBox: Data from JSON file as ConfigBox.
    """
    with open(path, "r") as f:
        content = json.load(f)

    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Saves an object as a binary file using joblib.

    Args:
        data (Any): Object to save.
        path (Path): Path to save file.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads an object from a binary file.

    Args:
        path (Path): Path to binary file.

    Returns:
        Any: Loaded object.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns size of a file in KB.

    Args:
        path (Path): Path to file.

    Returns:
        str: File size in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring: str, fileName: str):
    """
    Decodes a base64 image string and saves it as a JPEG file.

    Args:
        imgstring (str): Base64 encoded image string.
        fileName (str): File path to save image.
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)
    logger.info(f"Image decoded and saved at: {fileName}")


def encodeImageIntoBase64(imagePath: str) -> bytes:
    """
    Encodes an image file into base64 format.

    Args:
        imagePath (str): Path to image file.

    Returns:
        bytes: Base64 encoded image.
    """
    with open(imagePath, "rb") as f:
        return base64.b64encode(f.read())
