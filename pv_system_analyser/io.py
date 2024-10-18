"""File that defines functions that read different kind of data."""
import json
from pathlib import Path

import pandas as pd
from flyr import FlyrThermogram, unpack

from .image_processing.thermogram import TIFFThermogram
from .image_processing.masking import Mask
from .parser.utils import choose_parser


def _read_json_annotation(
    path: Path, dataset_type: str
) -> dict[str, str | int] | pd.DataFrame:
    """
    Function for reading JSON annotation file.
    
    Args:
        path: Path to JSON file.
        dataset_type: Type of dataset.

    Returns:
        Data from JSON file.
    """
    with path.open("r", encoding="utf-8") as file:
        match dataset_type:
            case 'segmentation':
                data = json.load(file)
            case 'classification':
                data = pd.read_json(file, orient='index').sort_index()
    return data

def read_mask(file_path: str) -> Mask:
    """Function for mask creation from JSON annotation file."""
    file_path = Path(file_path)
    dataset = [
        parent for parent in file_path.parents if parent.name == 'datasets'
    ]
    data = _read_json_annotation(file_path, dataset_type='segmentation')
    parser = choose_parser(dataset[0], data)
    return Mask(parser.masks)

def read_thermogram(
    file_path: str, tiff_info: dict[str, str] | None = None
) -> FlyrThermogram | TIFFThermogram:
    """Read thermogram data from RJPG and TIFF files.

    Args:
        file_path (str): Thermogram image file path.
        tiff_info (dict[str, str] | None): Informations needed to read tiff thermograms,
        like metadata and optical image paths. Default to None.

    Returns:
        FlyrThermogram | TIFFThermogram: Thermogram data.
    """
    file_format = Path(file_path).suffix.strip('.')
    if file_format in ("jpg", "JPG"):
        thermogram = unpack(file_path)
    elif file_format in ("tif", "tiff"):
        if tiff_info is None:
            tiff_info = {
                "metadata_path": None,
                "optical_path": None,
                "method": "thermomap",
            }
        thermogram = TIFFThermogram(file_path, **tiff_info)
    return thermogram

