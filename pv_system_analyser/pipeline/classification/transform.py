from pathlib import Path
from os.path import join

from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from pv_system_analyser.io import _read_json_annotation
from pv_system_analyser.pipeline.validation import check_dir

load_dotenv()


@check_dir
def read_dataset_metadata(data_dir: str) -> pd.DataFrame:
    """
    Read the dataset metadata from a JSON file in the specified data directory.
    
    Args:
        data_dir (str): The directory path where the dataset metadata file is located.
    
    Returns:
        pd.DataFrame: The dataset metadata as a pandas DataFrame.
    """
    metadata_path = next(Path(data_dir).glob('*.json'))
    annotation = _read_json_annotation(metadata_path, dataset_type='classification')
    return annotation

def remove_unwanted_classes(data: pd.DataFrame, unwanted_classes: list[str]) -> pd.DataFrame:
    """
    Remove the classes that will not be used in the analysis.
    
    Args:
        data (pd.DataFrame): The dataset containing the classes to be removed.
        unwanted_classes (list[str]): The list of class labels to be removed from the dataset.
    
    Returns:
        pd.DataFrame: The dataset with the unwanted classes removed.
    """
    return data.query(f'anomaly_class != {*unwanted_classes,}')

def fix_image_paths(data: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """
    Fix the image file paths in the dataset by joining the data directory path to the relative image file paths.
    
    Args:
        data (pd.DataFrame): The dataset containing the image file paths.
        data_dir (str): The directory path where the images are stored.
    
    Returns:
        pd.DataFrame: The dataset with the fixed image file paths.
    """
    data['image_filepath'] = data.image_filepath.apply(
        lambda image_path: join(data_dir, image_path)
    )
    return data

def split_dataset(
    dataset: pd.DataFrame, split_ratio: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and test sets, stratifying by the `anomaly_class` column.
    
    Args:
        dataset (pd.DataFrame): The dataset to be split.
        split_ratio (float, optional): The ratio of the dataset to be used for the training set. Defaults to 0.7.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The training and test sets.
    """
    train_data, test_data = train_test_split(
        dataset, 
        train_size=split_ratio, 
        stratify=dataset.anomaly_class
    )
    return train_data, test_data


if '__main__' == __name__ :
    
    data_dir = ''
    
    metadata_df = read_dataset_metadata(data_dir)
    metadata_df = remove_unwanted_classes(metadata_df, [])
    metadata_df = fix_image_paths(metadata_df, )
    split_dataset(metadata_df)
    ...