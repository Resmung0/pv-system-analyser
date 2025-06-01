#from functools import lru_cache
import pickle
import json
import shutil
from pathlib import Path
import logging

import numpy as np
from skimage import draw
import pandas as pd
from tqdm import tqdm
import cv2

from src.scripts.image.image_utils import read_thermogram


class YoloSegmentationConverter:
    def __init__(self, datasets_path: str, mask_shape: tuple[int, int] = (512, 640)) -> None:
        self.datasets_path = Path(datasets_path)
        self.mask_shape = mask_shape
    
    @staticmethod
    def convert_annotation(annotation_path: Path, mask_shape: tuple[int, int]) -> list[str]:
        with annotation_path.open('r', encoding='utf-8') as annotation_file:
            annotation = json.load(annotation_file)
        
        objs = []
        for instance in annotation['instances']:
            # Get object normalized bounding coordinates
            points = np.array(
                [[coords["y"], coords["x"]] for coords in instance["corners"]]
            )
            rr, cc = draw.polygon_perimeter(points[:,0], points[:,1], shape=mask_shape)
            points = np.array(list(zip(rr, cc)))
            points = (points / mask_shape).flatten()

            # Transform points from array to string
            points = np.array2string(points, formatter={'float': str})    
            for element in ['[', ']', '\n']:
                points = points.replace(element, '')
            
            # Add class index, e.g 1 = defective, 0 = normal
            # roi_class = '1 ' if instance["defected_module"] else '0 '
            
            # Add class index, e.g 0 = module
            objs.append("0 " + points.strip())
        return objs
    
    def convert_dataset_PST(self, dataset_path: Path):
        subdataset_path = list(dataset_path.iterdir())
        # Convert dataset to yolo format and save
        for data_path in tqdm(subdataset_path, desc='Processing subdatasets', leave=False):
            annotations_paths = list(
                data_path.joinpath('annotations').glob("*.json")
            )
            
            for annotation_path in tqdm(annotations_paths, desc="Converting annotations", leave=False):
                annotation = self.convert_annotation(annotation_path, self.mask_shape)
                
                # Create yolo segmentation folder    
                new_annotation_path = self.datasets_path.parent.with_name('intermediate')
                
                new_annotation_path = new_annotation_path.joinpath('yolo/segmentation', dataset_path.name)
                new_annotation_path = new_annotation_path.joinpath(data_path.name, 'annotations')
                if not new_annotation_path.exists():
                    new_annotation_path.mkdir(parents=True, exist_ok=True)
                
                new_annotation_path = new_annotation_path.joinpath(annotation_path.stem).with_suffix('.txt')
                
                # Save annotation
                with new_annotation_path.open('w', encoding='utf-8') as annotation_file:
                    annotation_file.writelines(annotation)
                
                # Save images
                paths = []
                for image_path in [annotation_path, new_annotation_path]:
                    image_path = image_path.parent.with_name('images')
                    if not image_path.exists():
                        image_path.mkdir(parents=True, exist_ok=True)
                    image_path = image_path.joinpath(annotation_path.stem)
                    paths.append(image_path.with_suffix('.jpg'))
                
                image = read_thermogram(paths[0].as_posix())
                cv2.imwrite(paths[1], image.render(palette='grayscale'))
    
    def convert_dataset_PSOAMI(self, dataset_path: Path):
        calibration_path, data_paths = list(dataset_path.iterdir())
                                
        # Get calibration files to remove image distortion
        mapx_path, mapy_path = list(calibration_path.glob('IR/*.pkl'))
        with open(mapx_path, 'rb') as mapx_file, open(mapy_path, 'rb') as mapy_file:
            mapx, mapy = pickle.load(mapx_file), pickle.load(mapy_file)
        
        for data_path in data_paths.iterdir():    
            # Save images
            images_paths = list(data_path.joinpath('thermal images').iterdir())
            for image_path in tqdm(images_paths, desc='Processing images', leave=False):
                image_path = image_path.as_posix()
                
                # Read thermogram
                image = read_thermogram(image_path).render(palette='grayscale')
                
                # Correct distortion
                image = cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                image_path = image_path.replace('raw', 'intermediate/yolo')
                image_path = Path(image_path.replace('thermal ', ''))
                if not image_path.parent.exists():
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(image_path.as_posix(), image)
            
            # Save annotation
            annotations_paths = list(data_path.joinpath('annotations').glob("*.json"))
            for annotation_path in tqdm(annotations_paths, desc="Converting annotations", leave=False):
                annotation = self.convert_annotation(annotation_path, self.mask_shape)
                new_annotation_path = annotation_path.as_posix().replace('raw', 'intermediate/yolo')
                new_annotation_path = Path(new_annotation_path.replace('datasets', ''))
                with new_annotation_path.open('w', encoding='utf-8') as annotation_file:
                    annotation_file.writelines(annotation)


    #@lru_cache(maxsize=None)
    def __call__(self) -> None:
        """Converts instance segmentation annotation to YOLO format."""
        
        datasets_paths = list(self.datasets_path.iterdir())
        
        for dataset_path in tqdm(datasets_paths, desc="Converting datasets"):
            print(f'\nHandling {dataset_path.name} dataset')
            
            match dataset_path.name:    
                case 'photovoltaic-system-thermography':
                    self.convert_dataset_PST(dataset_path)
                
                case 'photovoltaic-system-o-and-m-inspection':
                    self.convert_dataset_PSOAMI(dataset_path)



class YoloClassificationConverter:
    def __init__(self, datasets_path: str) -> None:
        self.datasets_path = Path(datasets_path)
    
    @staticmethod
    def convert_annotation(annotation_path: Path) -> pd.DataFrame:
        # Convert dataset to yolo format and save
        dataset = next(annotation_path.glob('*.json'))
        dataset = pd.read_json(dataset, orient="index").sort_index()
        return dataset.assign(
            image_filepath=dataset.image_filepath.str.replace('images/', '')
        )
    
    def __call__(self) -> None:
        """Converts classification datasets to YOLO format."""

        datasets_paths = list(self.datasets_path.iterdir())
        for dataset_path in tqdm(datasets_paths, desc="Converting datasets..."):
            # Create yolo classification folder
            new_dataset_path = dataset_path.parent.parent.with_name('intermediate')
            new_dataset_path = Path(new_dataset_path, 'yolo/classification')
            new_dataset_path = Path(new_dataset_path, dataset_path.name)
            if not new_dataset_path.exists():
                new_dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Read dataset metadata
            dataset = next(dataset_path.glob("*.json"))
            dataset = pd.read_json(dataset, orient="index").sort_index()
            
            # Convert dataset to yolo format and save
            dataset = dataset.assign(
                image_filepath=dataset.image_filepath.str.replace('images/', '')
            )        
            
            anomalies = dataset.anomaly_class.unique()
            for anomaly in tqdm(anomalies, desc='Processing folders', leave=False):
                anomaly_df = dataset[dataset.anomaly_class == anomaly]
                
                anomaly_class_path = Path(new_dataset_path, anomaly)
                anomaly_class_path.mkdir(exist_ok=True)            
                anomaly_df = anomaly_df.assign(
                    image_filepath=anomaly_df.image_filepath.map(lambda x: Path(anomaly_class_path, x))
                )
                
                image_filepath = anomaly_df.image_filepath.values
                for image_filepath in tqdm(image_filepath, desc='Saving images', leave=False):
                    old_image_filepath = Path(dataset_path, 'images', image_filepath.name)
                    shutil.copy(old_image_filepath, image_filepath)


if __name__ == '__main__':
    logging.info('Starting YOLO classification dataset converter')
    classification_converter = YoloClassificationConverter('data/raw/classification')
    classification_converter()
    logging.info('Classification dataset converted to YOLO format')
    
    logging.info('Starting YOLO segmnetation dataset converter')
    segmentation_converter = YoloSegmentationConverter('data/raw/segmentation', (512, 640))
    segmentation_converter()
    logging.info('Segmentation dataset converted to YOLO format')
    