from pathlib import Path

import pandas as pd
import cv2
from tqdm import tqdm

from utils.dataset_splitter import ClassificationSplitter


def split_classification_datasets(datasets_folder_path: str, sizes: dict[str, float]):
    
    datasets_folder_path = Path(datasets_folder_path)
    
    # Iterate by many classification datasets exists
    datasets_paths = list(datasets_folder_path.iterdir())
    for dataset_path in tqdm(datasets_paths, desc='Processing datasets'):
        if 'yolo' in dataset_path.as_posix():
            
            # Organize dataset to be used by classification splitter
            images_paths = []
            for cls_folder in dataset_path.iterdir():
                for img_path in cls_folder.iterdir():
                    images_paths.append(img_path)
            images_paths = pd.Series(images_paths, name='image_filepath')
            
            defects = []
            for cls_folder in dataset_path.iterdir():
                defect = [cls_folder.name] * len(list(cls_folder.iterdir()))
                defects.append(defect)
            defects = [anomaly for defect in defects for anomaly in defect]
            defects = pd.Series(defects, name='anomaly_class')
            
            dataset = pd.concat([images_paths, defects], axis=1)
        
            
        # Split and save the dataset
        splitter = ClassificationSplitter(dataset)
        splitted_dataset = splitter('anomaly_class', **sizes, stratify=True)
        
        splitted_dataset = tqdm(splitted_dataset, desc='Processing splitted datasets',  leave=False)        
        for index, dataset in enumerate(splitted_dataset):
            
            match index:
                case 0:
                    dataset_name = 'train'
                case 1:
                    dataset_name = 'val'
                case _:
                    dataset_name = 'test'
            
            images_paths = tqdm(dataset.image_filepath.values, desc='Saving images', leave=False)
            for image_path in images_paths:
                image = cv2.imread(image_path.as_posix())
                
                name = Path(image_path.parent.name, image_path.name)
                image_path = image_path.parent.with_name(dataset_name)
                image_path = image_path.joinpath(name).as_posix()
                image_path = Path(image_path.replace('intermediate', 'processed'))
                if not image_path.parent.exists():
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(image_path.as_posix(), image)
            
    
if __name__ == '__main__':
    sizes = {'train_size': 0.7, 'val_size': 0.2, 'test_size': 0.1}
    datasets_path = 'data/intermediate/yolo/classification'
    split_classification_datasets(datasets_path, sizes)
    