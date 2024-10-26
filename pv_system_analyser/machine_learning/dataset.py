from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Transform


@dataclass(slots=True)
class ClassificationDataset(Dataset):
    """Dataset that process images and labels to be used in classification
    task.
    
    Args:
        metadata (pd.DataFrame): Dataset metadata.
        transform (Compose): Transformations to apply to the images. Defaults to None.
        target_transform (Transform): Transformations to apply to the labels. Defaults to None.
    """
    metadata: pd.DataFrame
    transform: Compose | None = field(default_factory=lambda: None)
    target_transform: Transform | None = field(default_factory=lambda: None)

    def __len__(self) -> int:
        """
        Method to know the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return self.metadata.shape[0]
    
    def __getitem__(self, index: int) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """
        Retrieves an image and it's corresponding label from the dataset at the given index.
        
        Args:
            index (int): The index of the image and label to retrieve.
        
        Returns:
            tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]: A tuple containing the image and its label.
        """
        image = self._read_image(index)
        if self.transform:
            image = self.transform(image)
        
        label = self._read_label(index)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
    def _read_image(self, index: int) -> npt.NDArray:
        """
        Reads the image at the specified index from the dataset metadata.
        
        Args:
            index (int): The index of the image to read.
        
        Returns:
            npt.NDArray: The image data as a NumPy array.
        """
        return io.imread(self.metadata.loc[index, "image_filepath"])
    
    def _read_label(self, index: int) -> str:
        """
        Reads the anomaly class label for the image at the specified index.
        
        Args:
            index (int): The index of the image to retrieve the label for.
        
        Returns:
            str: The anomaly class label for the image.
        """
        return self.metadata.loc[index, "anomaly_class"]
