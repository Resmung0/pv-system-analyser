from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from pv_system_analyser.io import read_mask, read_thermogram

from .preprocessing import ToOneHot


@dataclass(slots=True)
class ClassificationDataset(Dataset):
    """Dataset that process images and labels to be used in classification
    task.

    Args:
        metadata (pd.DataFrame): Dataset metadata.
        transform (Compose): Transformations to apply to the images. Defaults to None.
        target_transform (ToOneHot): Transformation
        to apply to the labels. Defaults to None.
    """

    metadata: pd.DataFrame
    transform: Compose | None = field(default_factory=lambda: None)
    target_transform: ToOneHot | None = field(default_factory=lambda: None)

    def __len__(self) -> int:
        """
        Method to know the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return self.metadata.shape[0]

    def __getitem__(self, index: int) -> tuple[npt.NDArray[np.uint8], str]:
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

    def _read_image(self, index: int) -> npt.NDArray[np.uint8]:
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


class InstanceSegmentationDataset(Dataset):
    """
    Dataset that process thermograms and masks to be used in instance segmentation task.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Compose | None = None,
    ) -> None:
        """
        Args:
            data_dir (str): Data directory path.
            transform (Compose): Transformations to apply to images and masks. Defaults to None.
        """
        # Get all thermograms (RJPG or TIFF)
        self.data = Path(data_dir, "thermograms").iterdir()
        self.data = pd.DataFrame(
            [str(value) for value in sorted(list(self.data))],
            columns=["image_filepath"],
        )

        # Get all thermograms masks
        self.data["mask_filepath"] = self.data["image_filepath"].str.replace(
            "thermograms", "annotations"
        )
        self.data["mask_filepath"] = self.data["mask_filepath"].str.replace(
            "jpg|tiff", "json", regex=True
        )

        # Get all optical images and inspection/camera metadata, if thermograms are in TIFF format
        if Path(data_dir, "opticals").is_dir():
            self.data["optical_filepath"] = self.data["image_filepath"].str.replace(
                "thermograms", "opticals"
            )
            self.data["optical_filepath"] = self.data["optical_filepath"].str.replace(
                "tiff", "jpg"
            )

        if Path(data_dir, "metadata.csv").is_file():
            self.data["metadata_filepath"] = ["metadata.csv"] * self.data.shape[0]

        self.transform = transform

    def __len__(self) -> int:
        """Method to know the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Method that iterate in the dataset.

        Args:
            index (int): Dataset instance index.

        Returns:
            tuple[np.ndarray, np.ndarray]: Number of images in the dataset.
        """
        # Get extra data required by TIFF images
        extra_files = {}
        if "optical_filepath" in self.data.columns:
            extra_files["optical_path"] = self.data.loc[index, "optical_filepath"]
        if "metadata_filepath" in self.data.columns:
            extra_files["metadata_path"] = self.data.loc[index, "metadata_filepath"]

        # Read thermogram
        thermogram = read_thermogram(
            self.data.loc[index, "image_filepath"], extra_files
        )
        image = thermogram.render(palette="grayscale", unit="celsius")

        # Read mask
        mask = read_mask(Path(self.data.loc[index, "mask_filepath"]))

        # Apply transformations on image and mask
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask
