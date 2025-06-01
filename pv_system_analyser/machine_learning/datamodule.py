from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Sequential, ToTensor

from pv_system_analyser.factory import DataModuleFactory
from pv_system_analyser.io import _read_json_annotation

from .dataset import ClassificationDataset
from .preprocessing import ToOneHot


@DataModuleFactory.register("classification")
class ClassificationDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning data module for loading and preprocessing classification datasets.

    This data module provides functionality for loading and preprocessing classification datasets
    for use in PyTorch Lightning models. It supports loading train, validation, and test datasets
    from JSON annotation files, applying data augmentations, and converting targets to one-hot
    encoding.

    The data module can be configured with the following parameters:
    - `augmentations`: A Compose of data augmentation transforms to apply to the training dataset.
    - `batch_size`: The batch size for the data loaders.
    - `num_workers`: The number of worker processes to use for data loading.
    """

    def __init__(
        self,
        augmentations: Compose,
        batch_size: int = 32,
        num_workers: int = 1,
    ) -> None:
        # Allows to access init params with 'self.hparams' attribute
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Training hyperparams
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Transformations
        self.target_transform = ToOneHot(10)
        self.transforms = augmentations
        self._base_transforms = Sequential(ToTensor(), Normalize((0), (1)))

    def train_dataloader(self) -> DataLoader:
        """
        Returns a PyTorch DataLoader for the training dataset.

        This method reads the training metadata from a JSON file, creates a ClassificationDataset
        instance with the training metadata, and applies the specified data augmentation and
        preprocessing transformations. The resulting training dataset is then wrapped in a
        PyTorch DataLoader with the configured batch size, number of worker processes, and
        shuffle setting.
        """
        train_metadata = _read_json_annotation(
            path=Path("/assets/data/processed/classification/train_metadata.json"),
            dataset_type="classification",
        )
        transforms = (Compose(self.transforms, self._base_transforms),)
        return self._common_step(train_metadata, transforms, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """
        Returns a PyTorch DataLoader for the validation dataset.

        This method reads the validation metadata from a JSON file, creates a ClassificationDataset
        instance with the validation metadata, and applies the specified base transformations. The
        resulting validation dataset is then wrapped in a PyTorch DataLoader with the configured
        batch size, number of worker processes, and shuffle setting.
        """
        valid_metadata = _read_json_annotation(
            path=Path("/assets/data/processed/classification/valid_metadata.json"),
            dataset_type="classification",
        )
        return self._common_step(valid_metadata, self._base_transforms, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """
        Returns a PyTorch DataLoader for the testing dataset.

        This method reads the testing metadata from a JSON file, creates a ClassificationDataset
        instance with the testing metadata, and applies the specified base transformations. The
        resulting testing dataset is then wrapped in a PyTorch DataLoader with the configured
        batch size, number of worker processes, and shuffle setting.
        """
        test_metadata = _read_json_annotation(
            path=Path("/assets/data/processed/classification/test_metadata.json"),
            dataset_type="classification",
        )
        return self._common_step(test_metadata, self._base_transforms, shuffle=False)

    def _common_step(self, metadata, transform, shuffle: bool) -> DataLoader:
        dataset = ClassificationDataset(
            metadata=metadata,
            transform=transform,
            target_transform=self.target_transform,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )
