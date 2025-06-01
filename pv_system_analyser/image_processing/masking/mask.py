""" "
File to define the thermograms masks.
"""

from __future__ import annotations

from collections.abc import Hashable
from functools import lru_cache, wraps
from typing import Callable

import numpy as np
import pandas as pd

from pv_system_analyser.constants import BOOL_ARRAY, INT_ARRAY

from .property_extractor import MaskPropertyExtractor


def check_data_is_2D_array(data: BOOL_ARRAY) -> None:
    """
    Checks that the input data is a 2D numpy array.

    Args:
        data (numpy.ndarray[bool]): The input data to check.

    Raises:
        ValueError: If the input data is not a 2D array.
    """
    if not data.ndim != 2:
        raise ValueError("Data is not a 2D array.")


def check_data_is_numpy_array(data: BOOL_ARRAY) -> None:
    """
    Checks that the input data is a numpy array.

    Args:
        data (numpy.ndarray[bool]): The input data to check.

    Raises:
        ValueError: If the input data is not a numpy array.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data is not a numpy array.")


def check_data_is_bool_type(data: BOOL_ARRAY) -> None:
    """
    Checks that the input data is a boolean numpy array.

    Args:
        data (numpy.ndarray[bool]): The input data to check.

    Raises:
        ValueError: If the input data is not a boolean numpy array.
    """
    if not data.dtype == np.bool_:
        raise ValueError(f"New data is not a bool type: {data.dtype}.")


def check_data_is_binary_mask(data: BOOL_ARRAY) -> None:
    """
    Checks that the input data is a binary mask (a 2D numpy array containing only 0 and 1 values).

    Args:
        data (numpy.ndarray[bool]): The input data to check.

    Raises:
        ValueError: If the input data is not a supported binary mask.
    """
    unique_values = set(np.unique(data))
    if len(unique_values) != 2 and not unique_values.issubset({0, 1}):
        raise ValueError("Data is not a supported binary mask.")


def check_data_is_compatible(
    old_shape: tuple[int, ...], new_shape: tuple[int, ...]
) -> None:
    """
    Checks that the new data shape is compatible with the original data shape.

    Args:
        old_shape (tuple[int, ...]): The original data shape.
        new_shape (tuple[int, ...]): The new data shape to check.

    Raises:
        ValueError: If the new data shape is different from the original data shape.
    """
    if new_shape != old_shape:
        msg = f"""
        New data shape {new_shape} is different from the original data shape {old_shape}.
        """
        raise ValueError(msg)


def validate_mask(fn: Callable[..., None]) -> Callable[..., None]:
    validate_functions = [
        check_data_is_numpy_array,
        check_data_is_2D_array,
        check_data_is_binary_mask,
        check_data_is_bool_type,
    ]

    @wraps(fn)
    def wrapper(self, data: BOOL_ARRAY) -> None:
        for function in validate_functions:
            function(data)
        check_data_is_compatible(self._data.shape, data.shape)
        fn(data)

    return wrapper


class Mask(Hashable):
    """
    Class that establish all mask features and methods.
    """

    def __init__(self, data: BOOL_ARRAY) -> None:
        self._data = data
        self.property_extractor = MaskPropertyExtractor()

    def __getitem__(self, id: int) -> BOOL_ARRAY:
        return self._data[id]

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mask):
            return NotImplemented
        condition = self._data[0].flatten() == other.data[0].flatten()
        return condition.all()

    def __hash__(self) -> int:
        flatten_data = tuple(self._data[0].flatten())
        return hash(flatten_data)

    def __repr__(self) -> str:
        return f"Mask(data={self._data.shape})"

    @property
    def data(self) -> BOOL_ARRAY:
        return self._data

    @data.setter
    @validate_mask
    def data(self, new_data: BOOL_ARRAY) -> None:
        self._data = new_data

    @property
    @lru_cache
    def label_mask(self) -> INT_ARRAY:
        """
        Label mask creation from data.

        Returns:
            numpy.ndarray[uint8]: Label mask array.
        """
        label_mask = np.zeros(self._data.shape[1:])
        for index, bool_mask in enumerate(self._data, start=1):
            label_mask[bool_mask] = index
        return label_mask.astype("uint8")

    @property
    @lru_cache
    def binary_mask(self) -> INT_ARRAY:
        """
        Binary mask creation from label mask.

        Returns:
            numpy.ndarray[uint8]: Binary mask array.
        """
        return np.where(self.label_mask != 0, 1, 0).astype("uint8")

    @property
    @lru_cache
    def label_props(self) -> pd.DataFrame:
        """
        Label mask properties calculation.

        Returns:
            pd.DataFrame: Calculated properties table.
        """
        return self.property_extractor.extract(self.label_mask)

    @property
    @lru_cache
    def binary_props(self) -> pd.DataFrame:
        """
        Binary mask properties calculation.

        Returns:
            pd.DataFrame: Calculated properties table.
        """
        return self.property_extractor.extract(self.binary_mask)
