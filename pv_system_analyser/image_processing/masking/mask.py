""""
File to define the thermograms masks.
"""
from __future__ import annotations
from collections.abc import Sequence, Hashable
from functools import lru_cache
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from .validation import (
    check_data_is_2D_array,
    check_data_is_binary_mask,
    check_data_is_bool_type,
    check_data_is_compatible,
    check_data_is_numpy_array
)
from .property_extractor import MaskPropertyExtractor



@dataclass(slots=True)
class Mask(Sequence, Hashable):
    """
    Class that establish all mask features and methods.
    """
    data: npt.NDArray[np.bool]
    _data: npt.NDArray[np.bool] = field(init=False)
    property_extractor: MaskPropertyExtractor = field(init=False)

    def __post_init__(self) -> None:
        self.property_extractor = MaskPropertyExtractor()
    
    def __getitem__(self, id: int) -> npt.NDArray[np.bool]:
        return self.data[id]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __eq__(self, other: Mask) -> bool:
        condition = self.data[0].flatten() == other.data[0].flatten()
        return condition.all()
    
    def __hash__(self) -> int:
        flatten_data = tuple(self.data[0].flatten())
        return hash(flatten_data)
    
    def __repr__(self) -> str:
        return f"Mask(data={self.data.shape})"
    
    @property
    def data(self) -> npt.NDArray[np.bool]:
        return self._data
    
    @data.setter
    def data(self, new_data: npt.NDArray[np.bool]) -> None:
        check_data_is_2D_array(new_data)
        check_data_is_numpy_array(new_data)
        check_data_is_binary_mask(new_data)
        check_data_is_bool_type(new_data)
        check_data_is_compatible(self._data.shape, new_data.shape)
        self._data = new_data
    
    @property
    @lru_cache
    def label_mask(self) -> npt.NDArray[np.uint8]:
        """
        Label mask creation from data.

        Returns:
            npt.NDArray[np.uint8]: Label mask array.
        """
        label_mask = np.zeros(self.data.shape[1:])
        for index, bool_mask in enumerate(self.data, start=1):
            label_mask[bool_mask] = index
        return label_mask.astype("uint8")
    
    @property
    @lru_cache
    def binary_mask(self) -> npt.NDArray[np.uint8]:
        """
        Binary mask creation from label mask.

        Returns:
            npt.NDArray[np.uint8]: Binary mask array.
        """
        return np.where(self.label_mask != 0, 1, 0).astype('uint8')
    
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
