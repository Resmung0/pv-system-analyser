from dataclasses import dataclass, field
from itertools import permutations, starmap

import numpy as np
from skimage.measure import intersection_coeff

from pv_system_analyser.constants import BOOL_ARRAY
from pv_system_analyser.image_processing.masking.mask import Mask


@dataclass(slots=True)
class OverlapMaskAdjuster:
    """
    Adjust the mask by eliminating overlap between instances.
    Based on https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/279995.
    """

    is_overlapped: bool = field(default_factory=lambda: False, init=False)

    def __call__(self, mask: Mask) -> BOOL_ARRAY:
        """
        Method that remove instances overlaping in the mask.

        Args:
            mask (Mask): Thermogram mask that needs to be adjusted.

        Returns:
            Mask: Adjusted Mask.
        """
        data = mask.data
        self.is_overlapped = self._check_overlap(data)
        if self.is_overlapped:
            data = self._fix_overlap(data)
        return data

    @staticmethod
    def estimate_overlap(mask: Mask) -> float:
        def calculate_overlap(index_1: int, index_2: int) -> float:
            return intersection_coeff(mask[index_1], mask[index_2])

        n_masks = len(mask)
        generator = starmap(calculate_overlap, permutations(range(n_masks), 2))
        overlaps = np.fromiter(generator, dtype=float).reshape(n_masks, -1)
        return overlaps.sum(axis=1).mean()

    def _check_overlap(self, data: BOOL_ARRAY) -> bool:
        """
        Check if overlap exist in the mask.

        Args:
            data (numpy.ndarray[bool]): Array mask.

        Returns:
            bool: Confirmation if overlap exists.
        """
        mask = self.__change_instance_index(data).astype("uint8")
        return bool(np.any(mask.sum(axis=-1) > 1))

    def _fix_overlap(self, data: BOOL_ARRAY) -> BOOL_ARRAY:
        """
        Correct the overlapped mask.

        Args:
            data (numpy.ndarray[bool]): Array mask.

        Returns:
            numpy.ndarray[bool]: Corrected array mask.
        """
        data = self.__change_instance_index(data)
        data = self.__correct_overlap(data)
        return self.__bring_back_instance_index(data)

    @staticmethod
    def __change_instance_index(data: BOOL_ARRAY) -> BOOL_ARRAY:
        return data.transpose(1, 2, 0)

    @staticmethod
    def __correct_overlap(data: BOOL_ARRAY) -> BOOL_ARRAY:
        data = np.pad(data, [[0, 0], [0, 0], [1, 0]])
        data = np.argmax(data, axis=-1)
        data = np.eye(data.shape[-1], dtype="uint8")[data]
        data = data[..., 1:]
        data = data[..., np.any(data, axis=(0, 1))]
        return data.astype("bool")

    @staticmethod
    def __bring_back_instance_index(data: BOOL_ARRAY) -> BOOL_ARRAY:
        return data.transpose(2, 0, 1)
