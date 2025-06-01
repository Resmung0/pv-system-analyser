from dataclasses import dataclass, field

import numpy as np
from skimage.morphology import binary_closing
from skimage.segmentation import expand_labels

from pv_system_analyser.constants import BOOL_ARRAY, INT_ARRAY
from pv_system_analyser.image_processing.masking.mask import Mask


@dataclass(slots=True)
class ExpantionMaskAdjuster:  # type: ignore
    """Adjust the mask by expanding each instance domain."""

    distance: int = field(default_factory=lambda: 7)
    kernel_size: tuple[int, int] = field(default_factory=lambda: (9, 11))

    def __call__(self, mask: Mask) -> BOOL_ARRAY:
        """Method that expand each instance domain in the mask.

        Args:
            mask (Mask): Thermogram mask that needs to be adjusted.

        Returns:
            np.ndarray: Adjusted mask array.
        """
        # Handle label mask
        expanded_label_mask = self._expand_label_mask(mask.label_mask)
        expanded_binary_mask = self._binarize_expanded_label_mask(expanded_label_mask)

        # Handle binary mask
        original_binary_mask = self._close_binary_mask(mask.binary_mask)

        # Process them
        lines = self._subtract_binary_masks(expanded_binary_mask, original_binary_mask)
        expanded_label_mask = self._erase_overlapping_lines(expanded_label_mask, lines)
        return self.__create_layers(expanded_label_mask)

    def _close_binary_mask(self, binary_mask: INT_ARRAY) -> INT_ARRAY:
        kernel = np.ones(self.kernel_size, np.uint8)
        binary_mask = binary_closing(binary_mask, kernel)
        return binary_mask.astype("uint8")

    def _expand_label_mask(self, label_mask: INT_ARRAY) -> INT_ARRAY:
        return expand_labels(label_mask, distance=self.distance)

    @staticmethod
    def _binarize_expanded_label_mask(label_mask: INT_ARRAY) -> INT_ARRAY:
        return label_mask.astype("bool").astype("uint8")

    @staticmethod
    def _subtract_binary_masks(
        first_mask: INT_ARRAY, second_mask: INT_ARRAY
    ) -> INT_ARRAY:
        return first_mask ^ second_mask

    @staticmethod
    def _erase_overlapping_lines(
        label_mask: INT_ARRAY, binary_mask: INT_ARRAY
    ) -> INT_ARRAY:
        new_label_mask = label_mask.copy()
        new_label_mask[binary_mask == 1] = 0
        return new_label_mask

    @staticmethod
    def __create_layers(label_mask: INT_ARRAY) -> BOOL_ARRAY:
        labels = np.unique(label_mask)[1:]
        data = [np.where(label_mask == label, True, False) for label in labels]
        return np.array(data, dtype="bool")
