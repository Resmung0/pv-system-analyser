from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from skimage.morphology import binary_closing
from skimage.segmentation import expand_labels
from skimage.util import img_as_ubyte

from pv_system_analyser.masking.mask import Mask

@dataclass(slots=True)
class ExpantionMaskAdjuster(Callable):
    """Adjust the mask by expanding each instance domain."""

    distance: int = field(default_factory=lambda: 7)
    
    def __call__(self, mask: Mask) -> npt.NDArray[np.bool]:
        """Method that expand each instance domain in the mask.

        Args:
            mask (Mask): Thermogram mask that needs to be adjusted.

        Returns:
            np.ndarray: Adjusted mask array.
        """
        original_binary_mask = self._close_binary_mask(mask.binary_mask)
        expanded_label_mask = self._expand_label_mask(mask.label_mask)
        expanded_binary_mask = self._binarize_expanded_label_mask(expanded_label_mask)
        expanded_label_mask = self._erase_overlapping_lines(
            expanded_label_mask,
            [expanded_binary_mask, original_binary_mask]
        )
        return expanded_label_mask

    @staticmethod
    def _close_binary_mask(
        label_mask: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = binary_closing(label_mask, kernel)
        return img_as_ubyte(binary_mask)

    def _expand_label_mask(
        self, label_mask: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        return expand_labels(label_mask, distance=self.distance)
	
    @staticmethod
    def _binarize_expanded_label_mask(
        label_mask: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.bool]:
        return label_mask.astype("bool").copy()
	
    @staticmethod
    def _erase_overlapping_lines(
        label_mask: npt.NDArray[np.uint8], binary_masks: list[npt.NDArray[np.uint8]]
    ) -> npt.NDArray[np.uint8]:
        lines = np.stack(binary_masks).diff()
        label_mask[lines == 1] = 0
        return label_mask

