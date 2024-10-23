from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from pv_system_analyser.masking.mask import Mask

@dataclass(slots=True)
class OverlapMaskAdjuster(Callable):
    """Adjust the mask by eliminating overlap between instances.
    Based on https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/279995.
    """

    is_overlapped: bool = field(default_factory=lambda: False)
    
    def __call__(self, mask: Mask) -> npt.NDArray[np.bool]:
        """Method that remove instances overlaping in the mask.

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
    def _check_overlap(data: npt.NDArray[np.bool]) -> bool:
        """Check if overlap exist in the mask.

        Args:
            data (npt.NDArray[np.bool]): Array mask.

        Returns:
            bool: Confirmation if overlap exists.
        """
        mask = np.stack(data, axis=-1).astype("bool").astype("uint8")
        return np.any(mask.sum(axis=-1) > 1)

    def _fix_overlap(self, data: npt.NDArray[np.bool]) -> npt.NDArray[np.bool]:
        """Correct the overlapped mask.

        Args:
            data (np.ndarray): Array mask.

        Returns:
            np.ndarray: Corrected array mask.
        """
        data = self._change_instance_index(data)
        data = self._correct_overlap(data)
        data = self._bring_back_instance_index(data)
        return data
    
    @staticmethod
    def __change_instance_index(data: npt.NDArray[np.bool]) -> npt.NDArray[np.bool]:
        return np.stack(data, axis=1)
        
    @staticmethod
    def __correct_overlap(data: npt.NDArray[np.bool]) -> npt.NDArray[np.bool]:
        data = np.pad(data, [[0, 0], [0, 0], [1, 0]])
        data = np.argmax(data, axis=-1)
        data = np.eye(data.shape[-1], dtype="uint8")[data]
        data = data[..., 1:]
        return data[..., np.any(data, axis=(0, 1))]

    @staticmethod
    def __bring_back_instance_index(data: npt.NDArray[np.bool]) -> npt.NDArray[np.bool]:
        data = [
            data[:, :, i] for i in range(data.shape[-1])
        ]
        return np.array(data, dtype="bool")
    
