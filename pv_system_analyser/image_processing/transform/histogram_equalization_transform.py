from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from skimage import exposure
from skimage.util import img_as_ubyte

from pv_system_analyser.constants import INT_ARRAY
from pv_system_analyser.image_processing.masking import Mask
from pv_system_analyser.image_processing.thermogram.interface import Thermogram


@dataclass(slots=True)
class HistogramEqualizationTransform:
    """
    Transformation responsible for apply histogram equalization on a thermogram image, enhancing it's contrast.

    Args:
        clip_limit (float, optional): Maximum value to clip the histogram equalization. Defaults to 0.04.
        kernel_size (tuple[int, int], optional): Size of the kernel used. Defaults to (3, 3).
        method (Literal["local", "global"]): Histogram equalization methodology to apply ("global" or "local"). Defaults to "local".
    """

    clip_limit: float = field(default_factory=lambda: 0.04)
    kernel_size: tuple[int, int] = field(default_factory=lambda: (3, 3))
    method: Literal["local", "global"] = field(default_factory=lambda: "local")

    def __call__(self, thermogram: Thermogram, mask: Mask, **kwargs) -> INT_ARRAY:
        """
        Apply the respective histogram equalization methodology to thermogram region highlighted by mask.

        Args:
            thermogram (Thermogram): Thermogram image.
            mask (Mask): Thermogram mask that highlights each image region
            where to apply the transformation.

        Returns:
            numpy.ndarray[uint8]: Enhanced thermogram.
        """
        image = thermogram.render(**kwargs)
        binary_mask = mask.binary_mask.astype("bool")
        foreground = self._aply_mask(image, binary_mask, True)
        background = self._aply_mask(image, binary_mask, False)
        foreground = self._equalize_foreground(foreground, mask)
        return foreground + background

    def _equalize_foreground(self, image: INT_ARRAY, mask: Mask) -> INT_ARRAY:
        """
        Apply the histogram equalization to the foreground region of the thermogram.

        Args:
            image (numpy.ndarray[uint8]): Rendered thermogram image (8-bit).
            mask (Mask): Thermogram mask.

        Returns:
            numpy.ndarray[uint8]: Masked equalized image.
        """
        match self.method:
            case "local":
                image = exposure.equalize_adapthist(
                    image, self.kernel_size, self.clip_limit
                )
            case "global":
                image = exposure.equalize_hist(image, mask=mask)
        image = img_as_ubyte(image)
        image[image == image.min()] = 0
        return image

    @staticmethod
    def _aply_mask(image: INT_ARRAY, mask: INT_ARRAY, invert_mask: bool) -> INT_ARRAY:
        """Apply the mask in the thermogram, highlighting only the region of interest.

        Args:
            image (numpy.ndarray[uint8]): Rendered thermogram image (8-bit).
            mask (numpy.ndarray[uint8]): Thermogram mask.
            invert_mask (bool): Trigger to invert the mask intensities.

        Returns:
            numpy.ndarray[uint8]: Masked image.
        """
        mask = np.invert(mask) if invert_mask else mask
        image[mask] = 0
        return image
