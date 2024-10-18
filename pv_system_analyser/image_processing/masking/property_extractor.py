import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.measure import regionprops, regionprops_table


class MaskPropertyExtractor:
    
    def extract(self, mask: npt.NDArray[np.uint8]) -> pd.DataFrame:
        """Extract mask properties."""
        properties = self._calculate_standard_props(mask)
        properties['scale'] = properties.area.map(self._calculate_mask_scale)
        properties = self.__apply_mask_corners_calculation(mask, properties)
        return properties

    @staticmethod
    def _calculate_mask_scale(
        area: float, min_limit: int = 1024, max_limit: int = 9216
    ) -> str:
        """Calculate mask scale based on it's area. This is based on
        COCO dataset mask scale definition.

        Args:
            area (float): Mask area.
            min_limit (int, optional): Minimal area value. Defaults to 1024.
            max_limit (int, optional): Maximal area value. Defaults to 9216.

        Returns:
            str: Mask type.
        """

        mask_type = "medium"
        if area < min_limit:
            mask_type = "small"
        elif area > max_limit:
            mask_type = "large"
        return mask_type
    
    @staticmethod
    def _calculate_mask_corners(
        binary_mask: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Calculate the mask extreme points.

        Args:
            binary_mask (np.ndarray): Mask to calculate extreme points.

        Returns:
            np.ndarray: Mask extreme points.
        """

        # Find mask coordinates
        cnt = regionprops(binary_mask)[0].coords[:, ::-1]

        # Find extreme points from diagonal left
        sum_coords = cnt.sum(axis=1)
        top_left = cnt[np.argmin(sum_coords)]
        bottom_right = cnt[np.argmax(sum_coords)]

        # Find extreme points from diagonal right
        diff_coords = np.squeeze(np.diff(cnt, axis=1))
        top_right = cnt[np.argmin(diff_coords)]
        bottom_left = cnt[np.argmax(diff_coords)]

        # encapsulate corner points
        return np.array([top_left, top_right, bottom_right, bottom_left])

    @staticmethod
    def _calculate_standard_props(mask: npt.NDArray[np.uint8]) -> pd.DataFrame:
        properties = pd.DataFrame(
            regionprops_table(mask, properties=["area", "centroid"])
        )
        properties.columns = ["area", "centroid-height", "centroid-width"]
        return properties
    
    def __apply_mask_corners_calculation(
        self, mask: npt.NDArray[np.uint8], properties: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply mask corners calculation."""
        corners = []
        for label in np.unique(mask)[1:]:
            binary_mask = np.where(mask == label, 1, 0)
            mask_corners = self._calculate_mask_corners(binary_mask)
            corners.append(mask_corners.reshape(-1))
        
        corners = pd.DataFrame(corners)
        corners.columns = [
            "top-left-height",
            "top-left-width",
            "top-right-height",
            "top-right-width",
            "bottom-right-height",
            "bottom-right-width",
            "bottom-left-height",
            "bottom-left-width",
        ]
        return pd.concat([properties, corners], axis=1)
        
        