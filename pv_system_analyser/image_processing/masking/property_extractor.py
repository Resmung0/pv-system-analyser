import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from skimage.measure import regionprops, regionprops_table

from pv_system_analyser.constants import INT_ARRAY


class StandardPropsSchema(pa.DataFrameModel):
    area: Series[int] = pa.Field()
    centroid_height: Series[int] = pa.Field()
    centroid_width: Series[int] = pa.Field()


class MaskPropertyOutputSchema(pa.DataFrameModel):
    area: Series[int] = pa.Field()
    centroid_height: Series[int] = pa.Field()
    centroid_width: Series[int] = pa.Field()
    scale: Series[str] = pa.Field()
    top_left_height: Series[int] = pa.Field()
    top_left_width: Series[int] = pa.Field()
    top_right_height: Series[int] = pa.Field()
    top_right_width: Series[int] = pa.Field()
    bottom_right_height: Series[int] = pa.Field()
    bottom_right_width: Series[int] = pa.Field()
    bottom_left_height: Series[int] = pa.Field()
    bottom_left_width: Series[int] = pa.Field()


class MaskPropertyExtractor:
    """
    A class for extracting properties from binary masks.

    This class provides methods to calculate various properties of binary masks,
    including area, centroid, scale, and corners. It is designed to work with
    numpy arrays representing binary masks and returns the extracted properties
    as a pandas DataFrame.
    """

    @pa.check_types
    def extract(self, mask: INT_ARRAY) -> DataFrame[MaskPropertyOutputSchema]:
        """Extract mask properties."""
        properties = self._calculate_standard_props(mask)
        properties = self._calculate_mask_scale(properties)
        properties = self._calculate_mask_corners(mask, properties)
        return properties

    @staticmethod
    def _calculate_standard_props(mask: INT_ARRAY) -> DataFrame[StandardPropsSchema]:
        """
        Calculate standard mask properties such as area and centroid.

        Args:
            mask (numpy.ndarray[uint8]): The binary mask to calculate properties.

        Returns:
            DataFrame[StandardPropsSchema]: A DataFrame containing the calculated mask properties, including the area, centroid height, and centroid width.
        """
        properties = pd.DataFrame(
            regionprops_table(mask, properties=["area", "centroid"])
        )
        properties.columns = ["area", "centroid-height", "centroid-width"]
        return properties

    @staticmethod
    def _calculate_mask_scale(properties: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mask scale based on it's area. This is based on COCO dataset mask scale definition.

        Args:
            area (float): Mask area.
            min_limit (int): Minimal area value. Defaults to 1024.
            max_limit (int): Maximal area value. Defaults to 9216.

        Returns:
            str: Mask type.
        """

        def calculate_scale(
            area: float, min_limit: int = 1024, max_limit: int = 9216
        ) -> str:
            """
            Calculate mask scale based on it's area. This is based on COCO dataset mask scale definition.

            Args:
                area (float): Mask area.
                min_limit (int): Minimal area value. Defaults to 1024.
                max_limit (int): Maximal area value. Defaults to 9216.

            Returns:
                str: Mask type.
            """
            mask_type = "medium"
            if area < min_limit:
                mask_type = "small"
            elif area > max_limit:
                mask_type = "large"
            return mask_type

        properties["scale"] = properties.area.map(calculate_scale)
        return properties

    def _calculate_mask_corners(
        self, mask: INT_ARRAY, properties: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply mask corners calculation.

        This method calculates the extreme points (corners) of the mask regions in the
        provided binary mask. It iterates through each unique label in the mask,
        extracts the binary mask for that label, and then calculates the top-left,
        top-right, bottom-right, and bottom-left coordinates of the mask region.

        The calculated corner coordinates are then added as new columns to the provided
        properties DataFrame and returned.

        Args:
            mask (numpy.ndarray[uint8]]): The binary mask to calculate the corners for.
            properties (pd.DataFrame): The DataFrame containing the mask properties.

        Returns:
            pd.DataFrame: The updated properties DataFrame with the added corner
                coordinates.
        """

        def mask_corners_calculation(binary_mask: INT_ARRAY) -> INT_ARRAY:
            """
            Calculate the mask extreme points.

            Args:
                binary_mask (numpy.ndarray[uint8]]): Mask to calculate extreme points.

            Returns:
                numpy.ndarray[uint8]: Mask extreme points.
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

        corners = []
        for label in np.unique(mask)[1:]:
            binary_mask = np.where(mask == label, 1, 0)
            mask_corners = mask_corners_calculation(binary_mask)
            corners.append(mask_corners.reshape(-1))

        corners_df = pd.DataFrame(corners)
        corners_df.columns = [
            "top-left-height",
            "top-left-width",
            "top-right-height",
            "top-right-width",
            "bottom-right-height",
            "bottom-right-width",
            "bottom-left-height",
            "bottom-left-width",
        ]
        return pd.concat([properties, corners_df], axis=1)
