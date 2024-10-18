from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.transform import ProjectiveTransform, warp

from pv_system_analyser.image_processing.masking import Mask
from pv_system_analyser.image_processing.thermogram.interface import Thermogram


class PerspectiveTransform(Callable):
    """Transformation responsible for image perspective correction by applying
    homography to masked images.
    """

    def __init__(self) -> None:
        self.homography = ProjectiveTransform()
        self.source_points, self.destination_points = None, None


    def __call__(
        self,
        thermogram: Thermogram,
        mask: Mask,
        area_threshold: float = 1000,
        **kwargs
    ) -> npt.NDArray[np.uint8]:
        """Method to correct the perspective of each region of interest (ROI) in the image
        based on the label mask given. First, gets extreme points of each instance of the
        mask. Then, for each of them estimate the Homography matrix of the ROI and warp it.

        Args:
            thermogram (Thermogram): Thermogram image.
            mask (Mask): Thermogram mask that highlights each image region where to apply the transformation.
            area_threshold (float): Threshold area to filter instances. Default to 1000.

        Returns:
             npt.NDArray[np.uint8]: Transformed image.
        """

        self.source_points = self._get_mask_extreme_points(mask.properties)

        warped_images, destination_points = [], []
        for index, src_pts in enumerate(self.source_points):
            # Filter only the instances that it's area larger then threshold
            if mask.properties.loc[index, "area"] > area_threshold:
                # Get the projected values (destination points and projected shape of the warped image)
                dst_pts, warped_projected_shape = self._calculate_destination_values(src_pts)
                destination_points.append(dst_pts)

                # Estimate the homography matrix and warp the related region of the image
                self.homography.estimate(src_pts, dst_pts)
                
                # Filter only the instances that the homography matrix is inversible 
                # (determinant different from 0)
                if np.linalg.det(self.homography.params) != 0:
                    image = thermogram.render(**kwargs)
                    warped_image = self._warp(image, warped_projected_shape)
                    warped_images.append(warped_image)
        
        self.destination_points = np.array(destination_points)
        return warped_images
    
    
    def _calculate_destination_values(
        self, source_points: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.uint8], tuple[int, int]]:
        """Calculation of values related to the destination image.

        Args:
            source_points (npt.NDArray[np.uint8]): Points to start the transformation.

        Returns:
            tuple[npt.NDArray[np.uint8], tuple[int, int]]: Calculated values by the transformation.
            First is the destination points and then the projected warped image shape.
        """
        top_left, top_right, bottom_right, bottom_left = source_points
        
        w_1 = self.__calculate_width(bottom_right, bottom_left)
        w_2 = self.__calculate_width(top_right, top_left)
        approx_w = max(int(w_1), int(w_2))

        h_1 = self.__calculate_height(top_right, bottom_right)
        h_2 = self.__calculate_height(top_left, bottom_left)
        approx_h = max(int(h_1), int(h_2))
        
        destination_points = np.float32(
            [(0, 0), (approx_w - 1, 0), (approx_w - 1, approx_h - 1), (0, approx_h - 1)]
        )
        return destination_points, (approx_h, approx_w)
    
    @staticmethod
    def __calculate_width(
        right: tuple[int, int], left: tuple[int, int]
    ) -> float:
        x_coord = (right[0] - left[0]) ** 2
        y_coord = (right[1] - left[1]) ** 2
        return np.sqrt(x_coord + y_coord)
   
    @staticmethod
    def __calculate_height(
        top: tuple[int, int], bottom: tuple[int, int]
    ) ->  float:
        x_coord = (top[0] - bottom[0]) ** 2
        y_coord = (top[1] - bottom[1]) ** 2
        return np.sqrt(x_coord + y_coord)
    
    def _warp(
        self, image: npt.NDArray[np.uint8], projected_shape: tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        warped_image = warp(
            image, 
            self.homography.inverse,
            output_shape=projected_shape, 
            preserve_range=True
        )
        return warped_image.astype("uint8")
    
    @staticmethod
    def _get_mask_extreme_points(mask_properties: pd.DataFrame) -> npt.NDArray[np.uint8]:
    	return mask_properties.iloc[:, 4:].values.reshape(-1, 4, 2)

    	
