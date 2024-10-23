from skimage import draw, util
import numpy as np
import numpy.typing as npt

from pv_system_analyser.parser.interface import AnnotationParser


class PSTAnnotationParser(AnnotationParser):
    """
    Class that parse data extracted from Photovoltaic System Thermography dataset.
    """
    def __init__(
        self,
        data: dict[str, str | int],
        shape: tuple[int, int] = (512, 640)
    ) -> None:
        """        
        Args:
            shape (tuple[int, int], optional): Dimensions of corresponding image mask. 
            Defaults to (512, 640).
        """
        self.shape, self.__data = shape, data['instances']

    @property
    def masks(self) -> npt.NDArray[np.uint8]:
        return np.array([self._parse_mask(data) for data in self.__data])
    
    def _parse_mask(self, instance: dict[str, str | int]) -> npt.NDArray[np.uint8]:
        polygon = np.array(
            [[coords["y"], coords["x"]] for coords in instance["corners"]]
        )
        polygon = draw.polygon2mask(self.shape, polygon)
        return util.img_as_ubyte(polygon)
    
    @property
    def centers(self) -> npt.NDArray[np.uint8]:
        return np.array([instance['center'] for instance in self.__data])
    
    @property
    def classes(self) -> npt.NDArray[np.dtype.str]:
        return np.array([self._parse_class(data) for data in self.__data])

    @staticmethod
    def _parse_class(instance) -> str:
        return "defected" if instance["defected_module"] else "non-defected"
