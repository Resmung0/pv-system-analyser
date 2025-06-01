from functools import wraps
from typing import TypedDict

import numpy as np
from skimage import draw, util

from pv_system_analyser.constants import INT_ARRAY, STRING_ARRAY
from pv_system_analyser.parser.interface import AnnotationParser


class Coordinate(TypedDict):
    x: float
    y: float


class Instance(TypedDict):
    corners: list[Coordinate]
    center: Coordinate
    defected_module: bool


class PSTData(TypedDict):
    instances: list[Instance]


def validate_shape(func):
    @wraps(func)
    def wrapper(self, data, shape=(512, 640)):
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError("Shape must be a tuple of length 2")

        if not all(isinstance(dim, int) and dim > 0 for dim in shape):
            raise ValueError("Shape dimensions must be positive integers")

        return func(self, data, shape)

    return wrapper


class PSTAnnotationParser(AnnotationParser):
    """
    Class that parse data extracted from Photovoltaic System Thermography dataset.
    """

    @validate_shape
    def __init__(
        self,
        data: PSTData,
        shape: tuple[int, int] = (512, 640),
    ) -> None:
        self.__shape, self.__data = shape, data["instances"]

    @property
    def masks(self) -> INT_ARRAY:
        def parse_mask(corners: list[Coordinate]) -> INT_ARRAY:
            polygon = np.array([[coords["y"], coords["x"]] for coords in corners])
            polygon = draw.polygon2mask(self.__shape, polygon)
            return util.img_as_ubyte(polygon)

        parsed_mask = [parse_mask(data["corners"]) for data in self.__data]
        return np.array(parsed_mask, dtype="int")

    @property
    def centers(self) -> INT_ARRAY:
        return np.array([instance["center"] for instance in self.__data])

    @property
    def classes(self) -> STRING_ARRAY:
        def parse_class(is_defected: bool) -> str:
            return "defected" if is_defected else "non-defected"

        modules_classes = [parse_class(data["defected_module"]) for data in self.__data]
        return np.array(modules_classes, dtype=np.str_)
