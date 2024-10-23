from __future__ import annotations
from abc import ABC, abstractmethod

import numpy.typing as npt
import numpy as np

class Thermogram(ABC):
    """
    Base class that implements a thermogram image. It follows the 
    flyr.FlyrThermogram class implementation.
    """

    @property
    @abstractmethod
    def identifier(self) -> str:
        """
        Thermogram file name.

        Returns:
            str: Thermogram file name.
        """
        return NotImplementedError()

    @property
    @abstractmethod
    def kelvin(self) -> npt.NDArray[np.float16]:
        """
        Thermogram's temperature in Kelvin (K).

        Returns:
            ndarray: Thermogram temperature in Kelvin.
        """
        return NotImplementedError()

    @property
    @abstractmethod
    def celsius(self) -> npt.NDArray[np.float16]:
        """
        Thermogram's temperature in celsius (°C).

        Returns:
            ndarray: Thermogram temperature in Celsius.
        """
        return NotImplementedError()

    @property
    @abstractmethod
    def fahrenheit(self) -> npt.NDArray[np.float16]:
        """
        Thermogram's temperature in Fahrenheit (°F).

        Returns:
            ndarray: Thermogram temperature in Fahrenheit.
        """
        return NotImplementedError()

    @property
    @abstractmethod
    def optical(self) -> npt.NDArray[np.uint8]:
        """
        The thermogram's embedded photo.

        Returns:
            ndarray: Thermogram embedded photo.
        """
        return NotImplementedError()

    @property
    @abstractmethod
    def metadata(self) -> dict[str, str | int]:
        """
        Metadata related to thermogram temperature data.

        Returns:
            dict[str, str | int]: Metadata that build the thermogram.
        """
        return NotImplementedError()

    @abstractmethod
    def render(
        self,
        min_v: float | None = None,
        max_v: float | None = None,
        palette: str = "grayscale",
        unit: str = "kelvin",
    ) -> npt.NDArray[np.uint8]:
        """
        Renders the thermogram, transforming it into a 8-bit image with the given settings.

        Args:
            min_v (float): Minimal value to consider the thermogram's temperature range. Defaults to None.
            max_v (float): Maximal value to consider the thermogram's temperature range. Defaults to None.
            palette (str): Palette to render the thermogram. Default to "grayscale".
            unit (str): Unit to compute the thermal image. Default to "kelvin".

        Returns:
            ndarray: Thermogram rendered to a 8 bit image.
        """
        return NotImplementedError()

    @abstractmethod
    def adjust_metadata(self) -> Thermogram:
        """
        Adjust the metadata that build the thermogram.
        
        Returns:
            TIFFThermogram: Thermogram.
        """
        return NotImplementedError()