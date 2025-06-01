from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pv_system_analyser.constants import FLOAT_ARRAY, INT_ARRAY


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
        raise NotImplementedError()

    @property
    @abstractmethod
    def kelvin(self) -> FLOAT_ARRAY:
        """
        Thermogram's temperature in Kelvin (K).

        Returns:
            ndarray: Thermogram temperature in Kelvin.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def celsius(self) -> FLOAT_ARRAY:
        """
        Thermogram's temperature in celsius (°C).

        Returns:
            ndarray: Thermogram temperature in Celsius.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def fahrenheit(self) -> FLOAT_ARRAY:
        """
        Thermogram's temperature in Fahrenheit (°F).

        Returns:
            ndarray: Thermogram temperature in Fahrenheit.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def optical(self) -> INT_ARRAY | None:
        """
        The thermogram's embedded photo.

        Returns:
            ndarray: Thermogram embedded photo.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def metadata(self) -> dict[str, str | int] | None:
        """
        Metadata related to thermogram temperature data.

        Returns:
            dict[str, str | int]: Metadata that build the thermogram.
        """
        raise NotImplementedError()

    @abstractmethod
    def render(
        self,
        min_v: float | None = None,
        max_v: float | None = None,
        palette: str = "grayscale",
        unit: Literal["kelvin", "celsius", "fahrenheit"] = "kelvin",
    ) -> INT_ARRAY:
        """
        Renders the thermogram, transforming it into a 8-bit image with the given settings.

        Args:
            min_v (float): Minimal value to consider the thermogram's temperature range. Defaults to None.
            max_v (float): Maximal value to consider the thermogram's temperature range. Defaults to None.
            palette (str): Palette to render the thermogram. Default to "grayscale".
            unit (Literal["kelvin", "celsius", "fahrenheit"]): Unit to compute the thermal image. Default to "kelvin".

        Returns:
            numpy.ndarray[uint8]: Thermogram rendered to a 8 bit image.
        """
        raise NotImplementedError()

    @abstractmethod
    def adjust_metadata(self) -> Thermogram | None:
        """
        Adjust the metadata that build the thermogram.

        Returns:
            TIFFThermogram: Thermogram.
        """
        raise NotImplementedError()
