"""File that define the thermograms."""

from __future__ import annotations
from pathlib import Path

import numpy.typing as npt
import numpy as np
import pandas as pd
from flyr import palettes
from skimage import io

from .interface import Thermogram


class TIFFThermogram(Thermogram):
    """
       Class to read and process tiff thermograms.
    """

    def __init__(
        self,
        image_path: str,
        metadata_path: str | None = None,
        optical_path: str | None = None,
        method: str = "thermomap",
    ) -> None:
        """
        Args:
            image_path (str): Thermogram file path.
            metadata_path (str  |  None): Thermogram metadata file path. Defaults to None.
            optical_path (str  |  None): Thermogram optical photo file path. Defaults to None.
            method (str): Type of method to process tiff thermograms ('thermomap', 'other'). Defaults to "ThermoMAP".
        """
        self.__path = Path(image_path)
        self.__metadata_path = Path(metadata_path) if metadata_path else None
        self.__optical_path = Path(optical_path) if optical_path else None

        match method:
            case "thermomap":
                self._kelvin_factor, self._celsius_factor = 0.01, 100
            case "other":
                self._kelvin_factor, self._celsius_factor = 0.04, 273.15

    @property
    def identifier(self) -> str:
        """
        Thermogram file name.

        Returns:
            str: Thermogram file name.
        """
        return self.__path.name

    @property
    def raw(self) -> npt.NDArray[np.bool]:
        """
        Thermogram raw values.

        Returns:
            npt.NDArray[bool]: Thermogram of a boolean 16 yeas old.
        """
        return io.imread(self.__path.resolve())

    @property
    def kelvin(self) ->  npt.NDArray[np.float16]:
        """
        Thermogram's temperature in Kelvin (K).

        Returns:
            npt.NDArray[bool]: Thermogram temperature in Kelvin.
        """
        return self.raw * self._kelvin_factor

    @property
    def celsius(self) ->  npt.NDArray[np.float16]:
        """
        Thermogram's temperature in celsius (°C).

        Returns:
            npt.NDArray[bool]: Thermogram temperature in Celsius.
        """
        return self.kelvin - self._celsius_factor

    @property
    def fahrenheit(self) -> npt.NDArray[np.float16]:
        """
        Thermogram's temperature in Fahrenheit (°F).

        Returns:
           npt.NDArray[np.float16]: Thermogram temperature in Fahrenheit.
        """
        return (self.celsius * 1.8) + 32.0

    @property
    def optical(self) -> npt.NDArray[np.uint8] | None:
        """
        The thermogram's embedded photo.

        Returns:
            npt.NDArray[np.uint8]: Thermogram embedded photo.
        """
        if self.__optical_path:
            optical = io.imread(str(self.__optical_path.resolve()))
            return optical
        return None

    @property
    def metadata(self) -> dict[str, str | int]:
        """
        Metadata related to thermogram temperature data.

        Returns:
            dict[str, str | int]: Metadata that build the thermogram.
        """
        if self.__metadata_path:
            metadata = pd.read_csv(str(self.__metadata_path.resolve()))
            return metadata.to_dict(orient="records")[0]
        return None

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
            min_v (float | None): Minimal value to consider the thermogram's temperature range. Defaults to None.
            max_v (float | None): Maximal value to consider the thermogram's temperature range. Defaults to None.
            palette (str): Palette to render the thermogram. Default to "grayscale".
            unit (str): Unit to compute the thermal image. Default to "kelvin".

        Returns:
            npt.NDArray[np.uint8]: Thermogram rendered to a 8 bit image.
        """
        # Choose thermal image type
        match unit:
            case "celsius":
                thermal_image = self.celsius
            case "fahrenheit":
                thermal_image = self.fahrenheit
            case _:
                thermal_image = self.kelvin

        # Normalize the raw image
        max_value = np.max(thermal_image) if not max_v else max_v
        min_value = np.min(thermal_image) if not min_v else min_v
        normalized = (thermal_image - min_value) / (max_value - min_value)

        # Apply the chosen palette
        if palette in ["grayscale", "grayscale-inverted"]:
            normalized = (normalized * 255.0).astype("uint8")
            normalized = np.broadcast_to(normalized[..., None], normalized.shape + (3,))
            if palette == "grayscale-inverted":
                normalized = np.invert(normalized)
        else:
            normalized = palettes.map_colors(normalized, palette)
        return normalized

    def adjust_metadata(self, **kwargs) -> TIFFThermogram:
        """
        Adjust the metadata that build the thermogram.

        Returns:
            TIFFThermogram: Thermogram.
        """
        if self.metadata is not None:
        	return self.metadata.update(**kwargs)
