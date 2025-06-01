from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class AnnotationParser(ABC):
    """
    Interface class responsible for parse annotation data.
    """

    @property
    @abstractmethod
    def masks(self) -> npt.NDArray[np.uint8]:
        """
        Returns binary masks extracted from parsed annotation data.

        Returns:
            NDArray[uint8]: Array of shape (N, H, W) containing N binary masks
            where each mask has height H and width W. Values are 0 or 1.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def centers(self) -> npt.NDArray[np.uint8]:
        """
        Returns center coordinates for each detected object.

        Returns:
            NDArray[uint8]: Array of shape (N,2) containing (x,y) center coordinates
            for N detected objects.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def classes(self) -> npt.NDArray[np.str_]:
        """
        Returns class labels for each detected object.

        Returns:
            NDArray[str]: Array of shape (N,) containing categorical class labels
            as strings for N detected objects.
        """
        raise NotImplementedError()


class MetadataParser(ABC):
    """
    Interface class responsible for extracting metadata from files.
    """

    @property
    @abstractmethod
    def metadata(self) -> None:
        """
        Method that extracts metadata from files.
        """
        raise NotImplementedError()
