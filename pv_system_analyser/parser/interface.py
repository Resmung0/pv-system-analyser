from abc import ABC, abstractmethod

class AnnotationParser(ABC):
    """
    Interface class responsible for parse annotation data.
    """

    @abstractmethod
    @property
    def masks(self) -> None:
        """
        Method that extract masks from parsed data.
        """
        return NotImplementedError()
    
    @abstractmethod
    @property
    def centers(self) -> None:
        """
        Method that extract centers from parsed data.
        """
        return NotImplementedError()
    
    @abstractmethod
    @property
    def classes(self) -> None:
        """
        Method that extract classes from parsed data.
        """
        return NotImplementedError()

class MetadataParser(ABC):
    """
    Interface class responsible for extracting metadata from files.
    """
    @abstractmethod
    @property
    def metadata(self) -> dict[str, str]:
        """
        Method that extracts metadata from files.
        """
        return NotImplementedError()