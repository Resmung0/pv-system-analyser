import datetime

import exifread

from pv_system_analyser.parser.interface import MetadataParser


class TIFFMetadataParser(MetadataParser):
    """
    A parser for extracting metadata from TIFF image files. This class provides methods 
    for reading TIFF metadata, extracting relevant information, and correcting any issues
    with the extracted data.
    
    The parser handles the following metadata:
    - Drone information
    - Latitude, longitude, and altitude coordinates
    - Date and time information
    """

    def __init__(self, path: str) -> None:
        self.file_path = path

    @property
    def metadata(self) -> dict[str, str]:
        """
        This method returns extracted metadata, with keys corresponding 
        to the TIFF tags and values representing the tag values.
        
        Returns:
            dict[str, str]: Extracted Thermogram's metadata.
        """
        tags = self._read_path()
        metadata = self._extract_metadata(tags)
        self._correct_metadata(metadata)
        return metadata
    
    def _read_path(self) -> dict[str, str]:
        with open(self.path, 'rb') as file:
            tags = exifread.process_file(file)
        return tags
    
    def _extract_metadata(self, tags: dict[str, str]) -> dict[str, str]:
        """
        Extracts metadata from a dictionary of TIFF tags.
    
        Args:
            tags (dict[str, str]): A dictionary of TIFF tags, 
            where the keys are the tag names and the values 
            are the tag values.
        
        Returns:
            dict[str, str]: A dictionary of extracted metadata, 
            where the keys are the tag names (with some prefixes removed)
            and the values are the tag values.
        """
        metadata = {}
        for tag in tags.keys():
            values = tags[tag].values
            for to_remove in ('Image ', 'GPS ', 'EXIF '):
                tag = tag.replace(to_remove, '')
            metadata[tag] = values
        return metadata
    
    def _correct_metadata(self, metadata: dict[str, str]) -> dict[str, str]:
        """
        Corrects and normalizes the metadata extracted from a TIFF file.
        
        This method performs the following corrections and normalizations:
        - Corrects the 'drone' metadata by moving the value from the 'Tag 0x000B' key to the 'drone' key.
        - Corrects all values in the metadata dictionary, ensuring that single-element lists are converted 
        to their single value, and empty lists are converted to None.
        - Corrects the 'latitude', 'longitude', and 'altitude' coordinates by converting the raw TIFF tag values
        to decimal degrees or meters.
        - Corrects the 'DateTimeOriginal' metadata by parsing the date and time string into a datetime object.
        
        Args:
            metadata (dict[str, str]): The extracted metadata from the TIFF file.
        
        Returns:
            dict[str, str]: The corrected and normalized metadata.
        """
        metadata = self.__correct_drone(metadata)
        metadata = self.__correct_all_values(metadata)
        metadata = self.__correct_coordinate(metadata, 'latitude', ['GPSLatitude', 'GPSLatitudeRe'])
        metadata = self.__correct_coordinate(metadata, 'longitude', ['GPSLongitude', 'GPSLongitudeRef'])
        metadata = self.__correct_coordinate(metadata, 'altitude', ['GPSAltitude'])
        metadata = self.__correct_datetime(metadata)
        return metadata
    
    def __correct_drone(metadata: dict[str, str]) -> dict[str, str]:
        """
        Corrects the 'drone' metadata by moving the value from the 'Tag 0x000B' key to the 'drone' key,
        and removes the 'Tag 0x000B' key.
        
        Args:
            metadata (dict[str, str]): The metadata dictionary to be corrected.
        
        Returns:
            dict[str, str]: The corrected metadata dictionary.
        """
        metadata['drone'] = metadata['Tag 0x000B']
        del metadata['Tag 0x000B']
        return metadata
    
    @staticmethod
    def __correct_all_values(metadata: dict[str, str]) -> dict[str, str]:
        """
        Corrects the values in the metadata dictionary, ensuring that single-element lists are converted
        to their single value, and empty lists are converted to None.
        
        Args:
            metadata (dict[str, str]): The metadata dictionary to be corrected.
        
        Returns:
            dict[str, str]: The corrected metadata dictionary.
        """
        for key in metadata:
            old_value = metadata[key]
            if isinstance(old_value, list):
                match len(old_value):
                    case 1:
                        metadata[key] = old_value[0]
                    case 0:
                        metadata[key] = None
        return metadata
    
    def __correct_coordinate(self, metadata: dict[str, str], coordinate: str, tags: list[str]) -> dict[str, str]:
        """
        Corrects the coordinate values in the metadata dictionary.
        
        Args:
            metadata (dict[str, str]): The metadata dictionary to be corrected.
            coordinate (str): The name of the coordinate to be corrected ('latitude', 'longitude', or 'altitude').
            tags (list[str]): The keys in the metadata dictionary that contain the coordinate values.
        
        Returns:
            dict[str, str]: The corrected metadata dictionary.
        """
        values = [metadata.get(tag) for tag in tags]
        match coordinate:
            case 'latitude' | 'longitude':        
                metadata[coordinate] = self.__convert(*values)
            case _:
                if values[0] is not None:
                    metadata[coordinate] = values[0].num / values[0].den        
        for tag in tags:
            del metadata[tag]
        return metadata
    
    def __convert(self, value: float, reference: str) -> float:
        """
        Converts a value to degrees, taking into account the reference direction (North/South or East/West).
        
        Args:
            value (float): The value to be converted to degrees.
            reference (str): The reference direction, either 'S', 'W', or None.
        
        Returns:
            float: The value converted to degrees, with the reference direction applied.
        """
        if value is not None:
            value = self.__convert_to_degress(value)
            if reference == 'S' or reference == 'W':
                value = -value
        return value
    
    @staticmethod
    def __convert_to_degress(value) -> float:
        """
        Converts a value represented as a tuple of three numbers (degrees, minutes, seconds)
        to a floating-point value in degrees.
        
        Args:
            value (Tuple[float, float, float]): A tuple of three numbers representing the degrees,
            minutes, and seconds of a coordinate value.
        
        Returns:
            float: The coordinate value in degrees.
        """        
        d = float(value[0].num) / float(value[0].den)
        m = float(value[1].num) / float(value[1].den)
        s = float(value[2].num) / float(value[2].den)
        return d + (m / 60.0) + (s / 3600.0)
    
    @staticmethod
    def __correct_datetime(metadata: dict[str, str]) -> dict[str, str]:
        """
        Corrects the 'DateTimeOriginal' metadata field by parsing the value as a datetime object.
        
        Args:
            metadata (dict[str, str]): The metadata dictionary to be corrected.
        
        Returns:
            dict[str, str]: The corrected metadata dictionary.
        """
        metadata['DateTimeOriginal'] = datetime.strptime(
            metadata['DateTimeOriginal'], '%Y:%m:%d %H:%M:%S'
        )
        return metadata
