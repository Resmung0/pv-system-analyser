from .interface import Parser
from .dataset.pst_annotation_parser import PSTAnnotationParser


def choose_parser(
    dataset_name: str, data: dict[str, str | int]
) -> Parser:
    """
    Helper function to choose segmentation dataset proper parser.
    """
    match dataset_name:
        case 'Photovoltaic System Thermography':
            parser = PSTAnnotationParser(data)
        case 'Photovoltaic System O&M Inspection':
            ...
        case 'Photovoltaic System Thermal Inspection':
            ...
        case _:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")
    return parser

