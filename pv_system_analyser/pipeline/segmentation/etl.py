import kagglehub as kh
from dotenv import load_dotenv


load_dotenv()

def download_photovoltaic_system_thermography_dataset() -> None:
    """
    Donwload Photovoltaic System Thermography dataset from Kaggle platform
    """
    kh.dataset_download("marcosgabriel/photovoltaic-system-thermography")

def download_photovoltaic_system_o_and_m_inspection_dataset() -> None:
    """
    Donwload Photovoltaic System O&M Inspection dataset from Kaggle platform
    """
    kh.dataset_download("marcosgabriel/photovoltaic-system-o-and-m-inspection")


