from pathlib import Path
import logging
import os

import kagglehub as kh
from dotenv import load_dotenv

from pv_system_analyser.pipeline.validation import check_dir


load_dotenv()

def change_directory(dir: Path) -> None:
    os.chdir(dir)

def exclude_remaining_files(path: str) -> None:
    """
    Exclude remaining files.
    """
    dir_path = Path(path).joinpath("archive")
    for file in ["README.md", "LICENSE"]:
        dir_path.joinpath(file).unlink()
    file_path = dir_path.joinpath("2020-02-14_InfraredSolarModules/InfraredSolarModules")
    file_path.rename("data/raw/infrared-solar-modules")
    dir_path.rmdir()


@check_dir
def download_infrared_solar_modules_dataset(dst_dir: str) -> None:
    logging.info('Changing directory to ', dst_dir)
    change_directory(Path(dst_dir))
    logging.info('Downloading Infrared Solar Modules Dataset')
    kh.dataset_download("marcosgabriel/infrared-solar-modules")
    logging.info('Excluding remaining files')
    exclude_remaining_files("assets/data/raw")