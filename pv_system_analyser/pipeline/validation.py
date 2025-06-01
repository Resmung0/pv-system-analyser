from typing import Callable
import logging


def check_dir(fn: Callable) -> Callable:
    def wrapper(dir: str) -> None:
        checked_dir = Path(dir)
        if not checked_dir.exists():
            logging.warning(
                f"Directory {checked_dir.as_posix()} does not exist. Trying to create the directory."
            )
            checked_dir.mkdir()
        if not checked_dir.is_dir():
            msg = f"{checked_dir.as_posix()} it's not a directory"
            logging.error(msg)
            raise ValueError(msg)
        fn(dir)
    return wrapper
