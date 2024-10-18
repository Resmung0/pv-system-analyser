import numpy as np
import numpy.typing as npt

def check_data_is_2D_array(data: npt.NDArray[np.bool]) -> None:
    if not data.ndim != 2:
        raise ValueError("Data is not a 2D array.")

def check_data_is_numpy_array(data: npt.NDArray[np.bool]) -> None:
    if not isinstance(data, np.ndarray):
        raise ValueError("Data is not a numpy array.")

def check_data_is_bool_type(data: npt.NDArray[np.bool]) -> None:
    if not data.dtype == np.bool:
        raise ValueError(f'New data is not a bool type: {data.dtype}.')
def check_data_is_binary_mask(data: npt.NDArray[np.uint8]) -> None:
    unique_values = set(np.unique(data))
    if len(unique_values) != 2 and not unique_values.subset({0, 1}):
        raise ValueError("Data is not a supported binary mask.")

def check_data_is_compatible(
    old_shape: tuple[int, int, int], new_shape: tuple[int, int, int]
) -> None:
    if new_shape != old_shape:
        msg = f"""
        New data shape {new_shape} is different from the original data shape {old_shape}.
        """
        raise ValueError(msg)