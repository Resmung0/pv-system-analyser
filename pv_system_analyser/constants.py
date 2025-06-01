from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt

# Array Data Structures
INT_ARRAY: Final[TypeAlias] = npt.NDArray[np.uint8]
BOOL_ARRAY: Final[TypeAlias] = npt.NDArray[np.bool_]
FLOAT_ARRAY: Final[TypeAlias] = npt.NDArray[np.float16]
STRING_ARRAY: Final[TypeAlias] = npt.NDArray[np.str_]


__all__ = ["INT_ARRAY", "BOOL_ARRAY", "FLOAT_ARRAY", "STRING_ARRAY"]
