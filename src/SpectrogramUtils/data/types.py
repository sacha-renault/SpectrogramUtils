from typing import Union

import numpy as np
import numpy.typing as npt

MIXED_FLOAT_ARRAY = Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]
MIXED_FLOAT_ARRAY.__annotations__ = npt.NDArray[Union[np.float32, np.float64]]