""" Define types """

from typing import Union, Callable

import numpy as np
import numpy.typing as npt

# Array types
MixedPrecision2DArray = Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]
Complex2DArray = npt.NDArray[np.float32]
Complex3DArray = npt.NDArray[np.float32]

# Delegate (functions)
AudioPaddingFunction = Callable[[np.ndarray, int], np.ndarray]
