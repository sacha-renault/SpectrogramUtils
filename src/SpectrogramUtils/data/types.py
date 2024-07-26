""" Define types """

from typing import Union

import numpy as np
import numpy.typing as npt

MixedFloatArray = Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]
MixedFloatArray.__annotations__ = npt.NDArray[Union[np.float32, np.float64]]
