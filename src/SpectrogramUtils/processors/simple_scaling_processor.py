from typing import Generator, Union, Tuple

import numpy as np
from numpy._typing import NDArray
import numpy.typing as npt

from .abstract_data_processor import AbstractDataProcessor

class SimpleScalingProcessor(AbstractDataProcessor):
    def __init__(self, scale_factor_amplitude : int):
        self.sfa = scale_factor_amplitude

    def forward(self, data: np.ndarray) -> npt.NDArray[np.float64]:
        new_data = data.copy()
        new_data[::2] /= self.sfa
        new_data[1::2] += np.pi
        new_data[1::2] /= 2
        return new_data
    
    def backward(self, data: NDArray) -> npt.NDArray[np.float64]:
        new_data = data.copy()
        new_data[::2] *= self.sfa
        new_data[1::2] *= 2
        new_data[1::2] -= np.pi
        return new_data
