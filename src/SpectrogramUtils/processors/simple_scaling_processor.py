""" Module that define SimpleScalingProcessor"""
import numpy as np
from numpy._typing import NDArray

from .abstract_data_processor import AbstractDataProcessor

class SimpleScalingProcessor(AbstractDataProcessor):
    """
    Simple scaling that is useful to recenter the phase when STFT process is Magnitude/Phase.
    """
    def __init__(self, scale_magnitude: float, scale_phase: float, phase_shift: float = np.pi):
        """
        Initialize the SimpleScalingProcessor.

        Parameters:
        scale_magnitude (float): The magnitude scaling factor.
        scale_phase (float): The phase scaling factor.
        phase_shift (float, optional): The phase shift value. Default is np.pi.
        """
        self.scale_magnitude = scale_magnitude
        self.scale_phase = scale_phase
        self.phase_shift = phase_shift

    def forward(self, data: np.ndarray) -> NDArray[np.float64]:
        """
        Apply the forward scaling transformation to the data.

        Parameters:
        data (np.ndarray): The input data array.

        Returns:
        NDArray[np.float64]: The transformed data array.
        """
        new_data = data.copy()
        new_data[::2] /= self.scale_magnitude
        new_data[1::2] += self.phase_shift
        new_data[1::2] /= self.scale_phase
        return new_data.astype(np.float64)

    def backward(self, data: NDArray) -> NDArray[np.float64]:
        """
        Apply the backward scaling transformation to the data.

        Parameters:
        data (np.ndarray): The transformed data array.

        Returns:
        NDArray[np.float64]: The original data array.
        """
        new_data = data.copy()
        new_data[::2] *= self.scale_magnitude
        new_data[1::2] *= self.scale_phase
        new_data[1::2] -= self.phase_shift
        return new_data.astype(np.float64)
