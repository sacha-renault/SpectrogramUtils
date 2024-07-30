""" definition of MagnitudePhaseStftProcessor class"""
from typing import Tuple, List

import numpy as np

from ..data.types import MixedPrecision2DArray, Complex2DArray

from .abstract_stft_processor import AbstractStftComplexProcessor

class MagnitudePhaseStftProcessor(AbstractStftComplexProcessor):
    """Handle convertion in both direction Complex2DArray ↔ MixedPrecision2DArray.
    This StftProcessor split datas through magnitude and phase values

    Methods:
        - complexe_to_real
        - real_to_complexe
        - shape
        - num_stfts
    """
    def num_stfts(self, data : Complex2DArray) -> int:
        return data.shape[0] // 2

    def shape(self, input_stfts : List[Complex2DArray]) -> Tuple:
        return (2 * len(input_stfts), *(input_stfts[0].shape))

    def complexe_to_real(self,
            input_stft : Complex2DArray,
            data : MixedPrecision2DArray,
            index : int) -> None:
        data[2*index] = np.abs(input_stft)
        data[2*index + 1] = np.angle(input_stft)

    def real_to_complexe(self,
            data : MixedPrecision2DArray,
            index : int) -> Complex2DArray:
        return data[2*index] * np.exp(1j * data[2*index + 1])
