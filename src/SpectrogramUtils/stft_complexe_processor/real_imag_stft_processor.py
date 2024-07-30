""" definition of RealImageStftProcessor class"""
from typing import Tuple, List

from .abstract_stft_processor import AbstractStftComplexProcessor
from ..data.types import MixedPrecision2DArray, Complex2DArray

class RealImageStftProcessor(AbstractStftComplexProcessor):
    """Handle convertion in both direction Complex2DArray ↔ MixedPrecision2DArray.
    This StftProcessor split datas through real and imag values

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
        data[2*index] = input_stft.real
        data[2*index + 1] = input_stft.imag

    def real_to_complexe(self,
            data : MixedPrecision2DArray,
            index : int) -> Complex2DArray:
        return data[2*index] + 1j * data[2*index + 1]
