""" definition of AbstractStftComplexProcessor class"""
from abc import ABC, abstractmethod
from typing import Tuple, List

from ..data.types import MixedPrecision2DArray, Complex2DArray

class AbstractStftComplexProcessor(ABC):
    """Base abstract class for StftComplexProcessors.
    Handle convertion in both direction Complex2DArray ↔ MixedPrecision2DArray.

    Methods:
        - complexe_to_real
        - real_to_complexe
        - shape
        - num_stfts
    """
    @abstractmethod
    def num_stfts(self, data : Complex2DArray) -> int:
        """get the number of stft from data shape
        """


    @abstractmethod
    def shape(self, input_stfts : List[Complex2DArray]) -> Tuple:
        """return the shape the the stored data should have
        """

    @abstractmethod
    def complexe_to_real(self,
        input_stft : Complex2DArray,
        data : MixedPrecision2DArray,
        index : int) -> None:
        """transform the stft into real value and store it into data
        """

    @abstractmethod
    def real_to_complexe(self,
        data : MixedPrecision2DArray,
        index : int) -> Complex2DArray:
        """transform the stft into complexe value and return a single stft
        """
