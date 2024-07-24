from abc import ABC, abstractmethod
from typing import Union, Tuple, List

import numpy as np
import numpy.typing as npt

class AbstractStftComplexProcessor(ABC):
    @abstractmethod
    def num_stfts(self, data : npt.NDArray) -> int:
        """[summary] get the number of stft from data shape
        """
        ...


    @abstractmethod
    def shape(self, input_stfts : List[npt.NDArray[np.complex128]]) -> Tuple:
        """[summary] return the shape the the stored data should have
        """
        ...

    @abstractmethod
    def complexe_to_real(self, 
        input_stft : npt.NDArray[np.complex128], 
        data : Union[npt.NDArray[np.float64], npt.NDArray[np.float32]],
        index : int) -> None:
        """[summary] transform the stft into real value and store it into data
        """
        ...

    @abstractmethod
    def real_to_complexe(self, 
        data : Union[npt.NDArray[np.float64], npt.NDArray[np.float32]],
        index : int) -> npt.NDArray[np.float64]:
        """[summary] transform the stft into complexe value and return a single stft
        """
        ...