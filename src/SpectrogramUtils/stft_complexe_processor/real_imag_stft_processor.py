from typing import Union, Tuple, List

import numpy as np
from numpy._typing import NDArray
import numpy.typing as npt

from .abstract_stft_processor import AbstractStftComplexProcessor

class RealImageStftProcessor(AbstractStftComplexProcessor):
    def num_stfts(self, data : npt.NDArray) -> int:
        return data.shape[0] // 2
    
    def shape(self, input_stfts : List[npt.NDArray[np.complex128]]) -> Tuple:
        return (2 * len(input_stfts), *input_stfts[0].shape)
    
    def complexe_to_real(self, input_stft : npt.NDArray[np.complex128], 
        data : Union[npt.NDArray[np.float64], npt.NDArray[np.float32]],
        index : int) -> None:
        data[2*index] = input_stft.real
        data[2*index + 1] = input_stft.imag
    
    def real_to_complexe(self, 
        data : Union[npt.NDArray[np.float64], npt.NDArray[np.float32]],
        index : int) -> npt.NDArray[np.float64]:
        return data[2*index] + 1j * data[2*index + 1]