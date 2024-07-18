from typing import Union

import numpy as np
import numpy.typing as npt

from .abstract_data_processor import AbstractDataProcessor, AbstractFitDataProcessor, AbstractDestructiveDataProcessor
from ..data.types import MIXED_FLOAT_ARRAY

class DataProcessorWrapper:
    def __init__(self, processor : AbstractDataProcessor) -> None:
        self.__need_check = not isinstance(processor, AbstractDestructiveDataProcessor)
        self.__is_checked = False
        self.__processor = processor

    def forward(self, data : Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]) -> Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
        if self.__processor is None:
            raise Exception("Processor is None")
        if isinstance(self.__processor, AbstractFitDataProcessor) and not self.__processor.is_fitted:
            raise Exception("Fit processor must be fitted before using it.")
        if self.__need_check and not self.__is_checked:
            self.__processor._check_reversible()
            self.__is_checked = True
        return self.__processor.forward(data)
    
    def backward(self, data : Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]) -> Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
        if self.__processor is None:
            raise Exception("Processor is None")
        if isinstance(self.__processor, AbstractFitDataProcessor) and not self.__processor.is_fitted:
            raise Exception("Fit processor must be fitted before using it.")
        if self.__need_check and not self.__is_checked:
            self.__processor._check_reversible()
            self.__is_checked = True
        return self.__processor.backward(data)
    
    @property
    def processor(self) -> AbstractDataProcessor:
        return self.__processor