from abc import ABC, abstractmethod
from typing import Union, Tuple, List

import numpy as np
import numpy.typing as npt

class AbstractDataProcessor(ABC):
    @abstractmethod
    def forward(self, data : np.ndarray) -> npt.NDArray[np.float64]:
        """Preprocess datas, transformation must be reversible to get back to initial state in backward
        (i.e. self.backward(self.forward(data)) must be same as data)

        Args:
            data (np.ndarray): single data

        Returns:
            npt.NDArray[np.float64]: processed data
        """
        ...

    @abstractmethod
    def backward(self, data : np.ndarray) -> npt.NDArray[np.float64]:
        """Get back to inital state

        Args:
            data (np.ndarray): single processed data

        Returns:
            npt.NDArray[np.float64]: deprocessed data
        """
        ...

    def _check_reversible(self) -> None:
        rnd_data = np.random.rand(1,4,256,256)
        rnd_data_retrieval = self.backward(self.forward(rnd_data))
        err = np.mean(np.abs(rnd_data - rnd_data_retrieval))
        if rnd_data.shape != rnd_data_retrieval.shape or err > 1e-9:
            raise Exception(f"The data processor doesn't retreive the data properly. Max err : {1e-15}, found : {err}. Considere using a AbstractDestructiveDataProcessor")


class AbstractFitDataProcessor(AbstractDataProcessor, ABC):
    def __init__(self) -> None:
        self.__is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self.__is_fitted
    
    @is_fitted.setter
    def is_fitted(self, value : bool):
        self.__is_fitted = value
    
    @abstractmethod
    def load(self, file : Union[str, List[str]]) -> None:
        """Restaure the processor to a saved states, it should set is_fitted to True. 

        Args:
            file (Union[str, list[str]]): file or files to restaure a processor states. 
        """
        ...

    @abstractmethod
    def fit(self, fit_data : np.ndarray) -> None:
        """Fit the processor to training datas. It should set is_fitted to True

        Args:
            fit_data (np.ndarray): training data
        """
        ...

    @abstractmethod
    def save(self, file : Union[str, List[str]]) -> None:
        """Save the current state of the processor into a file

        Args:
            file (Union[str, list[str]]): file or files to save a processor states. 
        """
        ...
    

class AbstractDestructiveDataProcessor(AbstractDataProcessor, ABC):
    """#### A data processor that doesn't have a backward pass
    """
    def backward(self, _) -> npt.NDArray[np.float64]:
        raise Exception("Destructive Data Processor cannot backward")