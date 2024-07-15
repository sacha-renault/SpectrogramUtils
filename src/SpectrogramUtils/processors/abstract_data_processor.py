from abc import ABC, abstractmethod
from typing import Union, Tuple, List

import numpy as np
import numpy.typing as npt

class AbstractDataProcessor(ABC):
    def _forward(self, data : np.ndarray) -> npt.NDArray[np.float64]:
        return self.forward(data)

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

    def _backward(self, data : np.ndarray) -> npt.NDArray[np.float64]:
        return self.backward(data)

    @abstractmethod
    def backward(self, data : np.ndarray) -> npt.NDArray[np.float64]:
        """Get back to inital state

        Args:
            data (np.ndarray): single processed data

        Returns:
            npt.NDArray[np.float64]: deprocessed data
        """
        ...


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

    def _backward(self, data : np.ndarray) -> npt.NDArray[np.float64]:
        assert self.is_fitted, "Cannot use transformation before fitting the processor"
        return self.backward(data)
    
    def _forward(self, data : np.ndarray) -> npt.NDArray[np.float64]:
        assert self.is_fitted, "Cannot use transformation before fitting the processor"
        return self.forward(data)



