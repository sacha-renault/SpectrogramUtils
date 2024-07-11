from abc import ABC, abstractmethod
from typing import Union

import numpy as np

class AbstractDataProcessor(ABC):
    def f_forward(self, data : np.ndarray) -> np.ndarray:
        return self.forward(data)

    @abstractmethod
    def forward(self, data : np.ndarray) -> np.ndarray:
        ...

    def f_backward(self, data : np.ndarray) -> np.ndarray:
        return self.backward(data)

    @abstractmethod
    def backward(self, data : np.ndarray) -> np.ndarray:
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
    def load(self, file : Union[str, list[str]]) -> None:
        """#### Restaure the processor to a saved states, it should set is_fitted to True. 

        #### Args:
            - file (Union[str, list[str]]): file or files to restaure a processor states. 
        """
        ...

    @abstractmethod
    def fit(self, fit_data : np.ndarray) -> None:
        """#### Fit the processor to training datas. It should set is_fitted to True

        #### Args:
            - fit_data (np.ndarray): training data
        """
        ...

    @abstractmethod
    def save(self, file : Union[str, list[str]]) -> None:
        """#### Load the current state of the processor into a file

        #### Args:
            - file (Union[str, list[str]]): file or files to save a processor states. 
        """
        ...

    def f_backward(self, data : np.ndarray) -> np.ndarray:
        assert self.is_fitted, "Cannot use transformation before fitting the processor"
        return self.backward(data)
    
    def f_forward(self, data : np.ndarray) -> np.ndarray:
        assert self.is_fitted, "Cannot use transformation before fitting the processor"
        return self.forward(data)



