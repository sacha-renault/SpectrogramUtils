""" Module that define base class for data processor"""
from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np

from ..exceptions.lib_exceptions import BrokenProcessorException
from ..data.types import MixedPrecision2DArray

class AbstractDataProcessor(ABC):
    """ Base class for data processor """
    @abstractmethod
    def forward(self, data : MixedPrecision2DArray) -> MixedPrecision2DArray:
        """Preprocess datas, transformation must be reversible to get back to initial state in
        backward (i.e. self.backward(self.forward(data)) must be same as data)

        Args:
            data (MixedPrecision2DArray): single data

        Returns:
            MixedPrecision2DArray: processed data
        """

    @abstractmethod
    def backward(self, data : MixedPrecision2DArray) -> MixedPrecision2DArray:
        """Get back to inital state

        Args:
            data (MixedPrecision2DArray): single processed data

        Returns:
            MixedPrecision2DArray: deprocessed data
        """

    def check_reversible(self, precision : float = 1e-9) -> None:
        """Check if the data processor is reversible with a certain precision

        Raises:
            BrokenProcessorException: if the data processor isn't reversible
        """
        rnd_data = np.random.rand(1,4,256,256)
        rnd_data_retrieval = self.backward(self.forward(rnd_data))
        err = np.mean(np.abs(rnd_data - rnd_data_retrieval))
        if rnd_data.shape != rnd_data_retrieval.shape or err > precision:
            raise BrokenProcessorException(f"\
                The data processor doesn't retreive the data properly. Max err : {1e-15}, \
                found : {err}. Considere using a AbstractDestructiveDataProcessor")


class AbstractFitDataProcessor(AbstractDataProcessor, ABC):
    """ Base class for fit processor """
    def __init__(self) -> None:
        self.__is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """ Property that says if the processor is fitted"""
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

    @abstractmethod
    def fit(self, fit_data : np.ndarray) -> None:
        """Fit the processor to training datas. It should set is_fitted to True

        Args:
            fit_data (np.ndarray): training data
        """

    @abstractmethod
    def save(self, file : Union[str, List[str]]) -> None:
        """Save the current state of the processor into a file

        Args:
            file (Union[str, list[str]]): file or files to save a processor states.
        """


class AbstractDestructiveDataProcessor(AbstractDataProcessor, ABC):
    """#### A data processor that doesn't have a backward pass
    """
    def backward(self, _) -> MixedPrecision2DArray:
        raise BrokenProcessorException("Destructive Data Processor cannot backward")
