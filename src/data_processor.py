from abc import ABC, abstractmethod

import numpy as np

class AbstractDataProcessor(ABC):
    @abstractmethod
    def forward(self, data : np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, data : np.ndarray) -> np.ndarray:
        ...