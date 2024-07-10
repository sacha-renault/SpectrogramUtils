from typing import Generator, Union

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

from .abstract_data_processor import AbstractFitDataProcessor

def reshape_to_transform(data : np.ndarray) -> np.ndarray:
    return np.expand_dims(data.flatten(), axis = 1)

class TestAahah(AbstractFitDataProcessor):
    def __init__(self) -> None:
        self.ssc = StandardScaler()
        self.mms = MinMaxScaler(feature_range=(-1, 1))
        super().__init__()

    def forward(self, data: np.ndarray) -> np.ndarray:
        shape = data.shape
        data = self.ssc.transform(reshape_to_transform(data)).reshape(shape)
        data = self.mms.transform(reshape_to_transform(data)).reshape(shape)
        return data
    
    def backward(self, data: np.ndarray) -> np.ndarray:
        shape = data.shape
        data = self.mms.inverse_transform(reshape_to_transform(data)).reshape(shape)
        data = self.ssc.inverse_transform(reshape_to_transform(data)).reshape(shape)
        return data
    
    def fit(self, data: Union[np.ndarray, Generator[np.ndarray, None, None]]):
        shape = data[0].shape

        assert not self.is_fitted, "Fit should be done at once, use generator if the dataset is two large"

        for x in data:
            self.ssc.partial_fit(reshape_to_transform(x))

        for x in data:
            x = self.ssc.transform(reshape_to_transform(x)).reshape(shape)
            self.mms.partial_fit(reshape_to_transform(x))
        
        self.is_fitted = True