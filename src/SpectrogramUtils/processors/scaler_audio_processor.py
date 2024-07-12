from typing import Generator, Union, Tuple
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import numpy.typing as npt

from .abstract_data_processor import AbstractFitDataProcessor
from ..exceptions.lib_exceptions import UnknownProcessorSaveFileDataException

def reshape_to_transform(data : np.ndarray) -> npt.NDArray[np.float64]:
    return np.expand_dims(data.flatten(), axis = 1)

class MeanStandardScaler(StandardScaler):
    def __init__(self, mean_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mean_value = mean_value

    def transform(self, X, copy=None):
        X_scaled = super().transform(X, copy)
        return X_scaled + self.mean_value

    def inverse_transform(self, X, copy=None):
        X_unscaled = super().inverse_transform(X, copy)
        return X_unscaled - self.mean_value


class ScalerAudioProcessor(AbstractFitDataProcessor):
    def __init__(self, target_mean : float = 0, feature_range : Tuple[float, float] = (-1 ,1)) -> None:
        self.ssc = MeanStandardScaler(target_mean)
        self.mms = MinMaxScaler(feature_range=feature_range)
        super().__init__()

    def forward(self, data: np.ndarray) -> npt.NDArray[np.float64]:
        shape = data.shape
        data = self.ssc.transform(reshape_to_transform(data)).reshape(shape)
        data = self.mms.transform(reshape_to_transform(data)).reshape(shape)
        return data
    
    def backward(self, data: np.ndarray) -> npt.NDArray[np.float64]:
        shape = data.shape
        data = self.mms.inverse_transform(reshape_to_transform(data)).reshape(shape)
        data = self.ssc.inverse_transform(reshape_to_transform(data)).reshape(shape)
        return data
    
    def fit(self, data: Union[np.ndarray, Generator[np.ndarray, None, None]]):
        shape = data[0].shape

        assert not self.is_fitted, "Fit should be done at once, use generator if the dataset is too large"

        for x in data:
            self.ssc.partial_fit(reshape_to_transform(x))

        for x in data:
            x = self.ssc.transform(reshape_to_transform(x)).reshape(shape)
            self.mms.partial_fit(reshape_to_transform(x))
        
        self.is_fitted = True

    def save(self, file: str) -> None:
        data = {
            'ssc': self.ssc,
            'mms': self.mms
        }
        with open(file, 'wb') as f_out:
            pickle.dump(data, f_out)

    def load(self, file: str) -> None:
        with open(file, 'rb') as f_in:
            data = pickle.load(f_in)

        if data:
            self.ssc = data['ssc']
            self.mms = data['mms']
            self.is_fitted = True
        else:
            raise UnknownProcessorSaveFileDataException("Couldn't find datas for saved file")