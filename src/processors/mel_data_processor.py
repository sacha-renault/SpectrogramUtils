import librosa
import numpy as np

from .abstract_data_processor import AbstractDataProcessor

class MelDataProcessor(AbstractDataProcessor):
    def __init__(self, sr : int) -> None:
        self.sr = sr

    def forward(self, data: np.ndarray) -> np.ndarray:
        new_data = None
        for i in range(data.shape[0] // 2):
            spec = data[2*i] + 1j * data[2*i + 1]
            mel_dot = librosa.filters.mel(sr=self.sr, n_fft = (spec.shape[0] - 1) * 2, n_mels=(spec.shape[0] - 1) * 2)
            res = np.dot(mel_dot, np.abs(spec) ** 2)
            if new_data is None:
                new_data = np.zeros((data.shape[0] // 2, *res.shape))
            new_data[i] = res
        return new_data
    
    def backward(self, data: np.ndarray) -> np.ndarray:
        new_data = None
        for i in range(data.shape[0]):
            res = librosa.feature.inverse.mel_to_stft(new_data, power = 2, n_fft = (data.shape[0] - 1) * 2, n_mels=(data.shape[0] - 1) * 2)
            if new_data is None:
                new_data = np.zeros((data.shape[0] * 2, *res.shape))
            new_data[2*i] = np.real(res)
            new_data[2*i + 1] = np.imag(res)
        return new_data

