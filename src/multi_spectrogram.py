from typing import Union

import numpy as np
import librosa
from numba import njit
from matplotlib.axes import Axes

from .data import DisplayType
from .config import Config

class MultiSpectrogram:
    @classmethod
    def from_stfts(cls, config : Config, *stfts : np.ndarray):
        # assert same shape
        assert all(stft.shape == stfts[0].shape for stft in stfts), "All stfts shape should be the same"

        # create a data
        data = np.zeros(shape = (2 * len(stfts), *stfts[0].shape))
        for i, stft in enumerate(stfts):
            data[2*i] = stft.real
            data[2*i + 1] = stft.imag
        
        # Instanciate the class with the multi spectro 
        return cls(config, data)

    def __init__(self, config : Config, data : np.ndarray) -> None:
        # define config
        self.__conf = config

        # define data
        self.__data = data

    
    @property
    def data(self) -> np.ndarray:
        """#### return a stereo spectrograme. Indexes precisions :
        - 2*n : amplitude of stft at index n
        - 2*n + 1 : phase of stft at index n

        #### Returns:
            - np.ndarray: multi spectrogram
        """
        return self.__data
    
    @property
    def num_stfts(self) -> int:
        return self.__data.shape[0] // 2
    
    @property
    def shape(self):
        return self.__data.shape
    
    def show_image_on_axis(self, axis : Axes, display_type : DisplayType = DisplayType.MEAN, index : Union[int, None] = None, **axes_kwargs):
        """#### Show the spectrogram on the given axis 

        #### Args:
            - axis (axes.Axes): Axis to display the stereo spectrogram
            - display_type (DisplayType, optional): How to display the stereo spectrogram. Defaults to DisplayType.MEAN.
        """
        # init new data
        display_data = np.zeros_like(self.data[0])

        if display_type == DisplayType.STACK:
            raise Exception("Display type STACK is only available for wave display")

        elif display_type == DisplayType.MEAN:
            display_data += np.mean(self.get_stft(), axis = 0)
        
        elif display_type == DisplayType.INDEX:
            if index is None:
                display_data += self.get_stft(index)
            else:
                raise Exception("Can't display index if no index is provided")
        
        else:
            raise Exception("Unknown display type")

        axis.imshow(display_data, **axes_kwargs)

    def show_wave_on_axis(self, axis : Axes, display_type : DisplayType = DisplayType.MEAN, index : Union[int, None] = None, **kwargs):
        if display_type == DisplayType.STACK:
            axis.plot(self.get_waves(), **kwargs)
            

        elif display_type == DisplayType.MEAN:
            waves = self.get_waves()
            avg_wave = np.mean(waves, axis = 0)
            axis.plot(avg_wave, **kwargs)

        elif display_type == DisplayType.INDEX:
            if index is None:
                wave = self.get_wave(index)
                axis.plot(wave, **kwargs)
            else:
                raise Exception("Can't display index if no index is provided")

        else:
            raise Exception("Unknown display type")
    
    def get_stft(self, index : int) -> np.ndarray:
        return np.abs(self.data[2*index] + 1j * self.data[2*index + 1])
    
    @njit
    def get_stfts(self) -> np.ndarray:
        stfts = np.zeros(shape=(self.num_stfts, self.data[0].shape))
        for i in range(self.num_stfts):
            stfts[i] = self.get_stft(i)

    def get_wave(self, index : int) -> np.ndarray:
        stft = self.get_stft(index)
        wave = librosa.istft(stft, **self.__conf.get_istft_kwargs())
        return wave
    
    @njit
    def get_waves(self) -> np.ndarray:
        waves = None
        for i in range(self.num_stfts): 
            wave = self.get_wave(i)
            if waves == None:
                waves = np.zeros(shape=(self.num_stfts, *wave.shape))
            waves[i] = wave
        return waves
