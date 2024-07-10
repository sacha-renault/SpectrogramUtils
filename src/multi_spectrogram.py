from typing import Union

import numpy as np
import librosa
import soundfile as sf
from matplotlib.axes import Axes

from .data import DisplayType
from .config import Config
from .processors.abstract_data_processor import AbstractDataProcessor

class MultiSpectrogram:
    @classmethod
    def from_stfts(cls, config : Config, processor : AbstractDataProcessor = None, *stfts : np.ndarray):
        # assert same shape
        assert all(stft.shape == stfts[0].shape for stft in stfts), "All stfts shape should be the same"

        # create a data
        data = np.zeros(shape = (2 * len(stfts), *stfts[0].shape))
        for i, stft in enumerate(stfts):
            data[2*i] = stft.real
            data[2*i + 1] = stft.imag
        
        # Instanciate the class with the multi spectro 
        return cls(config, processor, data)

    def __init__(self, config : Config, processor : AbstractDataProcessor, data : np.ndarray) -> None:
        # define config
        self.__conf = config

        # Set processor
        self.__processor = processor 

        # define data
        self.__data = data

    
    def to_data(self, process_data : bool = False) -> np.ndarray:
        """#### return a stereo spectrogram. Indexes precisions :
        - 2*n : amplitude of stft at index n
        - 2*n + 1 : phase of stft at index n

        #### Args:
            - process (bool): if data need to be processed

        #### Returns:
            - np.ndarray: multi spectrogram
        """
        if process_data:
            if self.__processor is not None:
                return self.__processor.forward(self.__data)
            else:
                raise Exception("Askep processed data without providing a processor")
        else:
            return self.__data
        
    
    def get_amplitudes(self) -> np.ndarray:
        pass
        
    
    @property
    def num_stfts(self) -> int:
        """#### Number of stfts in the spectrogram, equivalent to the number of channels in the audio.
        """
        return self.__data.shape[0] // 2
    
    @property
    def shape(self):
        return self.__data.shape
    
    def show_image_on_axis(self, 
                           axis : Axes, 
                           display_type : DisplayType = DisplayType.MEAN, 
                           index : Union[int, None] = None, 
                           use_processor : bool = False,
                           *axes_args, 
                           **axes_kwargs):
        """#### Show the stft on the given axis 

        #### Args:
            - axis (axes.Axes): Axis to display the stereo spectrogram
            - display_type (DisplayType, optional): How to display the stereo spectrogram. Defaults to DisplayType.MEAN.
        """
        # init new data
        display_data = np.zeros_like(self.to_data(use_processor)[0], dtype=np.complex128)

        if display_type == DisplayType.STACK:
            raise Exception("Display type STACK is only available for wave display")

        elif display_type == DisplayType.MEAN:
            display_data += np.mean(self.to_data(use_processor), axis = 0)
        
        elif display_type == DisplayType.INDEX:
            if index is not None:
                display_data += self.get_stft(index, use_processor)
            else:
                raise Exception("Can't display index if no index is provided")
        
        else:
            raise Exception("Unknown display type")

        axis.imshow(np.abs(display_data), *axes_args, **axes_kwargs)

    def show_wave_on_axis(self, 
                          axis : Axes, 
                          display_type : DisplayType = DisplayType.STACK, 
                          index : Union[int, None] = None, 
                          *axes_args, 
                          **axes_kwargs):
        """#### Show the wave shape on a given axis 

        #### Args:
            - axis (axes.Axes): Axis to display the stereo spectrogram
            - display_type (DisplayType, optional): How to display the wave. Defaults to DisplayType.STACK.
        """
        if display_type == DisplayType.STACK:
            axis.plot(self.get_waves().transpose(), *axes_args, **axes_kwargs)            

        elif display_type == DisplayType.MEAN:
            waves = self.get_waves()
            avg_wave = np.mean(waves.transpose(), axis = 0)
            axis.plot(avg_wave, *axes_args, **axes_kwargs)

        elif display_type == DisplayType.INDEX:
            if index is not None:
                wave = self.get_wave(index)
                axis.plot(wave, *axes_args, **axes_kwargs)
            else:
                raise Exception("Can't display index if no index is provided")

        else:
            raise Exception("Unknown display type")
    
    def get_stft(self, index : int, use_processor : bool = False) -> np.ndarray:
        """#### Get a stft at a specified index

        #### Args:
            - index (int): index of channel to get the stft

        #### Returns:
            - np.ndarray: stft at the requested index
        """
        return self.to_data(use_processor)[2*index] + 1j * self.to_data(use_processor)[2*index + 1]

    def get_wave(self, index : int) -> np.ndarray:
        """#### Get the wave shape for the channel at the requested index

        #### Args:
            - index (int): requested channel index

        #### Returns:
            - np.ndarray: 1D wave shape
        """
        stft = self.get_stft(index, False)
        wave = librosa.istft(stft, **self.__conf.get_istft_kwargs())
        return wave
    
    def get_waves(self) -> np.ndarray:
        """#### Get the wave shape for all channels

        #### Returns:
            - np.ndarray: n-Dimentional wave shape
        """
        waves = None
        for i in range(self.num_stfts): 
            wave = self.get_wave(i)
            if waves is None:
                waves = np.zeros(shape=(self.num_stfts, *wave.shape))
            waves[i] = wave
        return waves
    
    def save_to_file(self, file_name : str) -> None:
        if not file_name.endswith(".wav"):
            file_name += ".wav"
        if self.num_stfts <=2:
            sf.write(file_name, self.get_waves().transpose(), samplerate=self.__conf.sample_rate)
        else:
            raise Exception("Cannot save audio if it isn't mono or stereo")
