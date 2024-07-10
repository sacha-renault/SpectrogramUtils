from typing import Callable

import numpy as np
import soundfile as sf
import librosa

from .config import Config
from .processors.abstract_data_processor import AbstractDataProcessor
from .multi_spectrogram import MultiSpectrogram
from .data import AudioPadding

class SpectrogramFactory:
    def __init__(self, 
                config : Config,
                data_processor : AbstractDataProcessor = None,
                audio_padder : Callable[[np.ndarray, int], np.ndarray] = AudioPadding.NONE
                ) -> None:
        # Set the config
        self.__config = config

        # Set the data processor
        self.__processor = data_processor

        # Set the padding functino
        self.__audio_padder = audio_padder

    def get_spectrogram_from_audio(self, audio_array : np.ndarray) -> MultiSpectrogram:
        # Get the channel number
        num_channels = audio_array.shape[0]

        # padding
        if self.__audio_padder is not None:
            audio_array = self.__audio_padder(audio_array, self.__config.audio_length)

        # If audio is mono, we can copy 
        if num_channels == 1 and self.__config.num_channel > 1:
            audio_array = np.stack([audio_array]*self.__config.num_channel, axis = 0)

        # If number of channels and desired channel number fit, we can do stfts
        if num_channels == self.__config.num_channel:
            pass
            # TODO normal processing
        
        else:
            raise Exception(f"Cannot handle data with {num_channels} channels")

    