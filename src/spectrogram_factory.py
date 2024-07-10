from typing import Callable, Union

import numpy as np
import soundfile as sf
import librosa

from .config import Config
from .processors.abstract_data_processor import AbstractDataProcessor, AbstractFitDataProcessor
from .multi_spectrogram import MultiSpectrogram
from .data import AudioPadding
from .utils import get_multi_stft

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

        # Check if config is correct
        if self.__audio_padder is not None and self.__config.audio_length is None:
            raise Exception("Audio Padder can't be userd without a audio_length configured in the config object.")

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
            stfts = get_multi_stft(audio_array, **self.__config.get_istft_kwargs())
            return MultiSpectrogram.from_stfts(self.__config, self.__processor, *stfts)
        
        else:
            raise Exception(f"Cannot handle data with {num_channels} channels")
        
    def get_spectrogram_from_path(self, file_path : str) -> MultiSpectrogram:
        # Load the audio
        audio, sample_rate = sf.read(file_path)

        # assert sample rate is correct
        assert sample_rate == self.__config.sample_rate, "Sample rate of the file is different from the one configured in the Config onbject"

        # return the spectro
        return self.get_spectrogram_from_audio(audio.transpose())
    
    def get_spectrograms(self, audio_or_file_list : list[Union[str, np.ndarray]]) -> list[MultiSpectrogram]:
        spectros : list[MultiSpectrogram] = []
        for audio_or_file in audio_or_file_list:
            if isinstance(audio_or_file, str):
                spectros.append(
                    self.get_spectrogram_from_path(audio_or_file))
            elif isinstance(audio_or_file, np.ndarray):
                spectros.append(
                    self.get_spectrogram_from_audio(audio_or_file))
            else:
                raise Exception(f"Couldn't handle type : {type(audio_or_file)}")
        return spectros
    
    def get_numpy_dataset(self, 
                          audio_or_file_list : list[Union[str, np.ndarray]], 
                          use_processor : bool,
                          fit_processor :  bool
                          ) -> np.ndarray:
        
        if self.__config.audio_length is None:
            raise Exception("Cannot create a numpy dataset with no audio length provided. \n" + \
                            "Set the audio_length field in the configuration.")
        if self.__audio_padder is None:
            raise Exception("Cannot create a numpy dataset with no audio padding function provided. \n" + \
                            "Set audio_padder argument in the factory constructor.")

        if use_processor and fit_processor:
            if isinstance(self.__processor, AbstractFitDataProcessor):
                self.__processor.fit() # TODO fit on datas
            else:
                raise Exception("Can only fit the processor on AbstractFitProcessor")
        spectros = self.get_spectrograms(audio_or_file_list)
        X_data = np.zeros((len(spectros), *spectros[0].shape))
        for i, spec in enumerate(spectros):
            X_data[i] = spec.to_data(use_processor)
        return X_data


    