from typing import Callable, Union, Optional, List, TYPE_CHECKING, Any
import warnings
import os

import numpy as np
import numpy.typing as npt
import soundfile as sf

from ..data.config import Config
from ..processors.abstract_data_processor import AbstractDataProcessor
from .multi_spectrogram import MultiSpectrogram
from ..data.data import AudioPadding, ListOrdering
from ..misc.utils import get_multi_stft
from ..exceptions.lib_exceptions import WrongConfigurationException, BadTypeException

# Conditional imports for type hints
if TYPE_CHECKING:
    import torch
    import tensorflow as tf

class SpectrogramFactory:
    def __init__(self, 
                config : Config,
                data_processor : Optional[AbstractDataProcessor] = None,
                audio_padder : Optional[Callable[[np.ndarray, int], np.ndarray]] = AudioPadding.NONE,
                ordering : ListOrdering = ListOrdering.ALTERNATE
                ) -> None:
        # Set the config
        self.__config = config

        # Set the data processor
        self.__processor = data_processor

        # Set the padding functino
        self.__audio_padder = audio_padder

        # set ordering
        self.__ordering = ordering

        # Check if config is correct
        if self.__audio_padder is not None and self.__config.audio_length is None:
            raise WrongConfigurationException("Audio Padder can't be userd without a audio_length configured in the config object.")
        
        self.__torch_available = None
        self.__tf_available = None

    def _check_tf(self):
        if self.__tf_available is None:
            try:
                import tensorflow as tf
                self.__tf_available = True
                self.__tf = tf
            except ImportError:
                self.__tf_available = False
                self.__tf = None
        return self.__tf_available
    
    def _check_torch(self):
        if self.__torch_available is None:
            try:
                import torch
                self.__torch_available = True
                self.__torch = torch
            except ImportError:
                self.__torch_available = False
                self.__torch = None
        return self.__torch_available

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
            return MultiSpectrogram.from_stfts(self.__config, self.__processor, self.__ordering, *stfts)
        
        else:
            raise WrongConfigurationException(f"Cannot handle data with {num_channels} channels. Configuration is set for {self.__config.num_channel} channels.")
        
    def get_spectrogram_from_path(self, file_path : str) -> MultiSpectrogram:
        # assert file exist
        assert os.path.isfile(file_path), f"The file {file_path} doesn't exist"

        # Load the audio
        audio, sample_rate = sf.read(file_path, always_2d=True)

        # assert sample rate is correct
        assert sample_rate == self.__config.sample_rate, "Sample rate of the file is different from the one configured in the Config onbject"

        # return the spectro
        return self.get_spectrogram_from_audio(audio.transpose())
    
    def get_spectrograms_from_files(self, audio_or_file_list : List[Union[str, np.ndarray]]) -> List[MultiSpectrogram]:
        spectros : list[MultiSpectrogram] = []
        for audio_or_file in audio_or_file_list:
            if isinstance(audio_or_file, str):
                spectros.append(
                    self.get_spectrogram_from_path(audio_or_file))
            elif isinstance(audio_or_file, np.ndarray):
                spectros.append(
                    self.get_spectrogram_from_audio(audio_or_file))
            else:
                raise BadTypeException(f"Couldn't handle type : {type(audio_or_file)}. Can only get file_path as str and audio_file as NDArray")
        return spectros
    
    def get_spectrogram_from_model_output(self, model_output : npt.NDArray[np.float64]) -> List[MultiSpectrogram]:
        """#### From model output, recreate a spectrogram, rearrange the amplitude phase if needed.
        /!\\ the model output should be shaped like (batch, channel, h, w)

        #### Args:
            - model_output (npt.NDArray[np.float64]): [description]

        #### Returns:
            - list[MultiSpectrogram]: Spectrograms from the model output
        """
        # Check if need to be reordered
        if self.__ordering == ListOrdering.AMPLITUDE_PHASE:
            new_order = np.arange(model_output.shape[1]).reshape(2, -1).flatten("F")
            model_output = model_output[:, new_order, :, :] # Rearange the data in correct order

        # Make spectrograms once it's on the correct order
        spectros : list[MultiSpectrogram] = []
        for x in model_output:
            # Backward process
            backward_processed = self.__processor.backward(x)

            # Get the spectrogram
            spectros.append(
                MultiSpectrogram(self.__config, self.__processor, self.__ordering, backward_processed))
            
        return spectros
    
    def get_numpy_dataset(self, 
                          audio_or_file_list : Union[List[Union[str, np.ndarray]], List[MultiSpectrogram]], 
                          use_processor : bool
                          ) -> npt.NDArray[np.float64]:
        
        if self.__config.audio_length is None:
            raise WrongConfigurationException("Cannot create a numpy dataset with no audio length provided. \n" + \
                            "Set the audio_length field in the configuration.")
        if self.__audio_padder is None:
            raise WrongConfigurationException("Cannot create a numpy dataset with no audio padding function provided. \n" + \
                            "Set audio_padder argument in the factory constructor.")

        spectros = self.get_spectrograms_from_files(audio_or_file_list)
        X_data = np.zeros((len(spectros), *spectros[0].shape))
        for i, spec in enumerate(spectros):
            X_data[i] = spec.to_rearanged_data(use_processor)
        return X_data
    
    
    def to_torch_dataset(self, 
                          audio_or_file_list : Union[List[Union[str, np.ndarray]], List[MultiSpectrogram]], 
                          use_processor : bool,
                          device_or_obj : Optional[Union[torch.device, Any]] = None
                          ) -> torch.Tensor:
        if not self._check_torch():
            raise ImportError("torch is not available")
        
        data = self.get_numpy_dataset(audio_or_file_list, use_processor)
        tensor = self.__torch.Tensor(data)

        if device_or_obj is not None:
            if isinstance(device_or_obj, torch.device):
                return tensor.to(device_or_obj)
            else:
                _device = getattr(device_or_obj, "device", None)
                if _device is not None:
                    return tensor.to(_device)
                else:
                    warnings.warn(f"The provided obj for device_or_obj ({type(device_or_obj)}) does not have a 'device' attribute. Returning the tensor without moving it to a device.")
                    return tensor
        else:
            return tensor
        
    def to_tf_dataset(self, 
                    audio_or_file_list : Union[List[Union[str, np.ndarray]], List[MultiSpectrogram]], 
                    use_processor : bool,
                    ) -> tf.Tensor:
        if not self._check_tf():
            raise ImportError("tensorflow is not available")
        
        data = self.get_numpy_dataset(audio_or_file_list, use_processor)
        return self.__tf.Tensor(data)


    