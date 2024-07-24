from typing import Callable, Union, Optional, List, Any
from collections.abc import Iterable 
import pickle
import json
import warnings
import os

import numpy as np
import numpy.typing as npt
import soundfile as sf

from ..data.config import Config
from ..processors.abstract_data_processor import AbstractDataProcessor
from ..processors.wrapper import DataProcessorWrapper
from .multi_spectrogram import MultiSpectrogram
from ..data.data import AudioPadding, ListOrdering
from ..misc.utils import get_multi_stft
from ..exceptions.lib_exceptions import WrongConfigurationException, BadTypeException
from ..stft_complexe_processor.abstract_stft_processor import AbstractStftComplexProcessor
from ..stft_complexe_processor.real_imag_stft_processor import RealImageStftProcessor
from .._version import version as __version__


class SpectrogramFactory:
    def __init__(self, 
                config : Config,
                stft_processor : Optional[AbstractStftComplexProcessor] = None,
                data_processor : Optional[AbstractDataProcessor] = None,
                audio_padder : Optional[Callable[[np.ndarray, int], np.ndarray]] = AudioPadding.NONE,
                ordering : ListOrdering = ListOrdering.ALTERNATE
                ) -> None:
        # Set the config
        assert isinstance(config, Config), f"config must be Config, not {type(config)}"
        self.__config = config

        # Set the data processor
        assert stft_processor is None or isinstance(stft_processor, AbstractStftComplexProcessor), f"stft_processor must be None or AbstractStftComplexProcessor, not {type(stft_processor)}"
        self.__stft_processor = stft_processor if stft_processor is not None else RealImageStftProcessor()

        # Set the data processor
        assert data_processor is None or isinstance(data_processor, (AbstractDataProcessor, DataProcessorWrapper)), f"data_processor must be None or AbstractDataProcessor, not {type(data_processor)}"
        self.__processor = data_processor if isinstance(data_processor, DataProcessorWrapper) else DataProcessorWrapper(data_processor)

        # Set the padding functino
        assert audio_padder is None or isinstance(audio_padder, Callable), f"audio_padder must be None or Callable, not {type(audio_padder)}"
        self.__audio_padder = audio_padder

        # set ordering
        assert isinstance(ordering, ListOrdering), f"ordering must be type ListOrdering, not {type(ordering)}"
        self.__ordering = ordering

        # Check if config is correct
        if (self.__audio_padder is not None and self.__config.audio_length is None) or \
            (self.__audio_padder is None and self.__config.audio_length is not None):
            raise WrongConfigurationException("audio_padder and audio_length (in config object) must be either both set or both None")
        
    def save(self, save_dir : str):
        # Check if dir exist
        if not os.path.isdir(os.path.dirname(save_dir)) and not os.path.dirname(save_dir) == "":
            raise NotADirectoryError(f"The directory doesn't exist : {os.path.dirname(save_dir)}")
        
        # check if dir already exist
        elif os.path.isdir(save_dir):
            raise FileExistsError("Directory already exist, delete it or choose an other name")
        
        # else save
        else:
            os.mkdir(save_dir)
            spath = lambda x : os.path.normpath(os.path.join(save_dir, x +".pkl"))
            with open(spath("data_processor"), "wb") as file:
                pickle.dump(self.__processor, file)
            with open(spath("stft_processor"), "wb") as file:
                pickle.dump(self.__stft_processor, file)
            with open(spath("ordering"), "wb") as file:
                pickle.dump(self.__ordering, file)
            with open(spath("config"), "wb") as file:
                pickle.dump(self.__config, file)
            with open(spath("audio_padder"), "wb") as file:
                pickle.dump(self.__audio_padder, file)
            with open(os.path.join(save_dir, "config.json"), "w") as file:
                json.dump({"version" : __version__}, file)

    @classmethod
    def from_file(cls, load_dir : str):
        if not os.path.isdir(load_dir):
            raise NotADirectoryError(f"The directory doesn't exist: {load_dir}")
        
        spath = lambda x: os.path.normpath(os.path.join(load_dir, x + ".pkl"))
        
        with open(os.path.join(load_dir, "config.json")) as file:
            json_config : dict = json.load(file)
            if json_config.get("version", "0.0.0") != __version__:
                warnings.warn(f"Found factory saved on version {json_config.get('version', '0.0.0')}. Current version is {__version__}. Factory might be broken, either install correct version or use at your own risk")
        with open(spath("data_processor"), "rb") as file:
            data_processor = pickle.load(file)
        with open(spath("stft_processor"), "rb") as file:
            stft_processor = pickle.load(file)
        with open(spath("ordering"), "rb") as file:
            ordering = pickle.load(file)
        with open(spath("config"), "rb") as file:
            config = pickle.load(file)
        with open(spath("audio_padder"), "rb") as file:
            audio_padder = pickle.load(file)

        return cls(config, stft_processor, data_processor, audio_padder, ordering)
        
    @property
    def num_channel(self) -> int:
        return self.__config.num_channel

    def get_spectrogram_from_audio(self, audio_array : np.ndarray) -> MultiSpectrogram:
        """
        Converts an audio array to a MultiSpectrogram instance.

        Args:
            audio_array (np.ndarray): 
                A numpy array representing the audio data. The shape of the array should be (num_channels, samples).

        Raises:
            WrongConfigurationException: 
                If the number of channels in the audio array does not match the expected number of channels in the configuration.
            WrongConfigurationException: 
                If the audio length or audio padding function is not provided in the configuration.

        Returns:
            MultiSpectrogram: 
                A MultiSpectrogram instance created from the given audio array.
        """
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
            return MultiSpectrogram.from_stfts(self.__config, self.__stft_processor, self.__processor, self.__ordering, *stfts)
        
        else:
            raise WrongConfigurationException(f"Cannot handle data with {num_channels} channels. Configuration is set for {self.__config.num_channel} channels.")
        
    def get_spectrogram_from_path(self, file_path : str) -> MultiSpectrogram:
        """
        Converts an audio file from the given file path to a MultiSpectrogram instance.

        Args:
            file_path (str): 
                The path to the audio file.

        Returns:
            MultiSpectrogram: 
                A MultiSpectrogram instance created from the audio file.

        Raises:
            AssertionError: 
                If the file does not exist or if the sample rate of the file does not match the configured sample rate.
        """
        # assert file exist
        assert os.path.isfile(file_path), f"The file {file_path} doesn't exist"

        # Load the audio
        audio, sample_rate = sf.read(file_path, always_2d=True)

        # assert sample rate is correct
        assert sample_rate == self.__config.sample_rate, "Sample rate of the file is different from the one configured in the Config onbject"

        # return the spectro
        return self.get_spectrogram_from_audio(audio.transpose())
    
    def get_spectrograms_from_files(self, audio_or_file_list : Iterable[Union[str, np.ndarray]]) -> List[MultiSpectrogram]:
        """
        Converts a list of audio file paths or numpy arrays to a list of MultiSpectrogram instances.

        Args:
            audio_or_file_list (Iterable[Union[str, np.ndarray]]): 
                A list containing either file paths to audio files (as strings) or numpy arrays representing audio data.

        Raises:
            BadTypeException: 
                If an element in the list is neither a string nor a numpy array.

        Returns:
            List[MultiSpectrogram]: 
                A list of MultiSpectrogram instances created from the audio files or arrays.
        """
        # Type assertion
        assert isinstance(audio_or_file_list, Iterable), f"audio_or_file_list must be an iterable, not {type(audio_or_file_list)}"

        # Init the return list
        spectros : list[MultiSpectrogram] = []

        # Iterate over the list to get spectrograms
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
    
    def get_spectrogram_from_model_output(self, model_output : npt.NDArray[np.float64], use_processor : bool = True) -> List[MultiSpectrogram]:
        """From model output, recreate a spectrogram, rearrange the amplitude phase if needed.
        /!\\ the model output should be shaped like (batch, channel, h, w)

        Args:
            model_output (npt.NDArray[np.float64]): [description]

        Returns:
            list[MultiSpectrogram]: Spectrograms from the model output
        """
        # Check if need to be reordered
        if self.__ordering == ListOrdering.AMPLITUDE_PHASE:
            new_order = np.arange(model_output.shape[1]).reshape(2, -1).flatten("F")
            model_output = model_output[:, new_order, :, :] # Rearange the data in correct order

        # Make spectrograms once it's on the correct order
        spectros : list[MultiSpectrogram] = []
        for x in model_output:
            # Backward process
            if use_processor:
                x = self.__processor.backward(x)

            # Get the spectrogram
            spectros.append(
                MultiSpectrogram(self.__config, self.__stft_processor, self.__processor, self.__ordering, x))
            
        return spectros
    
    def get_numpy_dataset(self, 
                          audio_or_file_list : Union[List[Union[str, np.ndarray]], List[MultiSpectrogram]], 
                          use_processor : bool
                          ) -> npt.NDArray[np.float64]:
        """
        Converts a list of audio files or MultiSpectrogram instances to a numpy dataset.

        Args:
            audio_or_file_list (Union[List[Union[str, np.ndarray]], List[MultiSpectrogram]]): 
                A list containing either file paths to audio files, numpy arrays representing audio data,
                or a list of MultiSpectrogram instances.
            use_processor (bool): 
                A boolean flag indicating whether to process the data before converting it to a numpy dataset.

        Raises:
            WrongConfigurationException: 
                If the audio length is not provided in the configuration.
            WrongConfigurationException: 
                If the audio padding function is not provided in the configuration.

        Returns:
            npt.NDArray[np.float64]: 
                A numpy array containing the processed audio data.
        """
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
    
    def _get_stft_shape(self):
        """#### Return the shape that a stft would have 

        #### Returns:
            - tuple[int, int]: stft shape
            >>> 
        """
        stft_config = self.__config.get_istft_kwargs()
        audio_length = stft_config.get("audio_length")
        window_length = stft_config.get("window_length")
        hop_length = stft_config.get("hop_length")
        n_fft = stft_config.get("n_fft")
        num_frames = (audio_length - window_length) // hop_length + 1
        num_frequency_bins = n_fft // 2 + 1
        return num_frequency_bins, num_frames


    