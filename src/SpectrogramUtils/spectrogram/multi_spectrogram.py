from typing import Union

import numpy as np
import numpy.typing as npt
import librosa
import soundfile as sf
from matplotlib.axes import Axes

from ..data.data import DisplayType, ListOrdering
from ..data.config import Config
from ..processors.abstract_data_processor import AbstractDataProcessor
from ..processors.wrapper import DataProcessorWrapper
from ..exceptions.lib_exceptions import NoProcessorException, NoIndexException, WrongDisplayTypeException, UnknownWavTypeException
from ..stft_complexe_processor.abstract_stft_processor import AbstractStftComplexProcessor

class MultiSpectrogram:
    @classmethod
    def from_stfts(cls, config : Config, stft_processor : AbstractStftComplexProcessor, processor : AbstractDataProcessor = None, ordering : ListOrdering = None, *stfts : np.ndarray):
        # assert same shape
        assert all(stft.shape == stfts[0].shape for stft in stfts), "All stfts shape should be the same"

        # create a data
        data = np.zeros(shape = stft_processor.shape(stfts))
        for i, stft in enumerate(stfts):
            stft_processor.complexe_to_real(stft, data, i)
        
        # Instanciate the class with the multi spectro 
        return cls(config, stft_processor, processor, ordering, data)

    def __init__(self, config : Config, stft_processor : AbstractStftComplexProcessor, processor : DataProcessorWrapper,  ordering : ListOrdering, data : np.ndarray) -> None:
        # define config
        assert isinstance(config, Config), f"config must be a Config object, not {type(config)}"
        self.__conf = config

        # define config
        assert isinstance(stft_processor, AbstractStftComplexProcessor), f"stft_processor must be a AbstractStftComplexProcessor object, not {type(stft_processor)}"
        self.__stft_processor = stft_processor

        # Set processor
        assert isinstance(processor, DataProcessorWrapper) or processor is None
        self.__processor = processor 

        # define data
        assert isinstance(data, np.ndarray), f"data must be a NDArray object, not {type(data)}"
        assert data.dtype == np.float64 or data.dtype == np.float32, f"data type must be float64 or float32, not {data.dtype}"
        self.__data = data

        # Define ordering
        assert isinstance(ordering, ListOrdering), f"ordering must be a ListOrdering object, not {type(data)}"
        self.__ordering = ordering

    
    def to_data(self, process_data : bool = False) -> npt.NDArray[np.float64]:
        """return a stereo spectrogram. Indexes precisions :
        - 2*n : amplitude of stft at index n
        - 2*n + 1 : phase of stft at index n

        Args:
            process (bool): if data need to be processed

        Returns:
            np.ndarray: multi spectrogram
        """
        if process_data:
            if self.__processor is not None:
                data = self.__processor.forward(self.__data)
            else:
                raise NoProcessorException("Askep processed data without providing a processor")
        else:
            data = self.__data
        return data
    
    def to_rearanged_data(self, process_data : bool = False):
        """return a stereo spectrogram with rearanged indexes. Indexes precisions :
        - 2*n : amplitude of stft at index n
        - 2*n + 1 : phase of stft at index n

        Args:
            process (bool): if data need to be processed

        Returns:
            np.ndarray: multi spectrogram
        """
        data = self.to_data(process_data)
        if self.__ordering == ListOrdering.AMPLITUDE_PHASE:
            data = np.concatenate((data[::2], data[1::2]), axis = 0) # Rearange the order of the amplitudes and phases
        return data
        
    
    def get_amplitudes(self) -> np.ndarray:
        # TODO
        raise NotImplementedError()
        
    @property
    def ordering(self) -> ListOrdering:
        """Return the list ordering set for this spectrogram
        """
        return self.__ordering
    
    
    @property
    def num_stfts(self) -> int:
        """Number of stfts in the spectrogram, equivalent to the number of channels in the audio.
        """
        return self.__stft_processor.num_stfts(self.__data)
    
    @property
    def shape(self):
        """return the shape of the spectrogram
        """
        return self.__data.shape
    
    def show_image_on_axis(self, 
                           axis : Axes, 
                           display_type : DisplayType = DisplayType.MEAN, 
                           index : Union[int, None] = None, 
                           use_processor : bool = False,
                           *axes_args, 
                           **axes_kwargs) -> None:
        """Show the stft on the given axis (If the processor make the mean of the stft not 0, it RECENTER on 0 !)

        Args:
            axis (axes.Axes): Axis to display the stereo spectrogram
            display_type (DisplayType, optional): How to display the stereo spectrogram. Defaults to DisplayType.MEAN.
            
        Raises:
            NoIndexException: DisplayType is INDEX but no index provided
            WrongDisplayTypeException: this DisplayType isn't handled for this method
        """
        # init new data
        display_data = np.zeros_like(self.to_data(use_processor)[0], dtype=np.complex128)

        if display_type == DisplayType.MEAN:
            data = self.get_stfts(use_processor)
            display_data += np.mean(data, axis = 0)
        
        elif display_type == DisplayType.INDEX:
            if index is not None:
                display_data += self.get_stft(index)
            else:
                raise NoIndexException("Can't display index if no index is provided")
            
        elif display_type == DisplayType.MAX:
            data = self.get_stfts(use_processor)
            display_data += np.max(data, axis = 0)

        elif display_type == DisplayType.MIN:
            data = self.get_stfts(use_processor)
            display_data += np.min(data, axis = 0)
        
        else:
            raise WrongDisplayTypeException(f"Cannot use display type {display_type.name} for image display")

        if self.__conf.power_to_db_intensity is not None:
            db_data = librosa.power_to_db(np.abs(display_data)**self.__conf.power_to_db_intensity)
            axis.imshow(db_data, *axes_args, **axes_kwargs)
            axis.set_xlabel('Time')
            axis.set_ylabel('Frequency')
        else:
            axis.imshow(np.abs(display_data), *axes_args, **axes_kwargs)
            axis.set_xlabel('Time')
            axis.set_ylabel('Frequency')

    def show_wave_on_axis(self, 
                          axis : Axes, 
                          display_type : DisplayType = DisplayType.STACK, 
                          index : Union[int, None] = None, 
                          *axes_args, 
                          **axes_kwargs) -> None:
        """Show the wave shape on a given axis 

        Args:
            axis (axes.Axes): Axis to display the stereo spectrogram
            display_type (DisplayType, optional): How to display the wave. Defaults to DisplayType.STACK.

        Raises:
            NoIndexException: DisplayType is INDEX but no index provided
            WrongDisplayTypeException: this DisplayType isn't handled for this method
        """
        axis.set_xlabel('Time')
        axis.set_ylabel('Amplitude')
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
                raise NoIndexException("Can't display index if no index is provided")

        else:
            raise WrongDisplayTypeException(f"Cannot use display type {display_type.name} for image display")
    
    def get_stft(self, index : int, use_processor : bool = False) -> npt.NDArray[np.complex128]:
        """Get a stft at a specified index

        Args:
            index (int): index of channel to get the stft

        Returns:
            np.ndarray: stft at the requested index
        """
        data = self.to_data(use_processor)
        return self.__stft_processor.real_to_complexe(data, index)
    
    def get_stfts(self, use_processor : bool = False) -> npt.NDArray[np.complex128]:
        stfts = np.zeros((self.num_stfts, *self.__data.shape[1:]), dtype=np.complex128)
        for i in range(self.num_stfts):
            stfts[i] = self.get_stft(i, use_processor)
        return stfts

    def get_wave(self, index : int) -> npt.NDArray[np.float64]:
        """Get the wave shape for the channel at the requested index

        Args:
            index (int): requested channel index

        Returns:
            np.ndarray: 1D wave shape
        """
        stft = self.get_stft(index, False)
        wave = librosa.istft(stft, **self.__conf.get_istft_kwargs())
        return wave
    
    def get_waves(self) -> npt.NDArray[np.float64]:
        """Get the wave shape for all channels

        Returns:
            np.ndarray: n-Dimentional wave shape
        """
        waves = None
        for i in range(self.num_stfts): 
            wave = self.get_wave(i)
            if waves is None:
                waves = np.zeros(shape=(self.num_stfts, *wave.shape))
            waves[i] = wave
        return waves
    
    def save_as_file(self, file_name : str) -> None:
        """save the file as wav

        Args:
            file_name (str): file name

        Raises:
            UnknownWavTypeException: Cannot save as wav file that has more than 2 channels. please call method 
            get_waves and setup a custom save function.
        """
        if not file_name.endswith(".wav"):
            file_name += ".wav"
        if self.num_stfts <=2:
            sf.write(file_name, self.get_waves().transpose(), samplerate=self.__conf.sample_rate)
        else:
            raise UnknownWavTypeException("Cannot save audio if it isn't mono or stereo")
