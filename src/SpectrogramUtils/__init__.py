from .data.data import DisplayType, AudioPadding, ListOrdering
from .data.config import Config
from .data.librosa_stft_args import LibrosaSTFTArgs
from .processors.abstract_data_processor import AbstractDataProcessor, AbstractFitDataProcessor, AbstractDestructiveDataProcessor
from .processors.scaler_audio_processor import ScalerAudioProcessor
from .processors.simple_scaling_processor import SimpleScalingProcessor
from .spectrogram.multi_spectrogram import MultiSpectrogram
from .spectrogram.spectrogram_factory import SpectrogramFactory
from .stft_complexe_processor.abstract_stft_processor import AbstractStftComplexProcessor
from .stft_complexe_processor.mag_phase_stft_processor import MagnitudePhaseStftProcessor
from .stft_complexe_processor.real_imag_stft_processor import RealImageStftProcessor

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

"""
SpectrogramUtils

SpectrogramUtils is a library designed for handling and processing audio spectrograms. It provides tools to open audio files as spectrograms, preprocess data, and easily integrate with deep learning models, especially for generative AI tasks.

Features:
- Open audio files as spectrograms using a factory pattern.
- Preprocess data efficiently.
- Provide a simple flow from complex arrays in STFT to float64 arrays.
- Retrieve data from a deep learning model directly as audio.
- Primarily designed for generative AI tasks, not for creating datasets for classifiers.

Usage:
    # Example usage
    import os
    from SpectrogramsUtils import SpectrogramFactory, Config, AudioPadding

    # Create a config
    config = Config(2, n_fft=512, audio_length=44_100*5) # Audio of 5 seconds

    # Factory with no audio processor
    factory = SpectrogramFactory(config, audio_padder=AudioPadding.RPAD_RCUT)

    # Load a single audio file
    spectrogram = factory.get_spectrogram_from_path("path/to/file.wav")

    # Load one spectrogram for each audio file in a folder
    audio_directory = "path/to/directory"
    files = [os.path.join(audio_directory, audio_file) for audio_file in os.listdir(audio_directory)]
    spectrograms = factory.get_spectrograms(files)

See more:
    https://github.com/sacha-renault/SpectrogramUtils
"""
