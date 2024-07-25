import pytest

import numpy as np

from src.SpectrogramUtils import SpectrogramFactory, MultiSpectrogram, Config, RealImageStftProcessor, MagnitudePhaseStftProcessor, ListOrdering
from src.SpectrogramUtils.processors.wrapper import DataProcessorWrapper
from src.SpectrogramUtils.exceptions.lib_exceptions import WrongConfigurationException

def test_multi_spectrogram_real_imag():
    config = Config(1)
    stft_processor = RealImageStftProcessor()
    data_processor = DataProcessorWrapper(None)

    data = [np.random.rand(256,256) + 1j*np.random.rand(256,256) for _ in range(4)]
    spec = MultiSpectrogram.from_stfts(config, stft_processor, data_processor, ListOrdering.ALTERNATE, *data)
    assert np.all(np.abs(spec.get_stfts() - data) < 1e-15)
    assert all([np.all(np.abs(spec.get_stft(i) - x) < 1e-15) for i,x in enumerate(data)])


def test_multi_spectrogram_mag_phase():
    config = Config(1)
    stft_processor = MagnitudePhaseStftProcessor()
    data_processor = DataProcessorWrapper(None)

    data = [np.random.rand(256,256) + 1j*np.random.rand(256,256) for _ in range(4)]
    spec = MultiSpectrogram.from_stfts(config, stft_processor, data_processor, ListOrdering.ALTERNATE, *data)
    assert np.all(np.abs(spec.get_stfts() - data) < 1e-15)
    assert all([np.all(np.abs(spec.get_stft(i) - x) < 1e-15) for i,x in enumerate(data)])

def test_multi_spectrogram_shape():
    config = Config(1)
    stft_processor = RealImageStftProcessor()
    data_processor = DataProcessorWrapper(None)

    data = [np.random.rand(256,256) + 1j*np.random.rand(256,256) for _ in range(4)]
    spec = MultiSpectrogram.from_stfts(config, stft_processor, data_processor, ListOrdering.ALTERNATE, *data)
    assert spec.shape == (len(data) * 2, *data[0].shape)
