import pytest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

from src.SpectrogramUtils import MultiSpectrogram, Config, RealImageStftProcessor, MagnitudePhaseStftProcessor, ListOrdering, DisplayType, AudioPadding
from src.SpectrogramUtils.extensions.torch import SpectrogramTorchFactory
from src.SpectrogramUtils.processors.wrapper import DataProcessorWrapper
from src.SpectrogramUtils.exceptions.lib_exceptions import WrongConfigurationException, WrongDisplayTypeException, NoIndexException, UnknownWavTypeException

def test_torch_available():
    assert SpectrogramTorchFactory.torch_available

def test_get_torch_dataset_batch_generator_from_preloaded():
    factory = SpectrogramTorchFactory(Config(2))
    tensor = torch.randn(size = (45, 10, 10))
    generator = factory.get_torch_dataset_batch_generator(tensor, 10, "cpu")
    for _ in range(4):
        data = next(generator)
        assert isinstance(data, torch.Tensor)
        assert data.shape == (10, 10, 10)
    data = next(generator)
    assert isinstance(data, torch.Tensor)
    assert data.shape == (5, 10, 10)
    


def test_get_torch_dataset_batch_generator_from_disk():
    factory = SpectrogramTorchFactory(Config(2, audio_length=5000), audio_padder=AudioPadding.LPAD_LCUT)
    tensor = torch.randn(10)

    with patch('os.path.isfile', return_value=True):
        with patch("soundfile.read", return_value = (np.random.rand(1000,2), 44100)) as sf_mock:
            generator = factory.get_torch_dataset_batch_generator(["file.wav"]*45, 10, tensor, use_processor=False)
            excpted_call_count = 0
            for _ in range(4):
                excpted_call_count+=10
                data = next(generator)
                assert isinstance(data, torch.Tensor)
                assert data.shape[0] == 10
                assert sf_mock.call_count == excpted_call_count
            data = next(generator)
            assert isinstance(data, torch.Tensor)
            assert sf_mock.call_count == excpted_call_count + 5
            assert data.shape[0] == 5


def test_get_torch_dataset():
    factory = SpectrogramTorchFactory(Config(2, audio_length=5000), audio_padder=AudioPadding.LPAD_LCUT)
    audios = [ np.random.rand(1,1000) for _ in range(10) ]
    dataset = factory.get_torch_dataset(audios, use_processor=False, device_or_obj = "cpu")
    assert dataset.shape[0] == 10
    dataset = factory.get_torch_dataset(audios, use_processor=False, device_or_obj = None)
    assert dataset.shape[0] == 10
