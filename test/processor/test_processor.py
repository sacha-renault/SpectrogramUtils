from unittest.mock import mock_open, patch
import pickle

from numpy._typing import NDArray
import pytest
import numpy as np

from src.SpectrogramUtils.processors.scaler_audio_processor import ScalerAudioProcessor
from src.SpectrogramUtils import AbstractDataProcessor
from src.SpectrogramUtils.exceptions.lib_exceptions import BrokenProcessorException

def test_scaler_data_processor_fit():
    processor = ScalerAudioProcessor()
    data = np.random.rand(100,100)
    processor.fit(data)
    result = processor.backward(processor.forward(data))
    assert processor.is_fitted
    assert np.mean(np.abs(data - result)) < 1e-15

def test_scaler_data_processor_save():
    processor = ScalerAudioProcessor()
    data = np.random.rand(100,100)
    processor.fit(data)

    mock_file = mock_open()
    with patch("builtins.open", mock_file), patch("pickle.dump") as mock_pickle_dump:
        processor.save("file.pkl")
        mock_file.assert_called_with("file.pkl", 'wb')
        expected_keys = {'ssc', 'mms'}
        args, _ = mock_pickle_dump.call_args
        saved_data = args[0]
        assert set(saved_data.keys()) == expected_keys


def test_scaler_data_processor_loadd():
    processor = ScalerAudioProcessor()
    data = np.random.rand(100,100)
    processor.fit(data)

    mock_file = mock_open()
    with patch("builtins.open", mock_file), patch("pickle.load") as mock_pickle_load:
        processor.load("file.pkl")
        mock_file.assert_called_with("file.pkl", 'rb')


def test_broken_processor():
    class BrokenProcessor(AbstractDataProcessor):
        def forward(self, data):
            return data * 2
        def backward(self, data: NDArray):
            return data * 2
    
    processor = BrokenProcessor()
    with pytest.raises(BrokenProcessorException):
        processor.check_reversible()
