from unittest.mock import mock_open, patch
import pickle

import pytest
import numpy as np

from src.SpectrogramUtils import SimpleScalingProcessor
from src.SpectrogramUtils.processors.wrapper import DataProcessorWrapper

def test_simple_scaling_processor():
    data = np.random.randint(0, 3, (4,256,256)) / 2
    data[::2] *= 255
    data[1::2] -= 0.5
    data[1::2] *= np.pi*2
    processor = DataProcessorWrapper(SimpleScalingProcessor(255, 2*np.pi, np.pi))
    assert np.all(data[::2] <= 255) and np.all(data[::2] >= 0)
    assert np.all(data[1::2] <= np.pi) and np.all(data[1::2] >= -np.pi)
    p_data =processor.forward(data)
    assert np.all(p_data[::2] <= 1) and np.all(p_data[::2] >= 0)
    assert np.all(p_data[1::2] <= 1) and np.all(p_data[1::2] >= 0)
    r_data =processor.backward(p_data)
    assert np.all(r_data - data < 1e-21)
