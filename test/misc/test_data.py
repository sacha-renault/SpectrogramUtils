from unittest.mock import mock_open, patch

import pytest
import numpy as np

from src.SpectrogramUtils.data.config import Config
from src.SpectrogramUtils.data.librosa_stft_args import LibrosaSTFTArgs

def test_conf():
    conf = Config(2)
    assert conf.stft_config == LibrosaSTFTArgs()
    assert conf.sample_rate == 44100
    assert conf.power_to_db_intensity is None
    assert conf.audio_length is None

def test_conf_failed():
    with pytest.raises(AssertionError):
        Config(2, stft_config=2)
    with pytest.raises(AssertionError):
        Config(2, sample_rate=-2)
    with pytest.raises(AssertionError):
        Config(2, power_to_db_intensity=-2)
    with pytest.raises(AssertionError):
        Config(2, audio_length=-2)