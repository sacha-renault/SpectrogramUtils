import pytest

import numpy as np
import tensorflow as tf

from src.SpectrogramUtils import Config, AudioPadding
from src.SpectrogramUtils.extensions.tf import SpectrogramTfFactory

def test_is_available():
    assert SpectrogramTfFactory.tf_available

def test_get_tf_dataset():
    factory = SpectrogramTfFactory(Config(2, audio_length=5000), audio_padder=AudioPadding.LPAD_LCUT)
    audios = [ np.random.rand(1,1000) for _ in range(10) ]
    dataset = factory.get_tf_dataset(audios, use_processor=False)
    assert tf.shape(dataset)[0] == 10
    assert isinstance(dataset, tf.Tensor)