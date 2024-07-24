import pytest

from src.SpectrogramUtils import SpectrogramFactory, Config, ScalerAudioProcessor, AudioPadding
from src.SpectrogramUtils.exceptions.lib_exceptions import WrongConfigurationException

def test_factory_constructor_success():
    config = Config(4)
    processor = ScalerAudioProcessor()
    factory = SpectrogramFactory(config, None, processor)
    assert factory.num_channel == 4

def test_factory_constructor_success_2():
    config = Config(4)
    factory = SpectrogramFactory(config, None)
    assert factory.num_channel == 4

def test_factory_constructor_success_3():
    config = Config(4, audio_length=1000)
    factory = SpectrogramFactory(config, None, None, AudioPadding.LPAD_LCUT)
    assert factory.num_channel == 4

def test_factory_constructor_fails():
    config = Config(4)
    with pytest.raises(AssertionError):
        SpectrogramFactory(config, 3)

def test_factory_constructor_fails_2():
    config = Config(4)
    with pytest.raises(AssertionError):
        SpectrogramFactory(config, None, 3)

def test_factory_constructor_fails_3():
    config = Config(4)
    with pytest.raises(WrongConfigurationException):
        SpectrogramFactory(config, None, None, lambda x, y : x)

def test_factory_constructor_fails_4():
    config = Config(4)
    with pytest.raises(AssertionError):
        SpectrogramFactory(config, None, None, 2)