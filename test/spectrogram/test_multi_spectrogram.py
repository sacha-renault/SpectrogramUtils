import pytest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from src.SpectrogramUtils import SpectrogramFactory, MultiSpectrogram, Config, RealImageStftProcessor, MagnitudePhaseStftProcessor, ListOrdering, DisplayType
from src.SpectrogramUtils.processors.wrapper import DataProcessorWrapper
from src.SpectrogramUtils.exceptions.lib_exceptions import WrongConfigurationException, WrongDisplayTypeException, NoIndexException, UnknownWavTypeException

def test_multi_spectrogram_real_imag():
    config = Config(1)
    stft_processor = RealImageStftProcessor()
    data_processor = DataProcessorWrapper(None)

    data = [np.random.rand(256,256) + 1j*np.random.rand(256,256) for _ in range(4)]
    spec = MultiSpectrogram.from_stfts(config, stft_processor, data_processor, ListOrdering.ALTERNATE, *data)
    assert np.all(np.abs(spec.get_stfts() - data) < 1e-15)
    assert all([np.all(np.abs(spec.get_stft(i) - x) < 1e-15) for i,x in enumerate(data)])
    assert spec.ordering == ListOrdering.ALTERNATE


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

def test_show_wave_on_axis():
    config = Config(1)
    stft_processor = RealImageStftProcessor()
    data_processor = DataProcessorWrapper(None)
    data = [np.random.rand(256,256) + 1j*np.random.rand(256,256) for _ in range(4)]
    spec = MultiSpectrogram.from_stfts(config, stft_processor, data_processor, ListOrdering.ALTERNATE, *data)
    ax = Mock()
    ax.plot = MagicMock()
    spec.show_wave_on_axis(ax, display_type=DisplayType.STACK)
    spec.show_wave_on_axis(ax, display_type=DisplayType.INDEX, index=0)
    spec.show_wave_on_axis(ax, display_type=DisplayType.MEAN)
    assert ax.plot.call_count == 3

def test_show_wave_fails():
    config = Config(1)
    stft_processor = RealImageStftProcessor()
    data_processor = DataProcessorWrapper(None)
    data = [np.random.rand(256,256) + 1j*np.random.rand(256,256) for _ in range(4)]
    spec = MultiSpectrogram.from_stfts(config, stft_processor, data_processor, ListOrdering.ALTERNATE, *data)
    ax = Mock()
    ax.plot = MagicMock()
    with pytest.raises(WrongDisplayTypeException):
        spec.show_wave_on_axis(ax, display_type=DisplayType.MIN)
        spec.show_wave_on_axis(ax, display_type=DisplayType.MAX)
    assert ax.plot.call_count == 0

def test_show_wave_fails_index():
    config = Config(1)
    stft_processor = RealImageStftProcessor()
    data_processor = DataProcessorWrapper(None)
    data = [np.random.rand(256,256) + 1j*np.random.rand(256,256) for _ in range(4)]
    spec = MultiSpectrogram.from_stfts(config, stft_processor, data_processor, ListOrdering.ALTERNATE, *data)
    ax = Mock()
    ax.plot = MagicMock()
    with pytest.raises(NoIndexException):
        spec.show_wave_on_axis(ax, display_type=DisplayType.INDEX)
    assert ax.plot.call_count == 0

def test_save_as_file():
    config = Config(1)
    stft_processor = RealImageStftProcessor()
    data_processor = DataProcessorWrapper(None)
    data = [np.random.rand(256,256) + 1j*np.random.rand(256,256) for _ in range(2)]
    spec = MultiSpectrogram.from_stfts(config, stft_processor, data_processor, ListOrdering.ALTERNATE, *data)
    array = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    spec.get_waves = MagicMock(return_value=array)
    spec.__conf = MagicMock()
    spec.__conf.sample_rate = 44100

    with patch('soundfile.write') as mock_sf_write:
        spec.save_as_file("test_audio")
        assert mock_sf_write.called
        args, kwargs = mock_sf_write.call_args
        assert args[0] == "test_audio.wav"
        assert np.array_equal(args[1], array.transpose())
        assert kwargs['samplerate'] == 44100

        config.num_channel = 3
        data = [np.random.rand(256,256) + 1j*np.random.rand(256,256) for _ in range(3)]
        n_spec = MultiSpectrogram.from_stfts(config, stft_processor, data_processor, ListOrdering.ALTERNATE, *data)
        with pytest.raises(UnknownWavTypeException, match="Cannot save audio if it isn't mono or stereo"):
            n_spec.save_as_file("test_audio")

def test_save_as_file_fails1():
        config = Config(1)
        stft_processor = RealImageStftProcessor()
        data_processor = DataProcessorWrapper(None)
        data = [np.random.rand(256,256) + 1j*np.random.rand(256,256) for _ in range(2)]
        spec = MultiSpectrogram.from_stfts(config, stft_processor, data_processor, ListOrdering.ALTERNATE, *data)
        array = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        spec.get_waves = MagicMock(return_value=array)
        spec.__conf = MagicMock()
        spec.__conf.sample_rate = 44100
        with patch('soundfile.write') as mock_sf_write:
            with pytest.raises(WrongConfigurationException):
                spec.save_as_file("test_audio", True, lambda x : x)
            assert not mock_sf_write.called

def test_save_as_file_success_ext():
        config = Config(1)
        stft_processor = RealImageStftProcessor()
        data_processor = DataProcessorWrapper(None)
        data = [np.random.rand(256,256) + 1j*np.random.rand(256,256) for _ in range(2)]
        spec = MultiSpectrogram.from_stfts(config, stft_processor, data_processor, ListOrdering.ALTERNATE, *data)
        array = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        spec.get_waves = MagicMock(return_value=array)
        spec.__conf = MagicMock()
        spec.__conf.sample_rate = 44100
        with patch('soundfile.write') as mock_sf_write:
            spec.save_as_file("test_audio", True)
        assert mock_sf_write.called
        args, kwargs = mock_sf_write.call_args
        assert np.array_equal(args[1], array.transpose() / np.max(np.abs(array)))
        assert kwargs['samplerate'] == 44100

def test_save_as_file_success_ext_2():
        config = Config(1)
        stft_processor = RealImageStftProcessor()
        data_processor = DataProcessorWrapper(None)
        data = [np.random.rand(256,256) + 1j*np.random.rand(256,256) for _ in range(2)]
        spec = MultiSpectrogram.from_stfts(config, stft_processor, data_processor, ListOrdering.ALTERNATE, *data)
        array = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        spec.get_waves = MagicMock(return_value=array)
        spec.__conf = MagicMock()
        spec.__conf.sample_rate = 44100
        with patch('soundfile.write') as mock_sf_write:
            spec.save_as_file("test_audio", normalization_func=lambda x : 2*x)
        assert mock_sf_write.called
        args, kwargs = mock_sf_write.call_args
        assert np.array_equal(args[1], array.transpose() * 2)
        assert kwargs['samplerate'] == 44100

def test_show_image():
    config = Config(1)
    stft_processor = RealImageStftProcessor()
    data_processor = DataProcessorWrapper(None)
    data = [np.random.rand(256,256) + 1j*np.random.rand(256,256) for _ in range(4)]
    spec = MultiSpectrogram.from_stfts(config, stft_processor, data_processor, ListOrdering.ALTERNATE, *data)
    ax = Mock()
    ax.imshow = MagicMock()
    spec.show_image_on_axis(ax, display_type=DisplayType.MIN)
    config.power_to_db_intensity = 2
    spec.show_image_on_axis(ax, display_type=DisplayType.MAX)
    spec.show_image_on_axis(ax, display_type=DisplayType.INDEX, index=0)
    spec.show_image_on_axis(ax, display_type=DisplayType.MEAN)
    assert ax.imshow.call_count == 4

def test_show_image_fails():
    config = Config(1)
    stft_processor = RealImageStftProcessor()
    data_processor = DataProcessorWrapper(None)
    data = [np.random.rand(256,256) + 1j*np.random.rand(256,256) for _ in range(4)]
    spec = MultiSpectrogram.from_stfts(config, stft_processor, data_processor, ListOrdering.ALTERNATE, *data)
    ax = Mock()
    ax.imshow = MagicMock()
    with pytest.raises(WrongDisplayTypeException):
        spec.show_image_on_axis(ax, display_type=DisplayType.STACK)
    with pytest.raises(NoIndexException):
        spec.show_image_on_axis(ax, display_type=DisplayType.INDEX)
    assert ax.imshow.call_count == 0
