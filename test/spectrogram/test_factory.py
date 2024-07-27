import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
import json
import pickle

import numpy as np
import librosa 

from src.SpectrogramUtils import SpectrogramFactory, Config, ScalerAudioProcessor, AudioPadding, \
    SimpleScalingProcessor, LibrosaSTFTArgs, ListOrdering, MultiSpectrogram, RealImageStftProcessor
from src.SpectrogramUtils.exceptions.lib_exceptions import WrongConfigurationException,BadTypeException
from src.SpectrogramUtils._version import version as __version__

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
        SpectrogramFactory(config, None, None, lambda x, _ : x)

def test_factory_constructor_fails_4():
    config = Config(4)
    with pytest.raises(AssertionError):
        SpectrogramFactory(config, None, None, 2)

def test_factory_get_numpy_dataset_fails():  
    config = Config(4)
    factory = SpectrogramFactory(config)
    with pytest.raises(WrongConfigurationException):
        factory.get_numpy_dataset([], False)

def test_get_stft_shape_no_center():
    stft_config = LibrosaSTFTArgs(center=False)
    config = Config(1, audio_length=5000, stft_config=stft_config)
    factory = SpectrogramFactory(config, data_processor=None, audio_padder=lambda x, _ : x)
    stft = librosa.stft(np.random.rand(5000), **stft_config)
    assert stft.shape == factory._get_stft_shape()

def test_get_stft_shape_center():
    stft_config=LibrosaSTFTArgs(center=True)
    config = Config(1, audio_length=5000, stft_config=stft_config)
    factory = SpectrogramFactory(config, data_processor=None, audio_padder=lambda x, _ : x)
    stft = librosa.stft(np.random.rand(5000), **stft_config)
    assert stft.shape == factory._get_stft_shape()

def test_factory_get_numpy_dataset_success():  
    config = Config(1, audio_length=10000, stft_config=LibrosaSTFTArgs())
    processor = SimpleScalingProcessor(1, 1, 0)
    factory = SpectrogramFactory(config, data_processor=processor, audio_padder=lambda x, _ : x)
    audio_arrays = [np.random.rand(1, 10000) for _ in range(5)]
    audio_stfts = librosa.stft(np.array(audio_arrays), **LibrosaSTFTArgs())
    datas = factory.get_numpy_dataset(audio_arrays, True)
    assert datas.shape == (len(audio_arrays), factory.num_channel*2, *factory._get_stft_shape())
    assert np.all(np.abs(audio_stfts.real - datas[:, ::2]) < 1e-15)
    assert np.all(np.abs(audio_stfts.imag - datas[:, 1::2]) < 1e-15)

def test_factory_get_numpy_dataset_success_alt():  
    config = Config(1, audio_length=10000, stft_config=LibrosaSTFTArgs())
    processor = SimpleScalingProcessor(1, 1, 0)
    factory = SpectrogramFactory(config, data_processor=processor, audio_padder=lambda x, _ : x, ordering=ListOrdering.AMPLITUDE_PHASE)
    audio_arrays = [np.random.rand(1, 10000) for _ in range(5)]
    audio_stfts = librosa.stft(np.array(audio_arrays), **LibrosaSTFTArgs())
    datas = factory.get_numpy_dataset(audio_arrays, True)
    assert datas.shape == (len(audio_arrays), factory.num_channel*2, *factory._get_stft_shape())
    assert np.all(np.abs(audio_stfts.real - datas[:, :datas.shape[1] // 2]) < 1e-15)
    assert np.all(np.abs(audio_stfts.imag - datas[:, datas.shape[1] // 2:]) < 1e-15)

def test_factory_get_spectrogram_exception():  
    config = Config(1, audio_length=10000, stft_config=LibrosaSTFTArgs())
    processor = SimpleScalingProcessor(1, 1, 0)
    factory = SpectrogramFactory(config, data_processor=processor, audio_padder=lambda x, _ : x, ordering=ListOrdering.AMPLITUDE_PHASE)
    with pytest.raises(BadTypeException):
        factory.get_spectrograms_from_files([3])
    
def test_factory_get_spectrogram_load_file():  
    config = Config(2, audio_length=10000, stft_config=LibrosaSTFTArgs())
    processor = SimpleScalingProcessor(1, 1, 0)
    factory = SpectrogramFactory(config, data_processor=processor, audio_padder=lambda x, _ : x, ordering=ListOrdering.AMPLITUDE_PHASE)
    mock_data = np.random.rand(10000, 1)
    mock_samplerate = 44100
    with patch('os.path.isfile', return_value=True):
        with patch('soundfile.read', return_value=(mock_data, mock_samplerate)) as mock_sf_read:
            factory.get_spectrograms_from_files(["path/file.wav"])
            mock_sf_read.assert_called_once_with("path/file.wav", always_2d=True)

def test_factory_get_spectrogram_channel_break():  
    config = Config(2, audio_length=10000, stft_config=LibrosaSTFTArgs())
    processor = SimpleScalingProcessor(1, 1, 0)
    factory = SpectrogramFactory(config, data_processor=processor, audio_padder=lambda x, _ : x, ordering=ListOrdering.AMPLITUDE_PHASE)
    with pytest.raises(WrongConfigurationException):
        factory.get_spectrogram_from_audio(np.random.rand(3, 1000))

def test_factory_from_model_output():
    config = Config(2, audio_length=10000, stft_config=LibrosaSTFTArgs())
    processor = SimpleScalingProcessor(1, 1, 0)
    factory = SpectrogramFactory(config, data_processor=processor, audio_padder=lambda x, _ : x, ordering=ListOrdering.AMPLITUDE_PHASE)
    model_output_mock = np.random.rand(8, 4, 256, 256)
    specs = factory.get_spectrogram_from_model_output(model_output_mock)
    assert all([isinstance(spec, MultiSpectrogram) for spec in specs])
    assert np.array_equal(model_output_mock[:,:2], np.array([spec.to_data()[::2] for spec in specs]))
    assert np.array_equal(model_output_mock[:,2:], np.array([spec.to_data()[1::2] for spec in specs]))


def test_factory_save():
    config = Config(2, audio_length=10000, stft_config=LibrosaSTFTArgs())
    processor = SimpleScalingProcessor(1, 1, 0)
    factory = SpectrogramFactory(config, data_processor=processor, audio_padder=AudioPadding.CENTER_RCUT, ordering=ListOrdering.AMPLITUDE_PHASE)
    with patch('src.SpectrogramUtils.spectrogram.spectrogram_factory.open', mock_open(), create=True) as open_mock:
        with patch("os.mkdir") as mkdir_mock:
            factory.save("save_dir")
            assert open_mock.call_count == 6
            assert mkdir_mock.call_count == 1

def test_factory_load():
    mock_file_contents = [
        bytes(json.dumps({"version": __version__}), 'utf-8'), 
        pickle.dumps(SimpleScalingProcessor(1,1,0)), 
        pickle.dumps(RealImageStftProcessor()), 
        pickle.dumps(ListOrdering.ALTERNATE), 
        pickle.dumps(Config(2, audio_length=5000)), 
        pickle.dumps(AudioPadding.CENTER_RCUT)
    ]
    mock_file_iter = iter(mock_file_contents)

    with patch('src.SpectrogramUtils.spectrogram.spectrogram_factory.open', mock_open(), create=True) as open_mock:
        with patch("os.path.isdir", return_value=True) as isdir_mock:
            open_mock.return_value.read.side_effect = lambda *args, **kwargs: next(mock_file_iter)
            factory = SpectrogramFactory.from_file("save_dir")
            assert open_mock.call_count == 6
            assert isdir_mock.call_count == 1

def test_factory_load_not_good_version():
    mock_file_contents = [
        bytes(json.dumps({"version": "pas la bonne version"}), 'utf-8'), 
        pickle.dumps(SimpleScalingProcessor(1,1,0)), 
        pickle.dumps(RealImageStftProcessor()), 
        pickle.dumps(ListOrdering.ALTERNATE), 
        pickle.dumps(Config(2, audio_length=5000)), 
        pickle.dumps(AudioPadding.CENTER_RCUT)
    ]
    mock_file_iter = iter(mock_file_contents)

    with patch('src.SpectrogramUtils.spectrogram.spectrogram_factory.open', mock_open(), create=True) as open_mock:
        with patch("os.path.isdir", return_value=True) as isdir_mock:
            with pytest.warns(match="Found factory saved on version*"):
                open_mock.return_value.read.side_effect = lambda *args, **kwargs: next(mock_file_iter)
                factory = SpectrogramFactory.from_file("save_dir")
                assert open_mock.call_count == 6
                assert isdir_mock.call_count == 1

def test_factory_load_not_a_dir():
    with patch("os.path.isdir", return_value=False) as isdir_mock:
        with pytest.raises(NotADirectoryError):
            factory = SpectrogramFactory.from_file("save_dir")

def test_factory_save_not_a_dir():
    stft_config = LibrosaSTFTArgs(center=False)
    config = Config(1, audio_length=5000, stft_config=stft_config)
    factory = SpectrogramFactory(config, data_processor=None, audio_padder=AudioPadding.CENTER_RCUT)
    with patch("os.mkdir") as mkdir_patch:
        with patch("os.path.isdir", return_value=False) as isdir_mock:
            with patch("os.path.dirname", return_value="123"):
                with pytest.raises(NotADirectoryError, match = "The directory doesn't exist :*"):
                    factory.save("save_dir")
                    assert not mkdir_patch.called 
                    assert isdir_mock.call_count == 1

def test_factory_save_not_a_file():
    stft_config = LibrosaSTFTArgs(center=False)
    config = Config(1, audio_length=5000, stft_config=stft_config)
    factory = SpectrogramFactory(config, data_processor=None, audio_padder=AudioPadding.CENTER_RCUT)
    values = iter([False, True])
    with patch("os.mkdir") as mkdir_patch:
        with patch("os.path.isdir") as isdir_mock:
            with patch("os.path.dirname", return_value=""):
                with pytest.raises(FileExistsError):
                    isdir_mock.return_value.read.side_effect = lambda : next(values)
                    factory.save("save_dir")
                    assert not mkdir_patch.called 