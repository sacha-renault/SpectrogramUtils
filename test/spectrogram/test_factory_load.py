import pytest
from unittest.mock import patch, mock_open
import json
import pickle

from src.SpectrogramUtils import SpectrogramFactory, Config, AudioPadding, \
    SimpleScalingProcessor, RealImageStftProcessor
from src.SpectrogramUtils.exceptions.lib_exceptions import \
   VersionNotCompatibleException
from src.SpectrogramUtils.misc.indexers import get_forward_indexer_amplitude_phase
from src.SpectrogramUtils._version import VERSION as __version__

def test_factory_load():
    mock_file_contents = [
        bytes(json.dumps({"version": __version__}), 'utf-8'),
        pickle.dumps(SimpleScalingProcessor(1,1,0)),
        pickle.dumps(RealImageStftProcessor()),
        pickle.dumps(get_forward_indexer_amplitude_phase(2)),
        pickle.dumps(Config(2, audio_length=5000)),
        pickle.dumps(AudioPadding.CENTER_RCUT)
    ]
    mock_file_iter = iter(mock_file_contents)

    with patch('src.SpectrogramUtils.spectrogram.spectrogram_factory.open', mock_open(), create=True) as open_mock:
        with patch("os.path.isdir", return_value=True) as isdir_mock:
            with patch("src.SpectrogramUtils.spectrogram.spectrogram_factory.are_versions_compatible", return_value=True) as versions_mock:
                open_mock.return_value.read.side_effect = lambda *args, **kwargs: next(mock_file_iter)
                factory = SpectrogramFactory.from_file("save_dir")
                assert open_mock.call_count == 6
                assert isdir_mock.call_count == 1

def test_factory_load_not_good_version():
    mock_file_contents = [
        bytes(json.dumps({"version": "0.0.0"}), 'utf-8'),
        pickle.dumps(SimpleScalingProcessor(1,1,0)),
        pickle.dumps(RealImageStftProcessor()),
        pickle.dumps(get_forward_indexer_amplitude_phase(2)),
        pickle.dumps(Config(2, audio_length=5000)),
        pickle.dumps(AudioPadding.CENTER_RCUT)
    ]
    mock_file_iter = iter(mock_file_contents)

    with patch('src.SpectrogramUtils.spectrogram.spectrogram_factory.open', mock_open(), create=True) as open_mock:
        with patch("os.path.isdir", return_value=True) as isdir_mock:
            with patch("src.SpectrogramUtils.spectrogram.spectrogram_factory.are_versions_compatible", return_value=False) as versions_mock:
                with pytest.raises(VersionNotCompatibleException):
                    open_mock.return_value.read.side_effect = lambda *args, **kwargs: next(mock_file_iter)
                    factory = SpectrogramFactory.from_file("save_dir")
                    assert open_mock.call_count == 6
                    assert isdir_mock.call_count == 1