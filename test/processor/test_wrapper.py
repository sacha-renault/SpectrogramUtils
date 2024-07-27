import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open

import numpy as np

from src.SpectrogramUtils import SpectrogramFactory, Config, ScalerAudioProcessor, AudioPadding, \
    SimpleScalingProcessor, LibrosaSTFTArgs, ListOrdering, MultiSpectrogram, RealImageStftProcessor
from src.SpectrogramUtils.processors.wrapper import DataProcessorWrapper
from src.SpectrogramUtils.exceptions.lib_exceptions import ProcessorNotFittedException, NoProcessorException
from src.SpectrogramUtils._version import version as __version__

def test_no_processor():
    wrapper = DataProcessorWrapper(None)
    with pytest.raises(NoProcessorException):
        wrapper.forward(np.random.rand(10,10))

    with pytest.raises(NoProcessorException):
        wrapper.backward(np.random.rand(10,10))


def test_not_fitted_processor():
    processor = ScalerAudioProcessor()
    wrapper = DataProcessorWrapper(processor)
    with pytest.raises(ProcessorNotFittedException):
        wrapper.forward(np.random.rand(10,10))

    with pytest.raises(ProcessorNotFittedException):
        wrapper.backward(np.random.rand(10,10))

    assert wrapper.processor is processor