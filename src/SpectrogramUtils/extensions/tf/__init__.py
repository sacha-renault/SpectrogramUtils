""" Extension for tensorflow """

import warnings

try:
    from ...spectrogram.spectrogram_factory_tf import SpectrogramTfFactory
except ImportError:
    warnings.warn("tensorflow wasn't found. Use default factory or install \
                   tensorflow to use tensorflow factory")
