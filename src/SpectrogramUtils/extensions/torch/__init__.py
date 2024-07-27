""" Extension for torch """
import warnings

try:
    from ...spectrogram.spectrogram_factory_torch import SpectrogramTorchFactory
except ImportError:
    warnings.warn("torch wasn't found. Use default factory or install torch to use torch factory")
    