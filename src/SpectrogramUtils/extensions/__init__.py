import warnings

try:
    from ..spectrogram.spectrogram_factory_tf import SpectrogramTfFactory
except ImportError:
    warnings.warn("tensorflow wasn't found. Use default factory or install tensorflow to use tensorflow factory")

try:
    from ..spectrogram.spectrogram_factory_torch import SpectrogramTorchFactory
except ImportError:
    warnings.warn("torch wasn't found. Use default factory or install torch to use torch factory")
