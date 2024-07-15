import warnings
from typing import Union, List

import numpy as np
import numpy.typing as npt

from .multi_spectrogram import MultiSpectrogram

# try import torch
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError as e:
    raise e

from .spectrogram_factory import SpectrogramFactory

class SpectrogramTfFactory(SpectrogramFactory):
    def to_torch_dataset(self, 
                          audio_or_file_list : Union[List[Union[str, npt.NDArray[np.float64]]], List[MultiSpectrogram]], 
                          use_processor : bool
                          ) -> tf.Tensor:
        if not TF_AVAILABLE:
            raise ImportError("torch is not available")
        
        data = self.get_numpy_dataset(audio_or_file_list, use_processor)
        tensor = tf.Tensor(data)
        return tensor
