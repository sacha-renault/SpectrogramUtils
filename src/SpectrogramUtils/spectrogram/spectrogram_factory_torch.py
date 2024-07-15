import warnings
from typing import Union, List, Optional, Any

import numpy as np
import numpy.typing as npt

from .multi_spectrogram import MultiSpectrogram

# try import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    raise e

from .spectrogram_factory import SpectrogramFactory

class SpectrogramTorchFactory(SpectrogramFactory):
    def to_torch_dataset(self, 
                          audio_or_file_list : Union[List[Union[str, npt.NDArray[np.float64]]], List[MultiSpectrogram]], 
                          use_processor : bool,
                          device_or_obj : Optional[Union[torch.device, Any]] = None
                          ) -> torch.Tensor:
        """
        Converts the given audio or file list to a PyTorch dataset.

        Args:
            audio_or_file_list (Union[List[Union[str, npt.NDArray[np.float64]]], List[MultiSpectrogram]]): 
                A list containing either file paths to audio files, numpy arrays representing audio data,
                or a list of MultiSpectrogram instances.
            use_processor (bool): 
                A boolean flag indicating whether to process the data before converting it to a PyTorch dataset.
            device_or_obj (Optional[Union[torch.device, Any]], optional): 
                A torch.device to move the tensor to, or an object with a 'device' attribute specifying the device.
                Defaults to None.

        Raises:
            ImportError: 
                If PyTorch is not available, this error is raised.

        Returns:
            torch.Tensor: 
                A PyTorch tensor containing the processed audio data, optionally moved to the specified device.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is not available")
        
        data = self.get_numpy_dataset(audio_or_file_list, use_processor)
        tensor = torch.Tensor(data)

        if device_or_obj is not None:
            if isinstance(device_or_obj, torch.device):
                return tensor.to(device_or_obj)
            else:
                _device = getattr(device_or_obj, "device", None)
                if _device is not None:
                    return tensor.to(_device)
                else:
                    warnings.warn(f"The provided obj for device_or_obj ({type(device_or_obj)}) does not have a 'device' attribute. Returning the tensor without moving it to a device.")
                    return tensor
        else:
            return tensor
