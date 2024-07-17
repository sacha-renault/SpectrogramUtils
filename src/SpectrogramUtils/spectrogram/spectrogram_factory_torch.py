import warnings
from typing import Union, List, Optional, Any, Generator

import numpy as np
import numpy.typing as npt

from .multi_spectrogram import MultiSpectrogram
from ..exceptions.lib_exceptions import BadTypeException

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
                          device_or_obj : Union[torch.device, str, Any]
                          ) -> torch.Tensor:
        """
        Converts the given audio or file list to a PyTorch dataset.

        Args:
            audio_or_file_list (Union[List[Union[str, npt.NDArray[np.float64]]], List[MultiSpectrogram]]): 
                A list containing either file paths to audio files, numpy arrays representing audio data,
                or a list of MultiSpectrogram instances.
            use_processor (bool): 
                A boolean flag indicating whether to process the data before converting it to a PyTorch dataset.
            device_or_obj [Union[torch.device, Any]]: 
                A torch.device to move the tensor to, or an object with a 'device' attribute specifying the device.

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
            self._to_device(tensor, device_or_obj)
        else:
            return tensor
        

    def _to_device(self, tensor : torch.Tensor, device_or_obj : Union[torch.device, str, Any]):
        device = self._get_device(device_or_obj)
        if device is not None:
            return tensor.to(device)
        else:
            warnings.warn(f"The provided obj for device_or_obj ({type(device_or_obj)}) does not have a 'device' attribute. Returning the tensor without moving it to a device.")
            return tensor     
            
    def _get_device(device_or_obj : Union[torch.device, str, Any]):
        if isinstance(device_or_obj, torch.device):
            return device_or_obj
        elif isinstance(device_or_obj, str):
            return str
        else:
            _device = getattr(device_or_obj, "device", None)
            if _device is not None:
                return _device
            else:
                return None
            

    def get_torch_dataset_batch_generator(self, 
                          file_list_or_tensor : Union[List[str], torch.Tensor], 
                          use_processor : bool,
                          batch_size : int,
                          device_or_obj : Union[torch.device, Any],
                          ) -> Generator[torch.Tensor]:
        """
        Generates batches of torch Tensors from a list of file paths or a pre-loaded torch Tensor.

        Args:
            file_list_or_tensor (Union[List[str], torch.Tensor]): 
                A list of file paths to load data from or a pre-loaded torch Tensor.
            use_processor (bool): 
                A flag indicating whether to use a processor for data loading and transformation.
            batch_size (int): 
                The size of each batch to generate.
            device_or_obj (Union[torch.device, Any]): 
                The device on which the tensors should be loaded. Can be a torch.device object or any object with a 'device' attribute.

        Raises:
            BadTypeException: 
                If the input type of `file_list_or_tensor` is neither a list of strings nor a torch Tensor.

        Returns:
            Generator[torch.Tensor, None, None]: 
                A generator that yields batches of torch Tensors.

        Yields:
            torch.Tensor: 
                Batches of data as torch Tensors, moved to the specified device.

        Examples:
            Generating batches from a list of file paths:
            >>> file_list = ["file1.wav", "file2.wav", "file3.wav"]
            >>> generator = factory.get_torch_dataset_batch_generator(file_list, use_processor=True, batch_size=2, device_or_obj='cpu')
            >>> for batch in generator:
            >>>     print(batch)

            Generating batches from a pre-loaded tensor:
            >>> tensor = torch.randn(100, 20)
            >>> generator = factory.get_torch_dataset_batch_generator(tensor, use_processor=False, batch_size=10, device_or_obj='cuda')
            >>> for batch in generator:
            >>>     print(batch)
        """

        # Type checking
        first_elt = file_list_or_tensor[0]
        assert all(map(lambda x : type(x), file_list_or_tensor) == type(first_elt)), "All instances in the input list must be same type"

        # If lst_type is string, we want a generator that load datas from disk 
        if isinstance(first_elt, str):
            return self._torch_disk_load_generator(file_list_or_tensor, batch_size, use_processor, device_or_obj)
        
        # In the other cases, we let the tensor on device, and generator just provides one batch on target device at a time
        elif isinstance(file_list_or_tensor, torch.Tensor):
            return self._torch_generator_to_device(file_list_or_tensor, batch_size, use_processor, device_or_obj)
        
        else:
            raise BadTypeException(f"Cannot make a generator from type : {(type(file_list_or_tensor))}")

    def _torch_disk_load_generator(self, 
                                   file_list : List[str], 
                                   batch_size : int, 
                                   use_processor : bool, 
                                   device_or_obj : Optional[Union[torch.device, Any]]
                                   ) -> Generator[torch.Tensor]:
        
        # Get devices and num batches
        device = self._get_device(device_or_obj)
        num_batches = len(file_list) // batch_size
        while True:
            for i in range(num_batches + 1): # + 1 because we could have a smaller batch in the end
                # Get files for current batch
                batch_files = file_list[i*batch_size:(i+1)*batch_size]

                # Get numpy dataset for this batch
                batch = self.get_numpy_dataset(batch_files, use_processor)

                # Convert into a torch tensor
                yield torch.Tensor(batch).to(device)


    def _torch_generator_to_device(self, 
                                   tensor: torch.Tensor, 
                                   batch_size: int, 
                                   device_or_obj: Optional[Union[torch.device, Any]]
                                   ) -> Generator[torch.Tensor, None, None]:
        # Get devices and num batches
        device = self._get_device(device_or_obj)
        num_batches = tensor.size(0) // batch_size
        while True:
            for i in range(num_batches + 1): # + 1 because we could have a smaller batch in the end
                batch = tensor[i*batch_size:(i+1)*batch_size].to(device)
                yield batch