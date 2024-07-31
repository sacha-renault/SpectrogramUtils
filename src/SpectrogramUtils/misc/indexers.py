""" Module that reference different function to get indexers array"""
import numpy as np

from ..data.types import ArangementPermutation

def get_forward_indexer_amplitude_phase(num_channel : int, dim_per_channel : int = 2) -> ArangementPermutation:
    """
    Generates an indexer that arranges data as [Amplitude, Amplitude, ..., Phase, Phase, ...].

    This function creates an indexer that organizes the data such that all phase components are
    grouped together followed by all amplitude components. This can be useful for processing
    audio data where phase and amplitude components need to be separated.

    Args:
        num_channel (int): The number of channels in the audio data.
        dim_per_channel (int, optional): The number of dimensions per channel. Defaults to 2.

    Returns:
        ArangementPermutation: An array of indices that can be used to reorder the data into the
        desired arrangement.
    """
    arangement = np.arange(num_channel * dim_per_channel)
    return np.concatenate((arangement[::2], arangement[1::2]))

def get_forward_indexer_phase_amplitude(num_channel : int, dim_per_channel : int = 2) -> ArangementPermutation:
    """
    Generates an indexer that arranges data as [Phase, Phase, ..., Amplitude, Amplitude, ...].

    This function creates an indexer that organizes the data such that all phase components are
    grouped together followed by all amplitude components. This can be useful for processing
    audio data where phase and amplitude components need to be separated.

    Args:
        num_channel (int): The number of channels in the audio data.
        dim_per_channel (int, optional): The number of dimensions per channel. Defaults to 2.

    Returns:
        ArangementPermutation: An array of indices that can be used to reorder the data into the
        desired arrangement.
    """
    arangement = np.arange(num_channel * dim_per_channel)
    return np.concatenate((arangement[1::2], arangement[::2]))
