""" Utility functions """
from typing import List

import librosa
import numpy as np
import numpy.typing as npt

from ..exceptions.lib_exceptions import UnknownStftShapeException
from ..data.types import Complex2DArray, MixedPrecision2DArray

def get_backward_indexer(forward_indexer : npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """For a given forward indexer, check it is valid and get it's backward indexing

    Args:
        forward_indexer (npt.NDArray[np.int_]): forward_indexer

    Returns:
        npt.NDArray[np.int_]: the backward indexer
    """
    # Assert the indexer is 1D array
    assert len(forward_indexer.shape) == 1,\
        f"forward_indexer should be a 1D array, found shape dimension: {len(forward_indexer.shape)}"

    # Assert int values
    assert forward_indexer.dtype == np.int_,\
        f"forward_indexer should be dtyped int, found : {forward_indexer.dtype}"

    # backward indexer
    backward_indexer = np.argsort(forward_indexer)

    # Check that forward_indexer is an arangement,
    # forward_indexer[backward_indexer] is sorted forward_indexer
    assert np.array_equal(np.arange(forward_indexer.shape[0]), forward_indexer[backward_indexer]),\
        "The indexer isn't an arrangement"

    # get the backward_indexer
    return backward_indexer

def get_multi_stft(audio_array : MixedPrecision2DArray, **stft_kwargs) -> List[Complex2DArray]:
    """ Get stft(s) with librosa"""
    # Ensure the input array is at most 2D
    assert len(audio_array.shape) <= 2

    # process stft
    result = librosa.stft(audio_array, **stft_kwargs)

    # check if it was multi-channel
    if len(result.shape) == 2:
        return [result]
    if len(result.shape) == 3:
        return list(result)

    # Else
    raise UnknownStftShapeException("Unknown shape during stft process")

def rpad_rcut(data : MixedPrecision2DArray, desired_audio_length : int) -> MixedPrecision2DArray:
    """ Pad or cut the audio array so that output has a length equal to desired_audio_length
    Args:
        data (MixedPrecision2DArray): the input audio array
        desired_audio_length (int): the target length for the audio
    Return
        (np.ndarray): correctly shaped audio array
    """
    assert len(data.shape) == 2, "Audio should be 2D array, use reshape(1, -1) for 1D array"
    audio_length = data.shape[1]
    if audio_length < desired_audio_length:
        padding_array = np.zeros((data.shape[0], desired_audio_length - audio_length))
        return np.concatenate((data, padding_array), axis = 1)

    # Else
    return data[:,:desired_audio_length]

def lpad_lcut(data : MixedPrecision2DArray, desired_audio_length : int) -> MixedPrecision2DArray:
    """ Pad or cut the audio array so that output has a length equal to desired_audio_length
    Args:
        data (MixedPrecision2DArray): the input audio array
        desired_audio_length (int): the target length for the audio
    Return
        (MixedPrecision2DArray): correctly shaped audio array
    """
    assert len(data.shape) == 2, "Audio should be 2D array, use reshape(1, -1) for 1D array"
    audio_length = data.shape[1]
    if audio_length < desired_audio_length:
        padding_array = np.zeros((data.shape[0], desired_audio_length - audio_length))
        return np.concatenate((padding_array, data), axis = 1)

    # Else
    return data[:,desired_audio_length:]

def center_pad_rcut(data : MixedPrecision2DArray, desired_audio_length : int) -> MixedPrecision2DArray:
    """ Pad or cut the audio array so that output has a length equal to desired_audio_length
    Args:
        data (MixedPrecision2DArray): the input audio array
        desired_audio_length (int): the target length for the audio
    Return
        (MixedPrecision2DArray): correctly shaped audio array
    """
    assert len(data.shape) == 2, "Audio should be 2D array, use reshape(1, -1) for 1D array"
    audio_length = data.shape[1]
    if audio_length < desired_audio_length:
        l_pad_length = (desired_audio_length - audio_length) // 2
        r_pad_length = l_pad_length + (desired_audio_length - audio_length) % 2
        l_padding_array = np.zeros((data.shape[0], l_pad_length))
        r_padding_array = np.zeros((data.shape[0], r_pad_length))
        return np.concatenate((l_padding_array, data, r_padding_array), axis = 1)

    # Else
    return data[:,:desired_audio_length]
