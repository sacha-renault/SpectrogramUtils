from enum import Enum, auto
from typing import Callable

class DisplayType(Enum):
    # Display the average bewteen left and right spectrogram
    MEAN = auto()
    """Display the average bewteen left and right spectrogram"""

    # Only available for wave display
    STACK = auto()
    """Only available for wave display. \nDisplay all the waves in one graph."""

    # Display the stft at specified index
    INDEX = auto()
    """Display the stft at specified index"""

class AudioPadding:
    NONE = None
    """No padding at all"""

    RPAD_RCUT = None
    """
        Add zeros values on the right if the audio is too small. 
        Cut the end of the audio if too long.
    """

    LPAD_LCUT = None
    """
        Add zeros on the left if the audio is too small. 
        Cut the start of the audio if too long 
    """

    CENTER_RCUT = None
    """
        Add zeros in the left and right to center the audio if too small.
        Cut the end of the audio if too long. 
    """