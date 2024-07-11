from enum import Enum, auto
from typing import Callable

from ..misc.utils import lpad_lcut, rpad_rcut, center_pad_rcut

class DisplayType(Enum):
    # # # # # # #
    MEAN = auto()
    """Display the average bewteen left and right spectrogram"""

    # # # # # # #
    STACK = auto()
    """Only available for wave display. \nDisplay all the waves in one graph."""

    # 
    INDEX = auto()
    """Display the stft at specified index"""

    # # # # # # #
    MAX = auto()
    """Display maximum value if more than one provided"""

    # # # # # # # 
    MIN = auto()
    """Display minimum value if more than one provided"""

class AudioPadding:
    NONE = None
    """No padding at all"""

    RPAD_RCUT = rpad_rcut
    """
        Add zeros values on the right if the audio is too small. 
        Cut the end of the audio if too long.
    """

    LPAD_LCUT = lpad_lcut
    """
        Add zeros on the left if the audio is too small. 
        Cut the start of the audio if too long 
    """

    CENTER_RCUT = center_pad_rcut
    """
        Add zeros in the left and right to center the audio if too small.
        Cut the end of the audio if too long. 
    """

class ListOrdering(Enum):
    ALTERNATE = auto()
    """
        Altenate bewteen amplitude and phase. 
        Any 2*n is an amplitude and 2*n + 1 is a phase
    """

    AMPLITUDE_PHASE = auto()
    """
        First half is all the amplitudes, and second hald is all phases
    """