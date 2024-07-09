from enum import Enum, auto

class DisplayType(Enum):
    # Display the average bewteen left and right spectrogram
    MEAN = auto()
    """Display the average bewteen left and right spectrogram"""

    # Only available for wave display
    STACK = auto()
    """Only available for wave display"""

    # Display the stft at specified index
    INDEX = auto()
    """Display the stft at specified index"""