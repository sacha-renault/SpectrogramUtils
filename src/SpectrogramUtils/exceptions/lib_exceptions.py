class SpectrogramUtilsException(Exception):
    pass

class UnknownStftShapeException(SpectrogramUtilsException):
    """#### Thrown if stft process create bad shaped stfts
    """
    pass

class UnknownProcessorSaveFileDataException(SpectrogramUtilsException):
    """#### Thrown if load method doesn't return a valid data to restaure processor state
    """
    pass

class NoProcessorException(SpectrogramUtilsException):
    """#### Thrown if try to use a processor in a factory without declaring one
    """
    pass

class NoIndexException(SpectrogramUtilsException):
    """#### Thrown if set display type to INDEX without providing one
    """
    pass

class WrongDisplayTypeException(SpectrogramUtilsException):
    """#### Thrown if a display type isn't available
    """
    pass

class WrongConfigurationException(SpectrogramUtilsException):
    """#### Thrown if configuration is broken
    """
    pass

class UnknownWavTypeException(SpectrogramUtilsException):
    """#### Thrown if trying to save to wav a file that has more than 2 channels, try to use custom save method after calling method get_waves()
    """
    pass

class BadTypeException(SpectrogramUtilsException):
    """#### Thrown if trying to load a spectrogram from a wrong type
    """
    pass