""" Module that sotre exception """

class SpectrogramUtilsException(Exception):
    """ Base exception for library """
class UnknownStftShapeException(SpectrogramUtilsException):
    """#### Thrown if stft process create bad shaped stfts
    """

class UnknownProcessorSaveFileDataException(SpectrogramUtilsException):
    """#### Thrown if load method doesn't return a valid data to restaure processor state
    """

class NoProcessorException(SpectrogramUtilsException):
    """#### Thrown if try to use a processor in a factory without declaring one
    """

class NoIndexException(SpectrogramUtilsException):
    """#### Thrown if set display type to INDEX without providing one
    """

class WrongDisplayTypeException(SpectrogramUtilsException):
    """#### Thrown if a display type isn't available
    """

class WrongConfigurationException(SpectrogramUtilsException):
    """#### Thrown if configuration is broken
    """

class UnknownWavTypeException(SpectrogramUtilsException):
    """#### Thrown if trying to save to wav a file that has more than 2 channels, try to use custom save method after calling method get_waves()
    """

class BadTypeException(SpectrogramUtilsException):
    """#### Thrown if trying to load a spectrogram from a wrong type
    """

class BrokenProcessorException(SpectrogramUtilsException):
    """#### Processor is broken and data pass through forward then backward are not retrieved
    """

class ProcessorNotFittedException(SpectrogramUtilsException):
    """#### Processor is not fitted before usage
    """