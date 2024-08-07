""" Module that define the DataProcessorWrapper"""
from .abstract_data_processor import AbstractDataProcessor, \
    AbstractFitDataProcessor, AbstractDestructiveDataProcessor
from ..exceptions.lib_exceptions import NoProcessorException, ProcessorNotFittedException
from ..data.types import MixedPrecision2DArray

class DataProcessorWrapper:
    """Container for DataProcessor, allow factory to know if it can use it
    """
    def __init__(self, processor : AbstractDataProcessor) -> None:
        self.__need_check = not isinstance(processor, AbstractDestructiveDataProcessor)
        self.__is_checked = False
        self.__processor = processor

    def forward(self, data : MixedPrecision2DArray) -> MixedPrecision2DArray:
        """Preprocess datas, transformation must be reversible to get back to initial state in
        backward (i.e. self.backward(self.forward(data)) must be same as data)

        Args:
            data (MixedPrecision2DArray): single data

        Returns:
            MixedPrecision2DArray: processed data
        """
        if self.__processor is None:
            raise NoProcessorException("Processor is None")
        if isinstance(self.__processor, AbstractFitDataProcessor) and not self.__processor.is_fitted:
            raise ProcessorNotFittedException("Fit processor must be fitted before using it.")
        if self.__need_check and not self.__is_checked:
            self.__processor.check_reversible()
            self.__is_checked = True
        return self.__processor.forward(data)

    def backward(self, data : MixedPrecision2DArray) -> MixedPrecision2DArray:
        """Get back to inital state

        Args:
            data (MixedPrecision2DArray): single processed data

        Returns:
            MixedPrecision2DArray: deprocessed data
        """
        if self.__processor is None:
            raise NoProcessorException("Processor is None")
        if isinstance(self.__processor, AbstractFitDataProcessor) and not self.__processor.is_fitted:
            raise ProcessorNotFittedException("Fit processor must be fitted before using it.")
        if self.__need_check and not self.__is_checked:
            self.__processor.check_reversible()
            self.__is_checked = True
        return self.__processor.backward(data)

    @property
    def processor(self) -> AbstractDataProcessor:
        """ Access the processor stored in the wrapper """
        return self.__processor
