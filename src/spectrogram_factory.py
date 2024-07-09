from .config import Config

class SpectrogramFactory:
    def __init__(self, config : Config) -> None:
        # Set the config
        self.__config = config
    
    