from .librosa_stft_args import LibrosaSTFTArgs

class Config:
    def __init__(self,
                 num_channel : int,
                 stft_config : LibrosaSTFTArgs = None,
                 sample_rate : int = 44100,
                 audio_length : int = None,
                 power_to_db_intensity : float = None
                 ) -> None:
        # Assert some
        assert num_channel > 0
        assert sample_rate > 0

        if audio_length is not None:
            assert audio_length > 0
        if power_to_db_intensity is not None:
            assert power_to_db_intensity > 0
        if stft_config is not None:
            assert isinstance(stft_config, LibrosaSTFTArgs)
        
        # Set attribute
        self.audio_length = audio_length
        self.num_channel = num_channel
        self.sample_rate = sample_rate
        self.power_to_db_intensity = power_to_db_intensity
        self.stft_config = stft_config if stft_config is not None else LibrosaSTFTArgs() # default values

    def get_istft_kwargs(self) -> dict:
        return self.stft_config