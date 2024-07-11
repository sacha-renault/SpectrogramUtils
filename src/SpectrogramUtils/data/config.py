class Config:
    def __init__(self,
                 num_channel : int,
                 sample_rate : int = 44100,
                 n_fft : int = None,
                 hop_length : int = None,
                 audio_length : int = None,
                 power_to_db_intensity : float = None
                 ) -> None:
        # Assert some
        assert num_channel > 0
        assert sample_rate > 0
        if hop_length is not None:
            assert hop_length > 0
        if audio_length is not None:
            assert audio_length > 0
        if n_fft is not None:
            assert n_fft > 0
        if power_to_db_intensity is not None:
            assert power_to_db_intensity > 0
        
        # Set attribute
        self.n_fft = n_fft 
        self.hop_length = hop_length
        self.audio_length = audio_length
        self.num_channel = num_channel
        self.sample_rate = sample_rate
        self.power_to_db_intensity = power_to_db_intensity

    def get_istft_kwargs(self) -> dict:
        kwargs = {}
        if self.n_fft is not None:
            kwargs["n_fft"] = self.n_fft
        if self.hop_length is not None:
            kwargs["hop_length"] = self.hop_length
        return kwargs