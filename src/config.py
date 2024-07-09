class Config:
    def __init__(self,
                 num_channel : int,
                 n_fft : int = None,
                 hop_length : int = None,
                 audio_length : int = None,
                 ) -> None:
        # Assert some
        assert audio_length > 0
        assert hop_length > 0
        assert num_channel > 0
        assert n_fft > 0
        
        # Set attribute
        self.n_fft = n_fft 
        self.hop_length = hop_length
        self.audio_length = audio_length
        self.num_channel = num_channel

    def get_istft_kwargs(self) -> dict:
        kwargs = {
            "n_fft" : self.n_fft,
            "hop_length" : self.hop_length
        }
        return kwargs