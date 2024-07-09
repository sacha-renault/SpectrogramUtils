class Config:
    def __init__(self,
                 n_fft : int = None
                 ) -> None:
        self.n_fft : int = n_fft 

    def get_istft_kwargs(self) -> dict:
        kwargs = {
            "n_fft" : self.n_fft
        }

        return kwargs