class LibrosaSTFTArgs(dict):
    def __init__(self, **kwargs):
        # Default arguments for librosa.stft
        default_args = {
            'n_fft': 2048,
            'hop_length': None,  # defaults to n_fft // 4
            'win_length': None,  # defaults to n_fft
            'window': 'hann',
            'center': True,
        }
        
        # Initialize with default arguments, then update with any user-provided arguments
        super().__init__({**default_args, **kwargs})
        
        # Handle derived default values
        if self['hop_length'] is None:
            self['hop_length'] = self['n_fft'] // 4
        if self['win_length'] is None:
            self['win_length'] = self['n_fft']