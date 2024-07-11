# SpectrogramsUtils

## I - Goals
- Open audio as spectrograms using a factory
- Preprocess datas
- Having a simple flow from complexe arrays in stft to float 64 arrays
- Get back datas from a DL model directly as audio

## II - Usage

### a) Basic usage : loading files
```python
# Imports
import os 

from SpectrogramsUtils import SpectrogramFactory
from SpectrogramsUtils import Config
from SpectrogramsUtils import AudioPadding

# Create a config
config = Config(2, n_fft = 512, audio_length = 44_100*5)

# Factory with no audio processor
factory = SpectrogramFactory(config, audio_padder = AudioPadding.RPAD_RCUT) 

# Load a single audio file
spectrogram = factory.get_spectrogram_from_path("path/to/file.wav")

# Load one spectrogram for each audio in a folder
audio_directory = "path/to/directory"
files = [
    os.path.join(audio_directory, audio_file) 
    for audio_file in os.listdir(audio_directory)
]
factory.get_spectrograms(files)
```

### b) Basic usage : display spectrograms
```python
# Imports
import matplotlib.pyplot as plt
from SpectrogramsUtils import DisplayType

# Load a single audio file with 2 channels (i.e. stereo audio)
spectrogram = factory.get_spectrogram_from_path("path/to/file.wav")

# Create axes
_, axs = plt.subplots(3, 1)

# Use config  field power_to_db_intensity to set the intensity of power_to_db
# By default, is config.power_to_db_intensity is None, it will display the spectrogram as amplitude, not as db
config.power_to_db_intensity = 2 # 2 is equivalent to amplitude_to_db, but user can set any value

# Display each channel and the mean in axes
spectrogram.show_image_on_axis(axs[0], DisplayType.INDEX, index = 0)
spectrogram.show_image_on_axis(axs[0], DisplayType.INDEX, index = 1)
spectrogram.show_image_on_axis(axs[0], DisplayType.MEAN)
```