# SpectrogramsUtils

***################ Work in progress #################***

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
spectrograms = factory.get_spectrograms(files)
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

# Use config  field power_to_db_intensity to set the intensity of power_to_db, 
# and display the spectrogram as dB
# By default, is config.power_to_db_intensity is None and
# it will display the spectrogram as amplitude, not as dB
config.power_to_db_intensity = 2 # 2 is equivalent to amplitude_to_db, but user can set any value

# Display each channel and the mean in axes
spectrogram.show_image_on_axis(axs[0], DisplayType.INDEX, index = 0) # Need to provide an index when display type is INDEX
spectrogram.show_image_on_axis(axs[1], DisplayType.INDEX, index = 1) # Need to provide an index when display type is INDEX
spectrogram.show_image_on_axis(axs[2], DisplayType.MEAN) # No need to provide an index when display type is not INDEX
plt.show()
```

### c) Basic usage : data processor
#### Usage
```python
# There is two kind of data processor
# 1 - Normal data processor : doesn't need to be fitted
# 2 - Fit data processor, a first step is to fit them before using

# Example using the ScalerAudioProcessor

# Imports
from SpectrogramsUtils.processors import ScalerAudioProcessor

# Init processor
processor = ScalerAudioProcessor(0.5, (0, 1)) # Target mean and min max

# Create config ... see a)
...

# Create factory
factory = SpectrogramFactory(config, processor = processor, audio_padder = AudioPadding.RPAD_RCUT) 

# Make a dataset of unprocessed datas
unprocessed_data = get_numpy_dataset(files, use_processor = False)

# Fit the processor
processor.fit(unprocessed_data)

# Now you can use processor any where : 
# In display 
spectrogram.show_image_on_axis(axs[0], DisplayType.INDEX, index = 0, use_processor = True)
# In dataset
processed_data = get_numpy_dataset(files, use_processor = True)

# Any fit processor that is used before calling fit will raise an error
# Normal processor processor doesn't need to be fit
```

#### Create
Data processor must inheritate from either AbstractFitDataProcessor or AbstractDataProcessor
- AbstractDataProcessor processors must implement
```python
class YourDataProcessor(AbstractDataProcessor):
    def forward(self, data : np.ndarray) -> npt.NDArray[np.float64]:
        """#### Preprocess datas, transformation must be reversible to get back to initial state in backward
        (i.e. self.backward(self.forward(data)) must be same as data)

        #### Args:
            - data (np.ndarray): single data

        #### Returns:
            - npt.NDArray[np.float64]: processed data
        """
        # TODO

    def backward(self, data : np.ndarray) -> npt.NDArray[np.float64]:
        """#### Get back to inital state

        #### Args:
            - data (np.ndarray): single processed data

        #### Returns:
            - npt.NDArray[np.float64]: deprocessed data
        """
        # TODO
```

- AbstractFitDataProcessor must implement
```python
class YourFitDataProcessor(AbstractFitDataProcessor):
    def fit(self, fit_data : np.ndarray) -> None:
        """#### Fit the processor to training datas. It should set is_fitted to True

        #### Args:
            - fit_data (np.ndarray): training data
        """
        # TODO
        # MUST SET is_fitted to True after fit is done
        self.is_fitted = True

    def save(self, file : Union[str, list[str]]) -> None:
        """#### Save the current state of the processor into a file

        #### Args:
            - file (Union[str, list[str]]): file or files to save a processor states. 
        """
        # TODO

    def load(self, file : Union[str, list[str]]) -> None:
        """#### Restaure the processor to a saved states, it should set is_fitted to True. 

        #### Args:
            - file (Union[str, list[str]]): file or files to restaure a processor states. 
        """
        # TODO
```

### c) Basic usage : use padding
#### Usage
Basic padding function already exist, it's simple to set them up in the factory
```python
# Imports
from SpectrogramsUtils import AudioPadding

# Create factory
factory1 = SpectrogramFactory(config, processor = processor, audio_padder = AudioPadding.RPAD_RCUT) 
factory1 = SpectrogramFactory(config, processor = processor, audio_padder = AudioPadding.LPAD_LCUT) 
```

You can create your own padding function
```python
def my_padding_function(audio_array : npt.NDArray, audio_length : int) -> npt.NDArray:
    # TODO
    # PAD the audio if too small compare to the target audio length
    # CUT the audio if it's too long
```