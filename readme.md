# SpectrogramUtils

**_################ Work in progress #################_**

SpectrogramUtils is a library designed for handling and processing audio spectrograms. It provides tools to open audio files as spectrograms, preprocess data, and easily integrate with deep learning models, especially for generative AI tasks.

## I - Goal and main purpose.

- Open audio files as spectrograms using a factory pattern.
- Preprocess data efficiently.
- Provide a simple flow from complex arrays in STFT to float64 arrays.
- Retrieve data from a deep learning model directly as audio.
- Primarily designed for generative AI tasks, not for creating datasets for classifiers.

## II - Installation

```sh
pip install SpectrogramUtils
```

## III - Usage

### a - Basic usage : loading files

```python
# Imports
import os

from SpectrogramUtils import SpectrogramFactory
from SpectrogramUtils import Config
from SpectrogramUtils import AudioPadding

# Create a config
config = Config(2, audio_length = 44_100*5)

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
spectrograms = factory.get_spectrograms_from_files(files)
```

You can specify argument that will be used during stft and istft process in the configuration object : 
```python 
from SpectrogramUtils import Config, LibrosaSTFTArgs

stft_config = LibrosaSTFTArgs(n_fft = 512, hop_length = ...) # specify any arg that can be used during a normal stft process except dtype and pad_mode
config = Config(2, audio_length = 44_100*5, stft_config = stft_config)
```

### b - Basic usage : display spectrograms

```python
# Imports
import matplotlib.pyplot as plt
from SpectrogramUtils import DisplayType

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

# You can also set power_to_db_intensity = None to display as amplitude
config.power_to_db_intensity = None
```

You can also display the wave shapes. You cannot use the processor to display the wave shapes. The wave shapes display can display stacked wave (mutli channel in one graph)
```python
spectrogram.show_wave_on_axis(axs[0], DisplayType.STACK)
```

### c - Basic usage : data processor

#### Usage

```python
# There is two kind of data processor
# 1 - Normal data processor : doesn't need to be fitted
# 2 - Fit data processor, a first step is to fit them before using

# Example using the ScalerAudioProcessor

# Imports
from SpectrogramUtils.processors import ScalerAudioProcessor

# Init processor
processor = ScalerAudioProcessor(0.5, (0, 1)) # Target mean and min max

# Create config ... see a)
...

# Create factory
factory = SpectrogramFactory(config, processor = processor, audio_padder = AudioPadding.RPAD_RCUT)

# Make a dataset of unprocessed datas
unprocessed_data = factory.get_numpy_dataset(files, use_processor = False)

# Fit the processor
processor.fit(unprocessed_data)

# Now you can use processor any where :
# In display
spectrogram.show_image_on_axis(axs[0], DisplayType.INDEX, index = 0, use_processor = True)
# In dataset
processed_data = factory.get_numpy_dataset(files, use_processor = True)

# Any fit processor that is used before calling fit will raise an error
# Normal processor processor doesn't need to be fit
processor.save("processor.pkl") # save to skip the fit phase next time
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

- AbstractFitDataProcessor must also implement

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

### d - Basic usage : use padding

#### Usage

Basic padding function already exist, it's simple to set them up in the factory

```python
# Imports
from SpectrogramUtils import AudioPadding

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
    # The input audio arrays are shaped (num_channel, array_length)
    # The desired output shape is (num_channel, audio_length)
```

### e - Basic usage : retrieve data from a DL model

```python
# Build a dataset and train a deep learning model.
# We are more talking about generative AI, we don't need to retreive any data from a classifier.
# In this example, we train a model to denoize some audio
noized_data = ... # load files
X_data = factory.get_numpy_dataset(noized_data) # Make your own dataset
clean_data = ... # load files
y_data = factory.get_numpy_dataset(clean_data) # Make your own dataset
... # Suppose your split your dataset into train, validation and test after ...

# Define your model, can be any framework
model = ...
model.fit(X_data_train, y_data_train, validation_data = (X_data_validation, y_data_validation)) # Example of fitting

# Let your model produce some outputs
outputs = model(X_data_test)

# Retrieve the output as spectrogram object.
# It will use processor.backward() to get unprocessed data,
# so use a custom DataProcessor that is reversible, otherwise, this step will be inconsistant.
# The shape of output must be (batch, channels, *stft.shape)
output_spectrograms = factory.get_spectrogram_from_model_output(outputs)

# You can now display, save, or anything you want ...
fig, axes = plt.subplots(output_spectrograms.shape[0], 2)
for i, spectrogram in enumerate(output_spectrograms):
    spectrogram.show_image_on_axis(axs[i][0], DisplayType.INDEX, index = i)
    spectrogram.show_wave_on_axis(axs[i][1], DisplayType.INDEX, index = i)
    spectrogram.save_as_file(f"output/<model_name>_{i+1}.wav")
```

### f - Basic usage : list ordering

There is now 2 ListOrdering possible that you can set in SpectrogramFactory. It allows to change the order amplitude and phase. When the datas are passed from a list of complexe 2D array to a float 3D array, it set the amplitude every 2\*n, and phase every 2\*n + 1. For a 3 channel audio, we would have [A1, P1, A2, P2, A3, P3] (An and Pn being Amplitude and Phase of channel n).

- ListOrdering.ALTERNATE (default) : this is the default ordering, it let the normal order : [A1, P1, ..., An, Pn].
- ListOrdering.AMPLITUDE_PHASE : when calling get_numpy_dataset, it change the order to : [A1, ..., An, P1, ..., Pn]. It rearange to normal order when calling get_spectrogram_from_model_output. (Spectrogram always store as ALTERNATE order, ListOrdering just allows to have the desired order in the dataset)

```python
from SpectrogramUtils import ListOrdering

factory = SpectrogramFactory(config, processor, AudioPadding.RPAD_RCUT, ListOrdering.AMPLITUDE_PHASE)
```

### g - Basic usage : extensions
You can use torch or tensorflow extension for the factory. I allows to get dataset as Tensors instead of numpy arrays.

```python 
from SpectrogramUtils.extension import SpectrogramTorchFactory # Will raise an error if torch isn't installed 

# You can now use 
files = ... # list of file to load, could also do it with audio arrays already loaded
dataset = factory.get_torch_dataset(files, True, "cuda") # get a torch.Tensor loaded on cuda device
```

Torch extension contains a method to create a batch generator. It can be used from two different ways. 
- From an existing Tensor, it moves one batch at a time on the target device
```python
dataset = factory.get_torch_dataset(files, True, "cpu") # In the case the dataset is too large to be put directly on cuda
generator = factory.get_torch_dataset_batch_generator(dataset, batch_size = 16, "cuda", infinite_generator = False) # infinite_generator = False will make the generator raise StopIteration once it has run accross all the dataset
for batch in generator: 
    # Do things with your batch 
``` 
- From files, it loads one batch a time on target device.
```python
files = ... # File in the dataset
generator = factory.get_torch_dataset_batch_generator(dataset, batch_size = 16, "cuda", infinite_generator = False) # infinite_generator = False will make the generator raise StopIteration once it has run accross all the dataset
for batch in generator: 
    # Do things with your batch 
``` 
