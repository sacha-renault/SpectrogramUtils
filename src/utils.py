import librosa
import numpy as np

def get_square_stft_pairs_by_audio_length(audio_len: int) -> list:
    pairs = []
    
    # Iterate over possible hop lengths
    for hop_length in range(1, min(audio_len, 500)):
        num_time_bins = audio_len // hop_length
        
        # Calculate nfft such that the number of frequency bins equals the number of time bins
        nfft = (num_time_bins - 1) * 2
        
        # Ensure nfft is a valid value
        if nfft > 0:
            # Find the closest power of 2 greater than or equal to this value
            nfft_power_of_2 = 2**np.ceil(np.log2(nfft)).astype(int)
            
            # Calculate the number of frequency bins
            num_freq_bins = nfft_power_of_2 // 2 + 1
            
            # Check if the STFT will be square
            if num_freq_bins == num_time_bins:
                # Add the (hop_length, nfft, shape) tuple to the list
                pairs.append((hop_length, nfft_power_of_2, (num_freq_bins, num_time_bins)))
    
    return pairs

def get_values_for_stft_shape(desired_stft_shape: tuple[int, int]) -> list:
    desired_num_freq_bins, desired_num_time_bins = desired_stft_shape
    
    # Calculate the required nfft to achieve the desired number of frequency bins
    nfft = (desired_num_freq_bins - 1) * 2 - 5
    
    # Ensure nfft is a power of 2 for optimal FFT computation
    nfft_power_of_2 = 2**np.ceil(np.log2(nfft)).astype(int)
    
    possible_values = []

    # Iterate over possible hop lengths to find audio lengths
    for hop_length in range(1, 500):  # Adjust the range if needed
        # Calculate the resulting audio length that would give the desired number of time bins
        audio_len = hop_length * (desired_num_time_bins - 1)
        
        # Add the (hop_length, nfft, audio_len) tuple to the list
        possible_values.append((hop_length, nfft_power_of_2, audio_len, (desired_num_freq_bins, desired_num_time_bins)))
    
    # Sort the list by audio length
    possible_values.sort(key=lambda x: x[2])
    
    return possible_values


stft = librosa.stft(np.random.rand(15000), hop_length=32, n_fft = 511)
print(stft.shape)