import numpy as np
import soundfile as sf
import pydub
from pydub.effects import compress_dynamic_range


def amplify_audio_to_0db(audio):
    # Find the maximum absolute value in the audio array
    max_amplitude = np.max(np.abs(audio))

    # Check if the maximum amplitude is zero to avoid division by zero
    if max_amplitude == 0:
        return audio

    # Calculate the scaling factor to reach 0 dB
    scaling_factor = 1.0 / max_amplitude

    # Amplify the audio by scaling all values
    amplified_audio = audio * scaling_factor

    return amplified_audio


def from_numpy_array(nparr, framerate):
    """
    Returns an AudioSegment created from the given numpy array.

    The numpy array must have shape = (num_samples, num_channels).

    :param nparr: The numpy array to create an AudioSegment from.
    :param framerate: The sample rate (Hz) of the segment to generate.
    :returns: An AudioSegment created from the given array.
    """
    # Check args
    if nparr.dtype.itemsize not in (1, 2, 4):
        raise ValueError("Numpy Array must contain 8, 16, or 32 bit values.")

    # Determine nchannels
    if len(nparr.shape) == 1:
        nchannels = 1
    elif len(nparr.shape) == 2:
        nchannels = nparr.shape[1]
    else:
        raise ValueError("Numpy Array must be one or two dimensional. Shape must be: (num_samples, num_channels), but is {}.".format(nparr.shape))

    # Fix shape if single dimensional
    nparr = np.reshape(nparr, (-1, nchannels))

    # Create an array of mono audio segments
    m = nparr[:, 0]
    dubseg = pydub.AudioSegment(m.tobytes(), frame_rate=framerate, sample_width=nparr.dtype.itemsize, channels=1)

    return dubseg

raw_data, samplerate = sf.read('DA_long_1_preComp.wav')
audio_int32 = np.int32(raw_data * 2147483647)
audio_segment = from_numpy_array(audio_int32, samplerate)

# Define the parameter ranges
thresholds = [-20.0]  # Lower thresholds for more compression
myratio = 10.0  # Higher ratio for more compression

attacks = [1.0, 5.0, 10.0]  # Faster attack times for quicker compression onset
releases = [50.0, 100.0, 150.0]  # Faster release times for quicker compression release

# attacks = [200.0]  # Faster attack times for quicker compression onset
# releases = [1000.0]  # Faster release times for quicker compression release

# Loop over the parameters
for threshold in thresholds:
    for attack in attacks:
        for release in releases:
            # Apply a dynamic range compression
            compressed_segment = compress_dynamic_range(audio_segment,
                                                        threshold=threshold,
                                                        ratio=myratio,
                                                        attack=attack,
                                                        release=release)

            compressed_numpy_int32 = compressed_segment.get_array_of_samples()
            compressed_numpy_int32 = np.array(compressed_numpy_int32)

            # Normalize and convert to float32 format
            compressed_numpy_float32 = (compressed_numpy_int32 / 2147483647).astype(np.float32)

            compressed_numpy_float32 = amplify_audio_to_0db(compressed_numpy_float32)

            # Create a filename that includes the parameters used
            new_name_wav = f'Norm_test_compression_t{threshold}_a{attack}_r{release}.wav'
            sf.write(new_name_wav, compressed_numpy_float32, samplerate, subtype='FLOAT')