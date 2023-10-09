import scipy.signal as signal
import numpy as np
import soundfile as sf

# Load the audio file
audio, sample_rate = sf.read('DA_long-0.wav')

# Calculate the filter coefficients
nyquist_freq = 0.5 * sample_rate
cutoff_freq = 60.0 / nyquist_freq
b, a = signal.butter(1, cutoff_freq, 'highpass', analog=False, output='ba')

# Apply the filter to the audio signal
filtered_audio = signal.filtfilt(b, a, audio)

# Save the filtered audio to a new file
sf.write('filtered_audio.wav', filtered_audio, sample_rate)
