import numpy as np
import soundfile as sf
import os
from pathlib import Path


input_wav_directory = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'WAV_TTS','Sound_effects','E2_wavs')

output_npy_path = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'WAV_TTS', 'noise_E2.npy')

X_data = []
i = 0


list_audio_paths = sorted(list(input_wav_directory.glob('*.wav')))

for current_wav_path in list_audio_paths:
    raw_data, samplerate = sf.read(current_wav_path)
    print("{} item {}, shape {}".format(i, current_wav_path.name, str(raw_data.shape)))

    X_data.append(raw_data)

    if (np.count_nonzero(raw_data) == 0):
        print("All zeros. {} item {}".format(i, current_wav_path.name))
    i = i + 1

print("Length of X_data is {}".format(len(X_data)))

# SAVE GT
np.save(output_npy_path, X_data)

