import numpy as np
import soundfile as sf
import os
from pathlib import Path
from pedalboard import (
    Pedalboard,
    Reverb,
)

input_wav_directory = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'WAV_TTS','Sound_effects','E3_wavs')

output_npy_path = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'WAV_TTS', 'noise_E3.npy')

X_data = []
i = 0


list_audio_paths = sorted(list(input_wav_directory.glob('*.wav')))

for current_wav_path in list_audio_paths:
    raw_data, samplerate = sf.read(current_wav_path)
    print("{} item {}, shape {}".format(i, current_wav_path.name, str(raw_data.shape)))

    # Create a Pedalboard instance
    pedalboard = Pedalboard([
        Reverb(
            # mix=0.5,
            # time=3.0,
            # damping=0.5,
            room_size=0.5,
        )
    ])

    single_noise = np.asarray(raw_data)

    single_noise = single_noise * 1.6

    # Apply the effect to the audio
    processed_audio = pedalboard(single_noise, sample_rate=samplerate)

    X_data.append(processed_audio)

    if (np.count_nonzero(raw_data) == 0):
        print("All zeros. {} item {}".format(i, current_wav_path.name))
    i = i + 1

print("Length of X_data is {}".format(len(X_data)))

output_path = f"example_E3_gain16_float.wav"
sf.write(output_path, X_data[898], samplerate, subtype='FLOAT')
# SAVE GT
np.save(output_npy_path, X_data)

