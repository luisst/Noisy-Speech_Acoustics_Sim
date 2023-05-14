import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import librosa
from pathlib import Path

wav_path = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'WAV_TTS',
                                      'All_WAVs','3999_Salli.wav')

# Load the speech audio produced by a Text-To-Speech System
speech_audio, fs = librosa.load(wav_path, sr=16000)


room = pra.ShoeBox([4,6], fs=fs, max_order=12, materials=pra.Material(0.3), ray_tracing=False, air_absorption=False)
# set max_order to a low value for a quick (but less accurate) RIR
# room = pra.Room.from_corners(corners, fs=fs, max_order=3, materials=pra.Material(0.2, 0.15), ray_tracing=True, air_absorption=True)
room.extrude(2.)

# Set the ray tracing parameters
# room.set_ray_tracing(receiver_radius=0.5, n_rays=10000, energy_thres=1e-5)

# add source and set the signal to WAV file content
room.add_source([1., 1., 0.5], signal=speech_audio)

# add two-microphone array
R = np.array([[3.5, 3.6], [2., 2.], [0.5,  0.5]])  # [[x], [y], [z]]
room.add_microphone(R)

# compute image sources
room.image_source_model()
room.simulate()

result_audio = room.mic_array.signals[0,:]
output_path = f"test_signal04.wav"
sf.write(output_path, result_audio, fs, subtype='FLOAT')