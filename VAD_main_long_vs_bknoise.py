import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import time
import sys

from cfg_pyroom import fs, check_folder, all_cfg
from DA_pyroom4 import DAwithPyroom

from utilities_functions import check_folder_for_process


"""
the conda environmet is pyro. Must use gcc version 10 to install
"""


INPUT_NPY_PATH = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'WAV_TTS', 'TTS_10K2023.npy')

audio_folder_pth = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'WAV_TTS', 'All_WAVs')

# If noise is required for later
NOISE_PATH_Audioset = Path.home().joinpath('Dropbox','DATASETS_AUDIO',
                                           'Noisy-Speech_Acoustics_Sim_NOISES',
                                           'Noises_all.npy')

NOISE_PATH_AOLME = Path.home().joinpath('Dropbox','DATASETS_AUDIO',
                                           'Noisy-Speech_Acoustics_Sim_NOISES',
                                           'AOLME_noises.npy')

NOISE_PATH_E1_SOFT = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'WAV_TTS', 'noise_E1.npy')

NOISE_PATH_E2_LOUD = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'WAV_TTS', 'noise_E2.npy')

BASE_PATH = INPUT_NPY_PATH.parent 

output_folder = BASE_PATH.joinpath('VAD_synthetic')
proc_log = output_folder.joinpath('process_log.txt')

t_start = time.time()
t_perf_start = time.perf_counter()
t_pc_start = time.process_time()

if not(check_folder_for_process(output_folder)):
    sys.exit('goodbye')

# list of all audios
list_audio_paths = sorted(list(audio_folder_pth.glob('*.wav')))


# DA Pyroom Object Instance
my_sim = DAwithPyroom(INPUT_NPY_PATH, NOISE_PATH_E1_SOFT, NOISE_PATH_E2_LOUD, 
                      proc_log, noise_flag = True,
                       min_gain = 0.7, max_gain = 1.0,
                    #    min_gain = 0.9, max_gain = 1.0,
                       min_offset = -0.4, max_offset = 0.4,
                       bk_num = 4)

# Simulate dataset with single speaker
my_dataset_simulated = my_sim.sim_vad_dataset()

print(f'Length of sim dataset: {len(my_dataset_simulated)}')

for idx_output in range(0, len(my_dataset_simulated)):
    # Use name from list of all audios

    new_name_wav = f'DA_long-{idx_output}.wav'
    output_path_wav = output_folder.joinpath(new_name_wav) 

    # Save as WAV 
    sf.write(str(output_path_wav), my_dataset_simulated[idx_output], 16000)


t_end = time.time()
t_perf_end = time.perf_counter()
t_pc_end = time.process_time()

print("Total time time : {}".format(t_end - t_start))
print("Total time perf_counter: {}".format(t_perf_end - t_perf_start))
print("Total time process_time : {}".format(t_pc_end - t_pc_start))
# Listen to a sample from the simulation
# sd.play(my_dataset_simulated[1,:], fs)
