import numpy as np
import sounddevice as sd
import soundfile as sf
import pickle
import time
import random
import sys
from pathlib import Path
from utilities_functions import check_folder_for_process, ffmpeg_split_audio

from cfg_pyroom_SD import fs, single_cfg
from DA_pyroom2 import DAwithPyroom2




"""
the conda environmet is pyro. Must use gcc version 10 to install
"""

audio_folder_pth = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'Simulation_Speaker_Geometry',
                                      'Phuong','wav_output')

# If noise is required for later
NOISE_PATH_Audioset = Path.home().joinpath('Dropbox','DATASETS_AUDIO',
                                           'Noisy-Speech_Acoustics_Sim_NOISES',
                                           'Noises_all.npy')

NOISE_PATH_AOLME = Path.home().joinpath('Dropbox','DATASETS_AUDIO',
                                           'Noisy-Speech_Acoustics_Sim_NOISES',
                                           'AOLME_noises.npy')

BASE_PATH = audio_folder_pth.parent 

output_folder = BASE_PATH.joinpath('DataAugmented')
proc_log = output_folder.joinpath('process_log.txt')

t_start = time.time()
t_perf_start = time.perf_counter()
t_pc_start = time.process_time()

glb_ite = 0

if not(check_folder_for_process(output_folder)):
    sys.exit('goodbye')

# list of all audios
list_audio_paths = sorted(list(audio_folder_pth.glob('*.wav')))


# DA Pyroom Object Instance
my_sim = DAwithPyroom2(NOISE_PATH_AOLME, single_cfg, proc_log, 0)

# Simulate dataset with single speaker
my_dataset_simulated, glb_ite = my_sim.sim_interviews_dataset(list_audio_paths, 
                                                    noise_flag=True)


for idx_output in range(0, len(my_dataset_simulated)):
    # Use name from list of all audios

    new_name_wav = f'DA-{list_audio_paths[idx_output].name}'
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



