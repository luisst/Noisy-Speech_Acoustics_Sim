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
from DA_pyroom import DAwithPyroom




"""
the conda environmet is pyro. Must use gcc version 10 to install
"""

INPUT_NPY_PATH = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'Simulation_Speaker_Geometry',
                                      'Phuong','input_single_speaker.npy')

number_parts = 3

# If noise is required for later
NOISE_PATH_Audioset = Path.home().joinpath('Dropbox','DATASETS_AUDIO',
                                           'Noisy-Speech_Acoustics_Sim_NOISES',
                                           'Noises_all.npy')

BASE_PATH = INPUT_NPY_PATH.parent 

output_folder = BASE_PATH.joinpath('DataAugmented')
proc_log = output_folder.joinpath('process_log.txt')

t_start = time.time()
t_perf_start = time.perf_counter()
t_pc_start = time.process_time()

glb_ite = 0

if not(check_folder_for_process(output_folder)):
    sys.exit('goodbye')

# Read length of npy audios
tmp_array = np.load(INPUT_NPY_PATH, allow_pickle=True)
total_audios_npy = len(tmp_array)
del tmp_array

# Randomly divide array into 3 list of indexes
parts_lengths_list = [total_audios_npy // number_parts + (1 if x < total_audios_npy % number_parts else 0)  for x in range (number_parts)]
my_list_indexes = list(range(0,total_audios_npy))
list_section_indexes = []

for section_number_values in parts_lengths_list:
    section_indexes = []
    for idx in range(0, section_number_values):
        rand_index = random.randint(0, len(my_list_indexes)-1)
        print(f'Size of my_list: {len(my_list_indexes)} \t Index random: {rand_index}')
        section_indexes.append(my_list_indexes.pop(rand_index))
    list_section_indexes.append(section_indexes)


# list_section_indexes = [[0,2,4], [8,7], [12,33]]

for position_idx in range(0,3):
    # DA Pyroom Object Instance
    my_sim = DAwithPyroom(INPUT_NPY_PATH, NOISE_PATH_Audioset, single_cfg, proc_log, 0)

    # Simulate dataset with single speaker
    my_dataset_simulated, glb_ite = my_sim.sim_dataset_single_speaker(list_section_indexes[position_idx], position=position_idx)

    # Define output path of each wav
    if not(output_folder.joinpath(f'position_{position_idx}').exists()):
        output_folder.joinpath(f'position_{position_idx}').mkdir()

    for idx_output in range(0, len(my_dataset_simulated)):
        new_name_wav = f'phuong_clean-position_{position_idx}_' + str(idx_output).zfill(3) + '.wav'
        output_path_wav = output_folder.joinpath(f'position_{position_idx}', new_name_wav) 

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



