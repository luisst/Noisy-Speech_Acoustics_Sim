import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import time
import shutil
import sys

from cfg_pyroom import fs, check_folder, all_cfg
from DA_pyroom4 import DAwithPyroom

from utilities_functions import check_folder_for_process


"""
the conda environmet is pyro. Must use gcc version 10 to install
"""

dataset_name = 'morph_PhuongYeti_Initial'

INPUT_NPY_PATH = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'AlterAI_morph_PhuongYeti_English', f'{dataset_name}.npy')

INPUT_NPY_PATH_NAMES = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'AlterAI_morph_PhuongYeti_English', f'{dataset_name}_names.npy')

audio_folder_pth = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'AlterAI_morph_PhuongYeti_English', 'morph_PhuongYeti_Initial.npy')

# If noise is required for later
NOISE_PATH_Audioset = Path.home().joinpath('Dropbox','DATASETS_AUDIO',
                                           'Noisy-Speech_Acoustics_Sim_NOISES',
                                           'Noises_all.npy')

NOISE_PATH_AOLME = Path.home().joinpath('Dropbox','DATASETS_AUDIO',
                                           'Noisy-Speech_Acoustics_Sim_NOISES',
                                           'AOLME_noises.npy')

NOISE_PATH_E1_SOFT = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'VAD_TTS2', 'noise_E1.npy')

NOISE_PATH_E2_LOUD = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'VAD_TTS2', 'noise_E2.npy')


NOISE_PATH_E3_DISTANCE = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'VAD_TTS2', 'noise_E3.npy')

BASE_PATH = INPUT_NPY_PATH.parent 

output_folder = BASE_PATH.joinpath('VAD_synthetic')
proc_log = output_folder.joinpath('process_log.txt')
OUTPUT_CSV_PATH = output_folder.joinpath('GT_logs')

t_start = time.time()
t_perf_start = time.perf_counter()
t_pc_start = time.process_time()

if not(check_folder_for_process(output_folder)):
    sys.exit('goodbye')

if not OUTPUT_CSV_PATH.exists():
    OUTPUT_CSV_PATH.mkdir()
else:
    shutil.rmtree(OUTPUT_CSV_PATH)
    OUTPUT_CSV_PATH.mkdir()

# list of all audios
list_audio_paths = sorted(list(audio_folder_pth.glob('*.wav')))


# DA Pyroom Object Instance
my_sim = DAwithPyroom(INPUT_NPY_PATH, INPUT_NPY_PATH_NAMES, NOISE_PATH_E1_SOFT, NOISE_PATH_E2_LOUD, 
                        NOISE_PATH_E3_DISTANCE, OUTPUT_CSV_PATH, output_folder,
                      proc_log, noise_flag = True, bk_noise_flag = False,
                       min_gain = 0.4, max_gain = 0.8,
                    #    min_gain = 0.9, max_gain = 1.0,
                       min_offset = -0.4, max_offset = 0.4,
                       bk_num = 7)

# Simulate dataset with single speaker
my_sim.sim_vad_dataset()

t_end = time.time()
t_perf_end = time.perf_counter()
t_pc_end = time.process_time()

print("Total time time : {}".format(t_end - t_start))
print("Total time perf_counter: {}".format(t_perf_end - t_perf_start))
print("Total time process_time : {}".format(t_pc_end - t_pc_start))
# Listen to a sample from the simulation
# sd.play(my_dataset_simulated[1,:], fs)
