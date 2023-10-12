from pathlib import Path
import time

from DA_pyroom5 import DAwithPyroom
from DA_pyroom_utils import gen_output_paths, gen_id

"""
the conda environmet is pyroLate for Linux, pyroomWin for Windows
"""

all_params = { # Set to 0 to have only bk_speakers
                'sp_gain' :     (1.1, True), # Gain of active speaker 
                'sp_num_segments' : (50, False), # Number of samples in the final output
                'bk_num' :      (7, False), # bk_speakers_number
                'length_min' :     (3, False), # Length of output audios (min)
                'sp_init_reduce' : (0.6, False), # Init reduction before inner gain_variation
                'sp_inner_gain_range':([1.2, 1.5], False), # Gain variation inside the sample
                'sp_inner_dur_range' : ([0.2, 0.4], False), # Min & max percentage of gain inside sample

                

                # Set to 0 to remove background speakers
                'bk_gain' :     (1.0, False), # Gain of all bk speakers together 
                'bk_num_segments' : (50, False), # Number of bk samples in the final output
                'bk_gain_range' :     ([0.4, 0.8], True), # Gain range bk speakers 
                'bk_offset_range' :   ([-0.4, 0.4], True), # offset perctange bk speakers
                'bk_init_reduce' : (0.4, False), # Init reduction before inner gain_variation
                'bk_inner_gain_range' : ([2,4, 2,5], False), # Gain variation inside bk sample
                'bk_inner_dur_range' : ([0.2, 0.4], False), # Duration percentage for gain inside bk sample
                'bk_ext_offset_range' : ([1, 4], False), # Offset of extend bk audio (secs)

                # Set to 0 to remove noises
                'ns_gain' :           (1.0, False),
                'ns_gain_range' :     ([0.9, 1.0], False),
                'ns_close_dist' :      (0.4, False), # distance to the center microphone
                'ns_number_samples':   (45, False), # Number of long distance noise samples

                # Store in log folder audios from inner stages
                'debug_store_audio':   (False, False), 
                }
run_name = 'initial_test'

INPUT_NPY_PATH = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'AlterAI_morph_PhuongYeti_English',
                                      'morph_PhuongYeti_Initial.npy')

INPUT_NPY_PATH_NAMES = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'AlterAI_morph_PhuongYeti_English',
                                      'morph_PhuongYeti_Initial_names.npy')

BASE_PATH = INPUT_NPY_PATH.parent 

NOISE_PATH_DICT = {'noise_npy_folder': Path.home().joinpath('Dropbox','DATASETS_AUDIO','VAD_TTS2'),
                   'noise_e1_soft' : 'noise_E1.npy',
                   'noise_e2_loud': 'noise_E1.npy',
                   'noise_e3_distance': 'noise_E3.npy'
                   }

run_id = gen_id(run_name, all_params)
OUTPUT_PATH_DICT = gen_output_paths(BASE_PATH, run_id)

t_start = time.time()
t_pc_start = time.process_time()

# DA Pyroom Object Instance
my_sim = DAwithPyroom(INPUT_NPY_PATH, INPUT_NPY_PATH_NAMES, 
                      NOISE_PATH_DICT, OUTPUT_PATH_DICT,
                      all_params)

# Simulate dataset with single speaker
my_sim.sim_vad_dataset()

print("Total time time : {}".format(time.time() - t_start))
print("Total time process_time : {}".format(time.process_time() - t_pc_start))

# Listen to a sample from the simulation
# sd.play(my_dataset_simulated[1,:], fs)
