from pathlib import Path
import time

from DA_pyroom8 import DAwithPyroom
from DA_pyroom_utils import gen_output_paths, gen_id

"""
the conda environmet is pyroLate for Linux, pyroomWin for Windows
"""

all_params = { # Set to 0 to have only bk_speakers
                'sp_gain' :     (0.8, True), # Gain of active speaker 
                'rnd_offset_secs' : (1, False), # max silence between speech segments 
                'bk_num' :      (6, True), # bk_speakers_number
                'output_samples_num' : (1, False), # Number of long audios generated
                'length_minutes' :     (3, False), # Length of output audios (min)
                'pattern_flag': (False, False), # Use pattern for conversation
                'double_talk_flag': (False, False), # Use double talk simulation
                'ordered_sp_flag': (True, False), # Use ordered speech segments

                'gain_var_flag' : (True, False), # Apply gain_variation 
                'sp_init_reduce' : (0.7, False), # Init reduction before inner gain_variation
                'sp_inner_gain_range':([1.2, 1.3], False), # Gain variation inside the sample
                'sp_inner_dur_range' : ([0.4, 0.6], False), # Min & max percentage of gain inside sample

                # Set to 0 to remove background speakers
                'bk_gain' :     (0.5, True), # Gain of all bk speakers together 
                'bk_num_segments' : (40, True), # Number of bk samples in the final output
                'bk_gain_range' :     ([0.3, 0.4], True), # Gain range bk speakers 
                'bk_init_reduce' : (0.5, False), # Init reduction before inner gain_variation
                'bk_inner_gain_range' : ([2,4, 2,5], False), # Gain variation inside bk sample
                'bk_inner_dur_range' : ([0.2, 0.4], False), # Duration percentage for gain inside bk sample
                'bk_ext_offset_range' : ([1, 4], False), # Offset of extend bk audio (secs)

                # Set to 0 to remove noises
                'ns_gain' :           (0.5, True), 
                'ns_gain_away' :      (0.5, False),
                'ns_gain_range' :     ([0.3, 0.4], True),
                'ns_number_samples':   (50, False), # Number of long distance noise samples

                # Store in log folder audios from inner stages
                'debug_store_audio':   (True, False) 
                }
run_name = 'g5-easy2'


FOLDER_WAV_BASE = Path.home().joinpath('Dropbox','DATASETS_AUDIO')
# INPUT_WAV_PATH = FOLDER_WAV_BASE / "TTS4_dvec" / "TTS4_input_wavs_balanced"
INPUT_WAV_PATH = FOLDER_WAV_BASE / "TTS4_dvec" / "TTS4_input_wavs"
# ORDERED_SPEECH_TXT = INPUT_WAV_PATH.joinpath('balanced_dataset_min100_bal200_shuffled.txt')
ORDERED_SPEECH_TXT = INPUT_WAV_PATH.joinpath('balanced_dataset_min40_bal300_shuffled.txt')

BASE_PATH = INPUT_WAV_PATH.parent 

NOISE_PATH_DICT = {'noise_npy_folder': Path.home().joinpath('Dropbox','DATASETS_AUDIO','WAV_TTS'),
                   'noise_e1_soft' : 'E1_noises.npy',
                   'noise_e2_loud': 'E2_noises.npy',
                   'noise_e3_distance': 'E3_noises_away.npy'
                   }

run_id = gen_id(run_name, all_params)
OUTPUT_PATH_DICT = gen_output_paths(BASE_PATH, run_id)

t_start = time.time()
t_pc_start = time.process_time()

# DA Pyroom Object Instance
my_sim = DAwithPyroom(INPUT_WAV_PATH, ORDERED_SPEECH_TXT, 
                      NOISE_PATH_DICT, OUTPUT_PATH_DICT,
                      all_params)

# Simulate dataset with single speaker
my_sim.sim_vad_dataset()

print("Total time time : {}".format(time.time() - t_start))
print("Total time process_time : {}".format(time.process_time() - t_pc_start))

# Listen to a sample from the simulation
# sd.play(my_dataset_simulated[1,:], fs)
