from pathlib import Path
import time

from DA_pyroom_far_end import DAwithPyroom
from DA_pyroom_utils import gen_output_paths, gen_id

"""
the conda environmet is pyroLate for Linux, pyroomWin for Windows
"""
"""
Cases:
1. echo only -> main speaker is 0
2. echo + near end -> echo + the main speaker talking
3. noise + near end -> main speaker with noise, bk_sp = 0
4. noise + echo + near end -> all of them together


Outputs:

[1, 2, 4]  --> pyroom output
- bk_speaker (clean)
- main speaker (clean)

[2] --> pyroom output
- main speaker (clean)

noise in this case doesn't matter
"""
all_params = { # Set to 0 to have only bk_speakers
                'case_audio':   (4, True), # 1: See cases in text file
                'rnd_offset_secs' : (2, False), # max silence between speech segments 
                'bk_num' :      (1, False), # bk_speakers_number
                'output_samples_num' : (10, False), # Number of long audios generated
                'length_min' :     (1, False), # Length of output audios (min)

                'sp_init_reduce' : (0.7, False), # Init reduction before inner gain_variation
                'sp_inner_gain_range':([1.2, 1.5], False), # Gain variation inside the sample
                'sp_inner_dur_range' : ([0.2, 0.4], False), # Min & max percentage of gain inside sample

                # Set to 0 to remove background speakers
                'bk_num_segments' : (12, False), # Number of bk samples in the final output
                'bk_gain_range' :     ([0.7, 0.9], False), # Gain range bk speakers 
                'bk_offset_range' :   ([-0.1, 0.1], False), # offset perctange bk speakers
                'bk_init_reduce' : (0.5, False), # Init reduction before inner gain_variation
                'bk_inner_gain_range' : ([2,4, 2,5], False), # Gain variation inside bk sample
                'bk_inner_dur_range' : ([0.2, 0.4], False), # Duration percentage for gain inside bk sample
                'bk_ext_offset_range' : ([1, 2], False), # Offset of extend bk audio (secs)

                # Set to 0 to remove noises
                'ns_gain_range' :     ([0.8, 1.0], False),
                'ns_close_dist' :      (0.4, False), # Distance ns source to mic
                'ns_number_samples':   (30, False), # Number of long distance noise samples
                }
run_name = 'Synth_delayed'

INPUT_WAV_PATH = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 
                                      'AlterAI_morph_PhuongYeti_English',
                                      'WAV_audios_phrases')

BASE_PATH = INPUT_WAV_PATH.parent 

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
my_sim = DAwithPyroom(INPUT_WAV_PATH, 
                      NOISE_PATH_DICT, OUTPUT_PATH_DICT,
                      all_params)

# Simulate dataset with single speaker
my_sim.sim_vad_dataset()

print("Total time time : {}".format(time.time() - t_start))
print("Total time process_time : {}".format(time.process_time() - t_pc_start))

# Listen to a sample from the simulation
# sd.play(my_dataset_simulated[1,:], fs)
