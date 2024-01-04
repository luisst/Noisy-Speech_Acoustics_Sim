import random
import numpy as np
import inspect

import scipy.signal as signal
import csv
import shutil
import sys
from cfg_pyroom_VAD import sr
from utilities_functions import check_folder_for_process
from pedalboard import Pedalboard, Reverb
import soundfile as sf


# Define the dictionary mapping substrings to values
name_mapping_TTS2 = {
    'James': 'S0',
    'Morgan': 'S1',
    'Jennifer': 'S2',
    'Sofia': 'S3',
    'Edward': 'S4',
    'Keira': 'S5'
}


def log_message(msg, log_file, mode, both=True):
    '''Function that prints and/or adds to log'''
    #Always log file
    with open(log_file, mode) as f:
        f.write(msg)

    #If {both} is true, print to terminal as well
    if both:
        print(msg, end='')

def sum_arrays(arrays):
    """
    Given a list of NumPy arrays, return a new NumPy array that is the element-to-element sum of all the arrays.
    """
    result = np.zeros(arrays[0].shape)  # Initialize the result with zeros
    for array in arrays:
        result += array
    return result

def gen_random_on_range(lower_value, max_value):
    """
    generates a random value between lower_value and max_value.
    """
    return round(lower_value + random.random()*(max_value - lower_value),
                    2)


def eliminate_noise_start_ending(signal, th):
    """
    Count using a non-optimized python alg the number of zeros
    at the end of the numpy array
    """

    # real_length is initialized with the total length of the signal
    real_length_end = int(len(signal))
    while abs(signal[real_length_end - 1]) < th:
        real_length_end = real_length_end - 1

    signal_trimmed_end = signal[0:real_length_end]
    
    real_length_start = 0
    while abs(signal_trimmed_end[real_length_start]) < th:
        real_length_start = real_length_start + 1
    
    
    signal_trimmed = signal_trimmed_end[real_length_start:]
    
    return signal_trimmed

def norm_noise_f32(noise_current_audio, gain_value = 1):
    """
    Apply gain independent of the max/min values of the input audio
    """

    if noise_current_audio.dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    vmin = noise_current_audio.min()
    vmax = noise_current_audio.max()

    audio_gained = noise_current_audio*gain_value

    vmin_gained = audio_gained.min()
    vmax_gained = audio_gained.max()

    nm_sum = np.sum(noise_current_audio)
    if np.isnan(nm_sum):
        try_inspect = inspect.stack()
        print(try_inspect[2].function)
        msg = f">> NAN in audio_float32\nVin: {vmin} | {vmax} --- Vout: {vmin_gained} | {vmax_gained}\
            function {try_inspect[2].function} | code_context {try_inspect[2].code_context} \
                || function {try_inspect[1].function} | context{try_inspect[1].code_context}\n"
        # log_message(msg, proc_log, 'a', True)
    
    nm_sum = np.sum(audio_gained)
    if np.isnan(nm_sum):
        try_inspect = inspect.stack()
        msg = f">> NAN in audio_gained\nVin: {vmin} | {vmax} --- Vout: {vmin_gained} | {vmax_gained}\
            function {try_inspect[2].function} | code_context {try_inspect[2].code_context} \
                || function {try_inspect[1].function} | context{try_inspect[1].code_context}\n"
        # log_message(msg, proc_log, 'a', True)

    return audio_gained


def norm_others_float32(audio_float32, gain_value, outmin, outmax):
    """
    Normalize float32 audio with the gain value provided
    """
    if audio_float32.dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    vmin = audio_float32.min()
    vmax = audio_float32.max()

    audio_gained = (outmax - outmin)*(audio_float32 - vmin)/(vmax - vmin) \
        + outmin

    audio_gained = audio_gained*gain_value

    vmin_gained = audio_gained.min()
    vmax_gained = audio_gained.max()

    nm_sum = np.sum(audio_float32)
    if np.isnan(nm_sum):
        try_inspect = inspect.stack()
        print(try_inspect[2].function)
        msg = f">> NAN in audio_float32\nVin: {vmin} | {vmax} --- Vout: {vmin_gained} | {vmax_gained}\
            function {try_inspect[2].function} | code_context {try_inspect[2].code_context} \
                || function {try_inspect[1].function} | context{try_inspect[1].code_context}\n"
        # log_message(msg, proc_log, 'a', True)
    
    nm_sum = np.sum(audio_gained)
    if np.isnan(nm_sum):
        try_inspect = inspect.stack()
        msg = f">> NAN in audio_gained\nVin: {vmin} | {vmax} --- Vout: {vmin_gained} | {vmax_gained}\
            function {try_inspect[2].function} | code_context {try_inspect[2].code_context} \
                || function {try_inspect[1].function} | context{try_inspect[1].code_context}\n"
        # log_message(msg, proc_log, 'a', True)



    return audio_gained


def gain_variation(original_audio, init_reduce = 0.6, 
                   factor_min=2.0, factor_max=2.5, 
                   min_duration_ratio = 0.2, max_duration_ratio = 0.4, 
                   verbose=False):
    # Reduce the gain to allow later increases
    audio_data = init_reduce * original_audio

    # Define the minimum and maximum duration of the gain increase
    min_duration = int(len(audio_data) * min_duration_ratio)
    max_duration = int(len(audio_data) * max_duration_ratio)

    # Generate a random duration for the gain increase
    duration_1 = np.random.randint(min_duration, max_duration)

    # Generate a random starting point for the gain increase
    start_1 = np.random.randint(0, len(audio_data) - duration_1)

    # Generate a random gain increase factor
    factor_1 = np.random.uniform(factor_min, factor_max)

    # Apply the gain increase
    audio_data[start_1:start_1+duration_1] = factor_1 * audio_data[start_1:start_1+duration_1]

    if verbose:
        print(f'start {start_1} ({duration_1}): {factor_1}')
    
    return audio_data


def extend_audio(audio, start_index, end_index, 
                 total_ext_length, 
                 verbose = False):

    extended_audio = np.zeros((int(total_ext_length),))
    
    if verbose:
        print(f'Exclusive portion: \nstart: {start_index} ({round(start_index/sr, 2)}) - end: {end_index} ({round(end_index/sr, 2)})\n')


    # Place the original audio at the exclusive portion
    final_audio = np.concatenate((extended_audio[:start_index], audio, extended_audio[end_index:]))
    
    return final_audio, (start_index, end_index)

def extend_audio_bk(audio, limit_lower, limit_upper, length_min = 3, offset_samples = 0, verbose = False):

    # Calculate how much silence we need to add to make the audio 8 minutes long
    ext_length = sr * (length_min * 60) + offset_samples
    extended_audio = np.zeros((int(ext_length),))

    # Calculate the range of indices where the original audio can be placed
    segment_start = limit_lower 
    segment_end = limit_upper - len(audio)

    # Calculate the start and end indices of the exclusive portion for this audio file
    start_index = int(segment_start + (segment_end - segment_start) * random.random())
    end_index = start_index + len(audio)

    if start_index < 0:
        print(f'Warning! BK|Negative start_index. Audio length: {len(audio)} vs {limit_upper - limit_lower}')
    if verbose:
        print(f'BK|start: {start_index} ({round(start_index/sr, 2)}) - end: {end_index} ({round(end_index/sr, 2)})')

    # Place the original audio at the exclusive portion
    final_audio = np.concatenate((extended_audio[:start_index], audio, extended_audio[end_index:]))
    
    return final_audio, (start_index, end_index)


def remove_dc_component(input_signal):
    nyquist_freq = 0.5 * sr
    cutoff_freq = 60.0 / nyquist_freq
    b, a = signal.butter(1, cutoff_freq, 'highpass', analog=False, output='ba')
    filtered_audio = signal.filtfilt(b, a, input_signal)
    return filtered_audio


def gen_random_gaussian(min_val, max_val, max_attempts=10000):
    mean = (min_val + max_val) / 2
    std_dev = (max_val - min_val) / 4
    for _ in range(max_attempts):
        value = np.random.normal(mean, std_dev)
        if min_val <= value <= max_val:
            return value
    raise ValueError("Failed to generate a valid value within the specified range after 10,000 attempts")


def generate_csv_file(GT_log, output_csv_path, indx,
                      names_mapping_dict,
                      single_name = 'DA_long',
                      only_speaker = True):
    GT_log = sorted(GT_log, key = lambda x: x[1])
    speaker_name = ''

    # specify filename for CSV file
    filename = output_csv_path.joinpath(f'{single_name}_{indx}.csv')
    # open file for writing with tab delimiter
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        # write each tuple to a row in the file
        for audio_name, start, end in GT_log:
            first_column_parts = audio_name.split('_')
            if len(first_column_parts) >= 2:
                substring = first_column_parts[-2]

                # Lookup the substring in the dictionary and replace it with the corresponding value
                if substring in names_mapping_dict:
                    speaker_name = names_mapping_dict[substring]
                else:
                    sys.exit(f"Substring '{substring}' not found in the dictionary.")
            else:
                sys.exit(f"Could not find the name of speaker in: {audio_name}")

            writer.writerow([speaker_name, 'Eng', round(start/sr,2), round(end/sr,2)])


def gen_output_paths(BASE_PATH, run_id):

    OUTPUT_FOLDER_PATH = BASE_PATH.joinpath(run_id)
    OUTPUT_WAV_PATH = OUTPUT_FOLDER_PATH.joinpath('WAV_OUTPUT')
    OUTPUT_LOG_PATH = OUTPUT_FOLDER_PATH.joinpath(f'{run_id}_log.txt')
    OUTPUT_CSV_PATH = OUTPUT_FOLDER_PATH.joinpath('GT_CSV')

    # if not OUTPUT_WAV_PATH.exists():
    #     OUTPUT_WAV_PATH.mkdir()
    # else:
    #     shutil.rmtree(OUTPUT_WAV_PATH)
    #     OUTPUT_WAV_PATH.mkdir()   

    if not(check_folder_for_process(OUTPUT_WAV_PATH)):
        sys.exit('goodbye')

    if not OUTPUT_CSV_PATH.exists():
        OUTPUT_CSV_PATH.mkdir()
    else:
        shutil.rmtree(OUTPUT_CSV_PATH)
        OUTPUT_CSV_PATH.mkdir()


    OUTPUT_PATH_DICT = {'output_wav_path' : OUTPUT_WAV_PATH,
                        'output_log_path' : OUTPUT_LOG_PATH,
                        'output_csv_path' : OUTPUT_CSV_PATH}

    return OUTPUT_PATH_DICT


def gen_id(run_name, dict_param):
    
    # Verify all values are a tuple 
    for key, value in dict_param.items():
        if not(isinstance(value, tuple)) or not(len(value) > 0):
            sys.exit('Error: Dictionary values must be tuples')

    run_id = run_name

    # Select the values with True in value[1]
    for key, value in dict_param.items():
        if value[1]:
            # Verify value is a scalar
            if isinstance(value[0], (float, int)):
                run_id = run_id + f'-{key}{value[0]}'

            # or Verify value is a range
            elif len(value[0]) == 2:
                run_id = run_id + f'-{key}{value[0][0]}~{value[0][1]}'
            
            # else give error
            else:
                sys.exit(f'ERROR: key {key} have unsupported length {len(value[0])}')
    return run_id

def convert_param_dict(all_dict):
    result = {}
    
    for key, value in all_dict.items():
        if isinstance(value, tuple) and len(value) > 0:
            result[key] = value[0]
        else:
            print(f'ERROR: dictionary does not have True/False values')
    
    return result


def read_audio_name(list_audio_paths, indx):
    current_audio_path = list_audio_paths[indx]
    raw_data, samplerate = sf.read(current_audio_path)
    if samplerate != sr:
        sys.exit(f'ERROR! Audio is not 16K: {current_audio_path.name}')

    raw_audio = np.asarray(raw_data)
    current_audio_name = current_audio_path.stem 

    return raw_audio, current_audio_name


def read_audio_name_from_dict(dict_speakers_paths, selected_speaker):

    list_current_speaker_paths = dict_speakers_paths[selected_speaker]

    indx_others = random.randint(0, len(list_current_speaker_paths)-1)
    current_audio_path = list_current_speaker_paths[indx_others]
    raw_data, samplerate = sf.read(current_audio_path)
    if samplerate != sr:
        sys.exit(f'ERROR! Audio is not 16K: {current_audio_path.name}')

    raw_audio = np.asarray(raw_data)
    current_audio_name = current_audio_path.stem 

    return raw_audio, indx_others, current_audio_name

def apply_reverb(input_wav, reverb_vals=(0.1, 1.6)):

    if reverb_vals[0] != 0:
        pedalboard = Pedalboard([
                Reverb(
                    room_size=reverb_vals[0],
                )
            ])

        single_noise = np.asarray(input_wav)

        single_noise = single_noise * reverb_vals[1] 

        # Apply the effect to the audio
        return pedalboard(single_noise, sample_rate=sr)
    else:
        return input_wav

def amplify_audio_to_0db(audio):
    # Find the maximum absolute value in the audio array
    max_amplitude = np.max(np.abs(audio))

    # Check if the maximum amplitude is zero to avoid division by zero
    if max_amplitude == 0:
        return audio

    # Calculate the scaling factor to reach 0 dB
    scaling_factor = 1.0 / max_amplitude

    # Amplify the audio by scaling all values
    amplified_audio = audio * scaling_factor

    return amplified_audio