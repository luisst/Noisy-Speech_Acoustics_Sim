import random
import numpy as np

import scipy.signal as signal
import csv
import shutil
import sys
from utilities_functions import check_folder_for_process
import soundfile as sf
import os

# from pydub import AudioSegment
# from pydub.effects import compress_dynamic_range
# import pydub

sr = 16000  # Sample rate for audio processing

# Define the dictionary mapping substrings to values
name_mapping_TTS2 = {
    'James': 'S0',
    'Morgan': 'S1',
    'Jennifer': 'S2',
    'Sofia': 'S3',
    'Edward': 'S4',
    'Keira': 'S5'
}


def create_random_sequence2(length=100):
    """
    Generate a list of booleans representing speech (True) and silence (False)
    in an educational conversation setting.
    
    :param length: Length of the list to generate
    :return: List of booleans with a structured pattern
    """
    pattern = []
    i = 0
    
    while i < length:
        choice = random.choices(
            ['speech_segment', 'alternating', 'silence_period'], 
            weights=[0.4, 0.3, 0.3]
        )[0]
        
        if choice == 'speech_segment':
            segment_length = random.randint(2, 5)
            pattern.extend([0] * segment_length)
            i += segment_length
        
        elif choice == 'alternating':
            segment_length = random.randint(4, 10)
            for _ in range(segment_length):
                pattern.append(random.choice([0, 1]))
            i += segment_length
        
        elif choice == 'silence_period':
            silence_length = random.choices([
                2, 3, 6, 10  # Longer silences are rarer
            ], weights=[0.5, 0.3, 0.15, 0.05])[0]
            pattern.extend([1] * silence_length)
            i += silence_length
    
    return pattern[:length]  # Trim in case we exceed length


def create_random_sequence(length, flip_probability=0.3):
    """
    Create a random sequence of 0s and 1s where the tendency is to have 
    consecutive numbers the same.

    Parameters:
    length (int): Length of the sequence.
    flip_probability (float): Probability of changing the current number 
                              from the previous one. Default is 0.2.
    
    Returns:
    list: Random sequence of 0s and 1s.
    """
    if length <= 0:
        return []

    # Initialize the sequence with all zeros
    sequence = [0] * length

    for i in range(1, length):
        if random.random() < flip_probability:
            # Flip the value with a certain probability
            sequence[i] = 1 - sequence[i-1]
        else:
            # Keep the same value as the previous element
            sequence[i] = sequence[i-1]

    return sequence


def log_message(msg, log_file, mode, both=True):
    '''Function that prints and/or adds to log'''
    #Always log file
    # Ensure the directory exists
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
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


def norm_others_float32(audio_float32, gain_value):
    """
    Normalize float32 audio with the gain value provided
    """
    if audio_float32.dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    vmin = audio_float32.min()
    vmax = audio_float32.max()

    # Normalize the audio to the range [-1, 1]
    audio_float32 = ((audio_float32 - vmin) / (vmax - vmin)) * 2 - 1

    audio_gained = audio_float32*gain_value

    # nm_sum = np.sum(audio_float32)
    # if np.isnan(nm_sum):
    #     try_inspect = inspect.stack()
    #     print(try_inspect[2].function)
    #     msg = f">> NAN in audio_float32\nVin: {vmin} | {vmax} --- Vout: {vmin_gained} | {vmax_gained}\
    #         function {try_inspect[2].function} | code_context {try_inspect[2].code_context} \
    #             || function {try_inspect[1].function} | context{try_inspect[1].code_context}\n"
        # log_message(msg, proc_log, 'a', True)
    
    # nm_sum = np.sum(audio_gained)
    # if np.isnan(nm_sum):
    #     try_inspect = inspect.stack()
    #     msg = f">> NAN in audio_gained\nVin: {vmin} | {vmax} --- Vout: {vmin_gained} | {vmax_gained}\
    #         function {try_inspect[2].function} | code_context {try_inspect[2].code_context} \
    #             || function {try_inspect[1].function} | context{try_inspect[1].code_context}\n"
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

def extend_audio_bk(audio, limit_lower, limit_upper, total_length, offset_samples, verbose = False):

    # Calculate how much silence we need to add to make the audio 8 minutes long
    ext_length_plus_offset = total_length + offset_samples
    extended_audio = np.zeros((int(ext_length_plus_offset),))

    # Calculate the range of indices where the original audio can be placed
    segment_start = limit_lower 
    segment_end = limit_upper - len(audio)

    # Calculate the start and end indices of the exclusive portion for this audio file
    start_offset = max(0, (segment_end - segment_start) * random.random()) 

    start_index = int(segment_start + start_offset)
    end_index = min(start_index + len(audio), int(limit_upper))

    if start_index < 0:
        print(f'Warning! BK|Negative start_index. Audio length: {len(audio)} vs {limit_upper - limit_lower}')
        print(f'    Details: segment start: {segment_start} - segment end: {segment_end} - start index: {start_index} - end index: {end_index}')
    if verbose:
        print(f'BK|start: {start_index} ({round(start_index/sr, 2)}) - end: {end_index} ({round(end_index/sr, 2)})')

    # Check if slice indeces are integers
    if not isinstance(start_index, int) or not isinstance(end_index, int):
        raise ValueError("Slice indices must be integers")

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



def generate_csv_file_tts3(GT_log, output_csv_path, indx,
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
            name_column_parts = audio_name.split('_')
            if len(name_column_parts) == 4:
                current_substring = name_column_parts[-3]
            elif len(name_column_parts) == 3:
                current_substring = name_column_parts[-2]
            else:
                sys.exit(f"Wrong filename format for WAVs: {audio_name}")
            
            ## Verify the current_substring starts with 'ID-' and has 2 digits
            if current_substring[:3] != 'ID-' or len(current_substring) != 5:
                sys.exit(f"Wrong filename format for WAVs, no ID-/d/d: {audio_name}")

            speaker_name = current_substring[-2:]

            writer.writerow([speaker_name, 'TBD', round(start/sr,2), round(end/sr,2), audio_name])


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
    current_audio_name = 'Placeholder'
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


def amplify_audio_to_3_2db(audio):
    # Find the maximum absolute value in the audio array
    max_amplitude = np.max(np.abs(audio))

    # Check if the maximum amplitude is zero to avoid division by zero
    if max_amplitude == 0:
        return audio

    # Calculate the scaling factor to reach 3.2 dB
    scaling_factor = 10 ** (3.2 / 20) / max_amplitude

    # Amplify the audio by scaling all values
    amplified_audio = audio * scaling_factor

    return amplified_audio

def normalize_to_minus1db(input_audio):
    # Convert -1 dB to a linear scale factor
    target_dB = -1
    scale_factor = 10 ** (target_dB / 20)

    # Calculate the peak normalization factor and adjust it to -1 dB
    peak_normalization_factor = scale_factor / np.max(np.abs(input_audio))
    normalized_audio = input_audio * peak_normalization_factor
    return normalized_audio


def from_numpy_array(nparr, framerate):
    """
    Returns an AudioSegment created from the given numpy array.

    The numpy array must have shape = (num_samples, num_channels).

    :param nparr: The numpy array to create an AudioSegment from.
    :param framerate: The sample rate (Hz) of the segment to generate.
    :returns: An AudioSegment created from the given array.
    """
    # Check args
    if nparr.dtype.itemsize not in (1, 2, 4):
        raise ValueError("Numpy Array must contain 8, 16, or 32 bit values.")

    # Determine nchannels
    if len(nparr.shape) == 1:
        nchannels = 1
    elif len(nparr.shape) == 2:
        nchannels = nparr.shape[1]
    else:
        raise ValueError("Numpy Array must be one or two dimensional. Shape must be: (num_samples, num_channels), but is {}.".format(nparr.shape))

    # Fix shape if single dimensional
    nparr = np.reshape(nparr, (-1, nchannels))

    # Create an array of mono audio segments
    m = nparr[:, 0]
    dubseg = pydub.AudioSegment(m.tobytes(), frame_rate=framerate, sample_width=nparr.dtype.itemsize, channels=1)

    return dubseg

def gen_labels_list(list_audio_paths):
    """
    Generate speaker labels for a list of audio paths.
    """
    list_spk_labels = []
    for audio_path in list_audio_paths:
        # Extract speaker label from the audio file name
        speaker_label_substring = audio_path.stem.split('_')[1]

        # Verify the label has the format 'ID-XX' where XX is a two-digit number
        if not (speaker_label_substring.startswith('ID-') and len(speaker_label_substring) == 5):
            sys.exit(f"Error: Invalid speaker label format in {audio_path.name}. Expected format 'ID-XX'.")
        
        # Extract the two-digit number from the label
        speaker_label = speaker_label_substring[3:]  # Get the last two characters after 'ID-'

        list_spk_labels.append(speaker_label)
    return list_spk_labels

def extract_main_speakers(list_audio_paths, selected_speakers):
    """
    Extract paths of main speakers based on selected speaker labels.
    
    :param list_audio_paths: List of all audio file paths.
    :param selected_speakers: List of selected speaker labels.
    :return: List of paths for the main speakers.
    """
    list_main_spk_paths = []
    
    for audio_path in list_audio_paths:
        # Extract speaker label from the audio file name
        speaker_label_substring = audio_path.stem.split('_')[1]
        
        # Verify the label has the format 'ID-XX' where XX is a two-digit number
        if not (speaker_label_substring.startswith('ID-') and len(speaker_label_substring) == 5):
            sys.exit(f"Error: Invalid speaker label format in {audio_path.name}. Expected format 'ID-XX'.")
        
        # Extract the two-digit number from the label
        speaker_label = speaker_label_substring[3:]  # Get the last two characters after 'ID-'
        
        if speaker_label in selected_speakers:
            list_main_spk_paths.append(audio_path)
    
    return list_main_spk_paths

# def compress_numpy_audio(input_audio):
    
#     audio_int32 = np.int32(input_audio * 2147483647)

#     audio_segment = from_numpy_array(audio_int32, sr)

#     # Apply a dynamic range compression
#     compressed_segment = compress_dynamic_range(audio_segment,
#                                                 threshold=-20.0,
#                                                 ratio=10.0,
#                                                 attack=200.0,
#                                                 release=1000.0)

#     compressed_numpy_int32 = compressed_segment.get_array_of_samples()
#     compressed_numpy_int32 = np.array(compressed_numpy_int32)

#     # Normalize and convert to float32 format
#     compressed_numpy_float32 = (compressed_numpy_int32 / 2147483647).astype(np.float32)

#     return compressed_numpy_float32
