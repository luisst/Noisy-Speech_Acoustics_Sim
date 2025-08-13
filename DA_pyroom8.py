#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 10:48:43 2021

Latest version 5: October 10th, 2023

@author: Luis Sanchez Tapia
"""

import random
import numpy as np
import pyroomacoustics as pra
import time
import sys
import os
import soundfile as sf
from pathlib import Path
from collections import Counter


from cfg_pyroom_VAD import gen_room_config, gen_rand_coordinates, \
    gen_rand_table_coords

from plot_configuration_room import plot_single_configuration

from DA_pyroom_utils import norm_others_float32, log_message, gain_variation, \
    gen_random_on_range, gen_random_on_range, gen_random_on_range, \
    sum_arrays, eliminate_noise_start_ending, extend_audio, convert_param_dict, \
    generate_csv_file_tts3, remove_dc_component, gen_random_gaussian, extend_audio_bk, \
    read_audio_name, amplify_audio_to_3_2db, amplify_audio_to_0db, \
    create_random_sequence, create_random_sequence2, gen_labels_list, extract_main_speakers


class DAwithPyroom(object):
    """
    Class for audio simulation using pyroom.
    input signal + 4 random crosstalk + background noise
    """

    def __init__(self, input_wav_path, ordered_speech_txt_path,
                 noise_dict, output_dict, 
                 all_params):
        """
        Initialize all the params from the dictionaries. Examples provided in the Main_long.py file
        """

        dict_params = convert_param_dict(all_params)

        # Double talk settings
        self.double_talk_flag = dict_params['double_talk_flag']
        self.pattern_flag = dict_params['pattern_flag']

        self.ordered_sp_flag = dict_params['ordered_sp_flag']

        self.audio_name = 'D1'
        self.audio_name2 = 'D2'

        noise_path1 = noise_dict['noise_npy_folder'].joinpath(noise_dict['noise_e1_soft'])
        noise_path2 = noise_dict['noise_npy_folder'].joinpath(noise_dict['noise_e2_loud'])
        noise_path3 = noise_dict['noise_npy_folder'].joinpath(noise_dict['noise_e3_distance'])

        self.noiseE1_data = np.load(noise_path1, allow_pickle=True) #
        self.noiseE2_data = np.load(noise_path2, allow_pickle=True) #
        self.data_E3 = np.load(noise_path3, allow_pickle=True) #
        self.ai_noise_directory = Path.home().joinpath('Dropbox','DATASETS_AUDIO','TTS3_dt','TTS3_noises_all') 

        self.max_iter = 0
        self.complete_flag = False
        self.list_audio_paths = []
        self.list_spk_labels = []
        self.list_main_spk_paths = []

        # Read all wav paths
        if self.ordered_sp_flag:

            # Read the ordered speech txt file
            with open(ordered_speech_txt_path, 'r') as f:
                lines = f.readlines()
            # Extract the wav file paths from the lines
            for line in lines:
                line = line.strip()
                if line.endswith('.wav'):
                    wav_path = Path(line)
                    if wav_path.exists():
                        self.list_audio_paths.append(wav_path)
                    else:
                        print(f"Warning: {wav_path} does not exist.")
            self.max_iter = len(self.list_audio_paths)

        else:
            self.list_audio_paths = sorted(list(input_wav_path.glob('*.wav')))
            self.list_spk_labels = gen_labels_list(self.list_audio_paths)


        self.output_csv_path = output_dict['output_csv_path']
        self.output_csv_path_2 = self.output_csv_path.parent.joinpath(f"{self.audio_name}_dt.csv")
        self.proc_log = output_dict['output_log_path']
        self.output_folder = output_dict['output_wav_path']

        self.bk_min_gain = dict_params['bk_gain_range'][0]
        self.bk_max_gain = dict_params['bk_gain_range'][1]

        self.bk_init_reduce = dict_params['bk_init_reduce']
        self.bk_inner_min = dict_params['bk_inner_gain_range'][0]
        self.bk_inner_max = dict_params['bk_inner_gain_range'][1]
        self.bk_inner_dur_min = dict_params['bk_inner_dur_range'][0] 
        self.bk_inner_dur_max = dict_params['bk_inner_dur_range'][1]

        self.gain_var_flag = dict_params['gain_var_flag']
        self.sp_init_reduce = dict_params['sp_init_reduce']
        self.sp_inner_min = dict_params['sp_inner_gain_range'][0]
        self.sp_inner_max = dict_params['sp_inner_gain_range'][1]
        self.sp_inner_dur_min = dict_params['sp_inner_dur_range'][0] 
        self.sp_inner_dur_max = dict_params['sp_inner_dur_range'][1]


        self.ns_min_gain = dict_params['ns_gain_range'][0]
        self.ns_max_gain = dict_params['ns_gain_range'][1]

        self.bk_num = dict_params['bk_num']
        self.sr         = 16000
        self.output_number = dict_params['output_samples_num']

        self.sp_gain = dict_params['sp_gain']
        self.ns_gain = dict_params['ns_gain']
        self.ns_gain_away = dict_params['ns_gain_away']
        self.bk_gain = dict_params['bk_gain']

        self.bk_num_segments = dict_params['bk_num_segments'] 
        self.rnd_offset_secs = dict_params['rnd_offset_secs']

        self.length_minutes = dict_params['length_minutes']
        self.min_offset_secs = dict_params['bk_ext_offset_range'][0]
        self.max_offset_secs = dict_params['bk_ext_offset_range'][1]
        self.store_sample = dict_params['debug_store_audio']
        self.number_noise_samples = dict_params['ns_number_samples'] 

        # self.max_silence_ai = 0.5
        # self.ns_ai_gain = 0.6

        self.max_silence_ai = 0.5
        self.ns_ai_gain = 0.3

        if  self.sp_gain <= 0:
            self.speaker_flag = False
        else:
            self.speaker_flag = True

        if self.ns_gain <= 0:
            self.noise_flag = False
        else:
            self.noise_flag = True

        if self.bk_gain <= 0:
            self.bk_flag = False
        else:
            self.bk_flag = True
        


        # Numpy output array according to the desired output
        self.x_data_DA = []

        global counter_small
        counter_small = 0


    def audio_mod(self, signal, gain_value,
                  outside_idx = 0,
                  noise_audio_name = 'noise_audio'):
        """
        Modifies the signal with a gain_value (percentage) and
        offset_value(percentage) and applies a random gain variation at the output.
        """

        signal_result = np.zeros_like((signal),
                                        dtype='float32')

        others_current_audio = signal
        

        # Verify the others_current_audio has a length greater than 0
        if len(others_current_audio) == 0:
            msg = f">> Audio has a length of 0. {outside_idx} - {noise_audio_name}\n"
            log_message(msg, self.proc_log, 'a', True)
            good_audio = False
            return good_audio, signal_result


        # Verify the others_current_audio is not empty
        if np.all(others_current_audio== 0):
            msg = f">> Audio has all 0. {outside_idx} - {noise_audio_name}\n"
            log_message(msg, self.proc_log, 'a', True)
            good_audio = False
            return good_audio, signal_result
        
        # Verify others_current_audio max and min values are not equal
        if min(others_current_audio) == max(others_current_audio):
            msg = f">> Audio max and min values are equal. {outside_idx} - {noise_audio_name}\n"
            log_message(msg, self.proc_log, 'a', True)
            good_audio = False
            return good_audio, signal_result

        # Apply gain value and convert to required output format
        signal_offset_norm  = norm_others_float32(others_current_audio,
                                                  gain_value = gain_value)

        # Verify the audio normalization did not failed
        audio_sum = np.sum(signal_offset_norm)
        if np.isnan(audio_sum):
            msg = f">> NaN found in Audio - {outside_idx} - {noise_audio_name}\n"
            log_message(msg, self.proc_log, 'a', True)
            good_audio = False
            return good_audio, signal_result

        signal_result = gain_variation(signal_offset_norm, init_reduce = self.bk_init_reduce,
                                       factor_min=self.bk_inner_min, factor_max=self.bk_inner_max,
                                       min_duration_ratio = self.bk_inner_dur_min, 
                                       max_duration_ratio = self.bk_inner_dur_max,
                                       verbose=False)
        
        good_audio = True

        return good_audio, signal_result


    def prepare_bk_audio(self):

        indx_others = random.randint(0, len(self.list_audio_paths)-1)
        others_audio_original, noise_audio_name = read_audio_name(self.list_audio_paths, indx_others)
        others_audio_f32 = others_audio_original.astype('float32')
        others_audio_trimmed = np.trim_zeros(others_audio_f32)

        audio_bk_ready = np.zeros_like(others_audio_f32)

        gain_value = gen_random_gaussian(self.bk_min_gain, self.bk_max_gain)

        # Skip if audio trimmed is all zeros
        if np.all(others_audio_trimmed == 0):
            msg = f">>>>> Audio is empty: {indx_others} {noise_audio_name}\n"
            log_message(msg, self.proc_log, 'a', True)
            good_audio = False
        else:
            good_audio, audio_bk_ready = self.audio_mod(others_audio_trimmed, gain_value,
                                        outside_idx=indx_others,
                                        noise_audio_name=noise_audio_name)
 
        return good_audio, audio_bk_ready, indx_others, noise_audio_name
  
  


    def gen_long_noise(self, current_audio_info, 
                        noise_gain_low, noise_gain_high,
                        verbose = False):

        length_current_audio, _, _ = current_audio_info 

        selected_arrays = []
        total_elements = 0
        while total_elements < length_current_audio:
            if random.choice([True, False]):
                indx_others = random.randint(0, len(self.noiseE1_data)-1)
                others_audio = self.noiseE1_data[indx_others].astype('float32')

            else:
                indx_others = random.randint(0, len(self.noiseE2_data)-1)
                others_audio = self.noiseE2_data[indx_others].astype('float32')

            others_audio = np.trim_zeros(others_audio)
            gain_value = gen_random_on_range(noise_gain_low, noise_gain_high)

            noise_ready  = others_audio * gain_value

            selected_arrays.append(noise_ready)
            total_elements += np.size(noise_ready)
            if verbose:
                print(f'len total: {length_current_audio} \t noise_gain: {gain_value} \t cnt: {np.size(noise_ready)}')

        return np.concatenate(selected_arrays)


    def generate_long_audio_from_folder(self, ai_noise_directory, length_current_audio, sr, max_silence_ai, verbose=False):
        """
        Generate a long audio file by sampling random audio files from a folder.
        Includes random seconds of silence between samples and trims the last audio
        to make the length exactly `length_current_audio`.

        Parameters:
            ai_noise_directory (str): Path to the folder containing audio files.
            length_current_audio (int): Desired length of the output audio in samples.
            sr (int): Sampling rate of the audio files.
            verbose (bool): If True, print debug information.

        Returns:
            np.ndarray: The generated long audio of length `length_current_audio`.
        """
        # Get a list of all audio files in the directory
        noise_audio_files = [os.path.join(ai_noise_directory, f) for f in os.listdir(ai_noise_directory) if f.endswith('.wav')]
        if not noise_audio_files:
            raise ValueError("No audio files found in the specified directory.")

        # Initialize the output audio array
        current_aiNoise_long_audio = np.zeros(length_current_audio, dtype='float32')
        total_length = 0

        while total_length < length_current_audio:
            # Randomly select an audio file
            ai_selected_file = random.choice(noise_audio_files)
            if verbose:
                print(f"Selected file: {ai_selected_file}")

            # Read the audio file
            current_ai_noise, _ = sf.read(ai_selected_file, dtype='float32')

            # Trim silence from the audio
            current_ai_noise = np.trim_zeros(current_ai_noise)

            # Generate a random silence duration (in samples)
            max_silence_duration = int(max_silence_ai * sr)  # Max silence of 0.5 seconds
            silence_duration = random.randint(0, max_silence_duration)
            silence = np.zeros(silence_duration, dtype='float32')

            # Concatenate the audio and silence
            audio_with_silence = np.concatenate((silence, current_ai_noise))

            # Check if adding this audio exceeds the desired length
            remaining_length = length_current_audio - total_length
            if len(audio_with_silence) > remaining_length:
                # Trim the audio to fit the remaining length
                audio_with_silence = audio_with_silence[:remaining_length]

            # Add the audio to the output
            current_aiNoise_long_audio[total_length:total_length + len(audio_with_silence)] = audio_with_silence
            total_length += len(audio_with_silence)

            if verbose:
                print(f"Current total length: {total_length}/{length_current_audio}")

        return current_aiNoise_long_audio


    def create_long_audio_others(self, current_audio_info, verbose = False):

        # Generate initial offset
        offset_value = gen_random_on_range(self.min_offset_secs*self.sr, self.max_offset_secs*self.sr)

        if verbose:
            print(f'Offset in secs: {round(offset_value/self.sr, 2)}')

        # Divide the n minutes into segments and randomly assign each segment to an audio file
        segments = np.linspace(0 + offset_value, self.length_minutes*60*self.sr + offset_value, self.bk_num_segments + 1)
        segment_limits = [(segments[i], segments[i+1]) for i in range(self.bk_num_segments)]
        random.shuffle(segment_limits)

        list_of_audios = []

        # Call the extend_audio function for each input audio file with its corresponding limit_lower and limit_upper values
        for idx in range(0, self.bk_num_segments):

            # Apply mods to audio here:
            good_audio, current_audio, random_idx, noise_audio_name = self.prepare_bk_audio()

            if not good_audio:
                print(f'<<< Skipped! Audio {random_idx} - {noise_audio_name} is not good')
                continue
            
            limit_lower, limit_upper = segment_limits[idx]

            if verbose:
                print(f'\nIndex selected: {random_idx} \t {limit_lower} \
                      ({round(limit_lower/self.sr, 2)}) - {limit_upper} \
                      ({round(limit_upper/self.sr, 2)})')

            length_extended_sp, _, _ = current_audio_info 

            total_length_audio = self.length_minutes * 60 * self.sr

            # Raise error if total_length_audio is different than length_extended_sp
            if total_length_audio != length_extended_sp:
                msg = f">> Error! total_length_audio != length_extended_sp | Audio {random_idx} - {noise_audio_name}\n"
                msg += f"total_length_audio: {total_length_audio} \t length_extended_sp: {length_extended_sp}\n"
                log_message(msg, self.proc_log, 'a', True)
                raise ValueError("total_length_audio != length_extended_sp")

            ext_audio_raw, _ = extend_audio_bk(current_audio, limit_lower, limit_upper, 
                                                        total_length_audio,
                                                        offset_value,
                                                        verbose = False)

            ext_audio = ext_audio_raw[0:total_length_audio]

            if verbose:
                print(f'Current segment extended {idx}: {len(ext_audio_raw)}')

            ext_audio = remove_dc_component(ext_audio)
            list_of_audios.append(ext_audio)

        result_audio = sum_arrays(list_of_audios)
        return result_audio


    def create_long_audio_main(self, max_offset = 4,\
                            gain_var_flag = True, sp_idx = -1, verbose = False):

        GT_log = []

        seg_idx = 0

        total_ext_length = self.sr * self.length_minutes * 60
        index_list = list(range(len(self.list_main_spk_paths)))
        list_of_audios = []

        prev_stop_time = 0
        labels_used = []

        while 1:
            if sp_idx == -1:
                # Choose a random element from the list
                random_idx = random.choice(index_list)

                # Read wav file
                raw_audio, current_audio_name = read_audio_name(self.list_main_spk_paths, 
                                                                random_idx)
                current_label_selected = current_audio_name.split('_')[0][3:]
                labels_used.append(current_label_selected)
                if verbose:
                    log_message(f'{random_idx} - lbl: {current_label_selected}\n', self.proc_log, 'a', both=True)
            else:
                random_idx = sp_idx
                if sp_idx == len(self.list_audio_paths)-1:
                    self.complete_flag = True
                    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>New iteration, sp_idx: {sp_idx}')
                    sp_idx = 0
                else:
                    sp_idx += 1

                # Read wav file
                raw_audio, current_audio_name = read_audio_name(self.list_audio_paths, 
                                                                random_idx)

            total_length_audio = len(raw_audio)

            # # Uniform distribution random
            # rnd_offset = int(max_offset*self.sr*random.random())

            # Gaussian distribution std_dev = 1.5 
            rnd_offset = int(gen_random_gaussian(0.2, max_offset)*self.sr)

            start_index = seg_idx + rnd_offset 
            end_index = start_index + total_length_audio

            if (total_ext_length - end_index) < total_length_audio:
                # print(f'Last iteration: {start_index} - {end_index} | audio_len: {total_length_audio} | left: {total_ext_length - end_index}')
                break

            if start_index < 0:
                print(f'Error! negative start time: {start_index}')

            # if verbose:
            #     print(f'\nIndex selected: {random_idx} \t {start_index} - {end_index} | offset: {rnd_offset}')


            seg_idx = end_index
            # Apply mods to audio here:

            if gain_var_flag:
                current_audio = gain_variation(raw_audio, 
                                            init_reduce = self.sp_init_reduce,
                                            factor_min=self.sp_inner_min, 
                                            factor_max=self.sp_inner_max,
                                            min_duration_ratio = self.sp_inner_dur_min, 
                                            max_duration_ratio = self.sp_inner_dur_max,
                                            verbose=False)
            else:
                current_audio = raw_audio
            
            packed_extended_audio_result = extend_audio(current_audio, 
                                                        start_index, end_index, 
                                                        total_ext_length, verbose=False)

            ext_audio = packed_extended_audio_result[0]
            current_start_time, current_stop_time = packed_extended_audio_result[1] 

            if current_start_time <= prev_stop_time:
                print(f'Error! there should be no overlaps! {current_start_time} - {current_stop_time}')
            
            prev_stop_time = current_stop_time

            GT_log.append((current_audio_name, current_start_time, current_stop_time))

            if len(ext_audio) > total_ext_length:
                print(f'Warning, main audio length is {len(ext_audio)}, larger than {len(ext_audio)}')
                ext_audio = ext_audio[0:total_ext_length]

            ext_audio = remove_dc_component(ext_audio)
            list_of_audios.append(ext_audio)

        # Select which main speakers to add into the result audio
        number_audios = len(list_of_audios)
        result_audio = np.zeros_like(list_of_audios[0])

        # Conversation Pattern
        if self.pattern_flag:
            bool_sequence = create_random_sequence(number_audios)
            # bool_sequence = create_random_sequence2(number_audios)
        else:
            bool_sequence = [0]*number_audios 

        # print(f'Bool sequence:\n{bool_sequence}')

        GT_log_filtered = []
        labels_used_filtered = []
        for i in range(0, number_audios):
            if bool_sequence[i] == 0:
                result_audio += list_of_audios[i]
                GT_log_filtered.append(GT_log[i])
                if not(self.ordered_sp_flag):
                    labels_used_filtered.append(labels_used[i])
            # else:
            #     print(f'Audio {i} skipped')
        
        # Print labels used and count how much times each label was used
        if not(self.ordered_sp_flag):
            label_counts_filtered = Counter(labels_used_filtered)
            label_counts_init = Counter(labels_used)
            for label, count in label_counts_filtered.items():
                print(f'Label {label} used {count} times | initial {label_counts_init[label]} times')

        return result_audio, GT_log_filtered, sp_idx


    def long_distance_noise(self, number_samples=45, length_min=3, verbose = False):
        audio_samples = random.sample(list(self.data_E3), number_samples)

        # Calculate the total length of all audio samples
        total_audio_length = sum(len(audio) for audio in audio_samples)

        while(True):
            if (length_min*60*self.sr) < total_audio_length:
                audio_samples.pop()
                total_audio_length = sum(len(audio) for audio in audio_samples)
                if verbose:
                    print(f'Warning, Value {number_samples} is too big -> n-1')
                continue
            else:
                # Calculate the maximum length of silence to insert between audio samples
                max_silence_length = ((length_min * 60 * self.sr) - total_audio_length) // (number_samples + 1)

                # Create the final audio array
                audio = np.zeros(length_min * 60 * self.sr)

                # Insert each audio sample into the final audio array at a random position with silence in between
                start = random.randint(0, max_silence_length)
                for audio_sample in audio_samples:
                    audio_length = len(audio_sample)
                    end = start + audio_length
                    audio[start:end] += audio_sample
                    start = end + random.randint(0, max_silence_length)

                return audio

    def sim_single_signal(self, indx=0, sp_idx = -1):
        """
        Pyroomacoustics simulation with 3 random other audios
        from the same x_data + 1 noise from AudioSet
        """

        cfg_info = gen_room_config()

        shoebox_vals, mic_dict, lower_left_point, upper_right_point, \
        lower_left_table, upper_right_table, noise_coords, ai_noise_coords, abs_coeff, fs = cfg_info

        # Create 3D room and add sources
        room = pra.ShoeBox(shoebox_vals,
                           fs=fs,
                           materials=pra.Material(abs_coeff),
                           max_order=12)

        # randomly select the coordinates for the main speaker:
        main_speaker_coords, tuple_outside_inside = gen_rand_table_coords(lower_left_point,
                                upper_right_point,  lower_left_table,
                                upper_right_table, table_margin = 0.3)

        # Sample 5 speakers from the dataset
        if not(self.ordered_sp_flag):
            self.selected_speakers = random.sample(list(set(self.list_spk_labels)), 5)
            # log_message(f'Selected speakers: {self.selected_speakers}\n', self.proc_log, 'a', both=True)
            self.list_main_spk_paths = extract_main_speakers(self.list_audio_paths, self.selected_speakers) 


        result_audio, GT_log1, sp_idx = self.create_long_audio_main(max_offset = self.rnd_offset_secs,
                                                            gain_var_flag = self.gain_var_flag,
                                                            sp_idx = sp_idx,
                                                            verbose = False)
        print(f'main index after main audio: {sp_idx}')

        generate_csv_file_tts3(GT_log1, self.output_csv_path,
                          indx, self.audio_name, only_speaker=True) 

        length_current_audio = len(result_audio)
        outmin_current = result_audio.min()
        outmax_current = result_audio.max()

        current_audio_info = [length_current_audio, outmin_current, outmax_current]

        room.add_source(main_speaker_coords,
                        signal=result_audio*self.sp_gain)

        # Second main speaker
        if self.double_talk_flag:
            result_audio2, GT_log2, _ = self.create_long_audio_main(max_offset = self.rnd_offset_secs,
                                                                gain_var_flag = self.gain_var_flag,
                                                                verbose = False)

            generate_csv_file_tts3(GT_log2, self.output_csv_path,
                            indx, self.audio_name2, only_speaker=True) 



            main_speaker_coords2 = [main_speaker_coords[0] + 0.4, main_speaker_coords[1] + 0.4, main_speaker_coords[2]]

            room.add_source(main_speaker_coords2,
                            signal=result_audio2*self.sp_gain)
            
            if indx == 0:
                output_path = self.proc_log.parent.joinpath("main_audio_sample_dt.wav")
                sf.write(output_path, result_audio2*self.sp_gain, self.sr, subtype='FLOAT')

        if indx == 0:
            output_path = self.proc_log.parent.joinpath("main_audio_sample.wav")
            sf.write(output_path, result_audio*self.sp_gain, self.sr, subtype='FLOAT')

            print(f'Length of {indx}th main audio: {len(result_audio)}({round(len(result_audio)/self.sr, 2)})')

        single_cfg = {'mic': [mic_dict['mic_0'][0], mic_dict['mic_0'][1]],
                    'main': [main_speaker_coords[0], main_speaker_coords[1]],
                    'others': [],
                    'regions': [(lower_left_point, upper_right_point),
                                (lower_left_table, upper_right_table),
                                tuple_outside_inside[0],
                                tuple_outside_inside[1]],
                    'room': [shoebox_vals[0], shoebox_vals[1]]}

        # Add background speakers
        if self.bk_flag:
            for i in range(0, self.bk_num):

                result_audio = self.create_long_audio_others(current_audio_info, 
                                                                verbose = False)
                                                            
                # Randomly select the coordinates based on position
                rand_coordinates_other = gen_rand_coordinates(shoebox_vals[0], shoebox_vals[1],
                                                            lower_left_point,
                                                            upper_right_point)

                if self.store_sample:
                    if (indx == 0) and (i == 1):
                        output_path = self.proc_log.parent.joinpath(f"others_long_audio_sample_{i}.wav")
                        sf.write(output_path, result_audio*self.bk_gain, self.sr, subtype='FLOAT')
                        print(f'{indx}-Length of {i}th bk_audio: {len(result_audio)}({round(len(result_audio)/self.sr, 2)})')

                single_cfg['others'].append((rand_coordinates_other[0], rand_coordinates_other[1]))

                room.add_source(rand_coordinates_other,
                                signal=result_audio*self.bk_gain)

        # Add Noise 
        if self.noise_flag:
            # Lalal noise
            ai_long_noise = self.generate_long_audio_from_folder(self.ai_noise_directory,
                                                            length_current_audio,
                                                            self.sr,
                                                            self.max_silence_ai,
                                                            verbose=False)

            if self.store_sample and (indx == 0):
                output_path = self.proc_log.parent.joinpath("long_ai_lalal_noise.wav")
                sf.write(output_path, ai_long_noise*self.ns_ai_gain, self.sr, subtype='FLOAT')
                print(f'{indx}-Length of LALAL noise: {len(ai_long_noise)}({round(len(ai_long_noise)/self.sr, 2)})')


            room.add_source(ai_noise_coords,
                            signal = ai_long_noise*self.ns_ai_gain)

            # Distance E1 & E2 Noise
            noise_audio_long = self.gen_long_noise(current_audio_info, self.ns_min_gain, self.ns_max_gain)

            if self.store_sample and (indx == 0):
                output_path = self.proc_log.parent.joinpath("long_noise.wav")
                sf.write(output_path, noise_audio_long*self.ns_gain, self.sr, subtype='FLOAT')
                print(f'{indx}-Length of noise: {len(noise_audio_long)}({round(len(noise_audio_long)/self.sr, 2)})')

            single_cfg['noise'] = (noise_coords[0], noise_coords[1])
            
            room.add_source(noise_coords,
                            signal = noise_audio_long*self.ns_gain)
        
            # Distance E3 Noise
            rand_coordinates_noise_E3 = gen_rand_coordinates(shoebox_vals[0], shoebox_vals[1],
                                                            lower_left_point,
                                                            upper_right_point)

            long_noise_E3 = self.long_distance_noise(number_samples=self.number_noise_samples,
                                                    length_min = self.length_minutes)
            if self.store_sample and (indx == 0):
                output_path = self.proc_log.parent.joinpath("long_noise_E3_X.wav")
                sf.write(output_path, long_noise_E3*self.ns_gain_away, self.sr, subtype='FLOAT')
                print(f'{indx}-Length of distance noise: {len(long_noise_E3)}({round(len(long_noise_E3)/self.sr, 2)})')

            room.add_source(rand_coordinates_noise_E3,
                            signal = long_noise_E3*self.ns_gain_away)


        sim_audio = np.zeros((length_current_audio), dtype='float32')

        R = np.array([[mic_dict["mic_0"][0], mic_dict["mic_1"][0]],
                    [mic_dict["mic_0"][1], mic_dict["mic_1"][1]], 
                    [mic_dict["mic_0"][2], mic_dict["mic_1"][2]]])
        room.add_microphone(R)

        # Compute image sources
        room.image_source_model()
        room.simulate()

        # Simulate audio
        sim_audio = room.mic_array.signals[0, :]

        sum_noise = np.sum(sim_audio)
        if np.isnan(sum_noise):
            msg = f">>> NaN found after pyroom\n"
            msg += f"{indx} -- Len {length_current_audio}\n"
            log_message(msg, self.proc_log, 'a', True)

        return sim_audio, sp_idx


    def sim_vad_dataset(self):
        prev_time = time.process_time()
        msg = f'Starting simulation of {self.output_number} audios\n'
        log_message(msg, self.proc_log, 'w', True)

        # random or ordered main speaker
        if self.ordered_sp_flag:
            main_idx = 0
            # main_idx = 8956
            # main_idx = 15162
        else:
            main_idx = -1

        # for indx in range(0, self.output_number):
        indx = 0
        while True:
            if self.ordered_sp_flag:
                if self.complete_flag:
                    print(f'End of ordered speech list: {indx}')
                    break
            else:
                if indx == self.output_number:
                    print(f'End of random speech list: {indx}')
                    break
            
            print(f'{indx} - Processing {main_idx} audio')
            single_x_DA, main_idx = self.sim_single_signal(indx=indx, sp_idx=main_idx)

            # Calculate percentage of done main_idx compared to self.max_iter
            if self.ordered_sp_flag:
                percentage_done = (main_idx / self.max_iter) * 100
                print(f'\t\tpost main_idx: {main_idx} | {percentage_done:.2f}%')

            single_x_DA_trimmed = eliminate_noise_start_ending(single_x_DA, 0.00001)

            # Apply a high-pass filter to remove the DC component 
            normalized_audio = amplify_audio_to_0db(single_x_DA_trimmed)

            normalized_audio = remove_dc_component(normalized_audio)


            # Apply 3.2 DB of gain to the audio
            amp_audio = amplify_audio_to_3_2db(normalized_audio)

            new_name_wav = f'preComp{self.audio_name}_{indx}.wav'
            output_path_wav = self.output_folder.joinpath(new_name_wav) 
            sf.write(str(output_path_wav), amp_audio, self.sr, subtype='FLOAT')

            if indx%5 == 0:
                current_time_100a = time.process_time()
                time_100a = current_time_100a - prev_time

                msg = f"{indx} | time for 5 audios: {time_100a:.2f}\n"
                log_message(msg, self.proc_log, 'a', True)
                prev_time = current_time_100a
            
            indx += 1

        print("These number of small audios:" + str(counter_small))

