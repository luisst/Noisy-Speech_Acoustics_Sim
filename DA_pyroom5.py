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
import soundfile as sf

from cfg_pyroom_VAD import cfg_info, gen_rand_coordinates, \
    gen_rand_table_coords, sr

from plot_configuration_room import plot_single_configuration

from DA_pyroom_utils import norm_others_float32, log_message, gain_variation, \
    gen_random_on_range, norm_noise_f32, gen_random_on_range, gen_random_on_range, \
    sum_arrays, eliminate_noise_start_ending, extend_audio, convert_param_dict, \
    generate_csv_file, remove_dc_component, gen_random_gaussian, extend_audio_bk, \
    read_audio_name, apply_reverb, amplify_audio_to_0db, name_mapping_TTS2


class DAwithPyroom(object):
    """
    Class for audio simulation using pyroom.
    input signal + 4 random crosstalk + background noise
    """

    def __init__(self, input_wav_path,
                 noise_dict, output_dict, 
                 all_params):
        """
        Initialize all the params from the dictionaries. Examples provided in the Main_long.py file
        """
        self.audio_name = 'DA_long'
        noise_path1 = noise_dict['noise_npy_folder'].joinpath(noise_dict['noise_e1_soft'])
        noise_path2 = noise_dict['noise_npy_folder'].joinpath(noise_dict['noise_e2_loud'])
        noise_path3 = noise_dict['noise_npy_folder'].joinpath(noise_dict['noise_e3_distance'])

        self.noiseE1_data = np.load(noise_path1, allow_pickle=True) #
        self.noiseE2_data = np.load(noise_path2, allow_pickle=True) #
        self.data_E3 = np.load(noise_path3, allow_pickle=True) #

        # Read all wav paths
        self.list_audio_paths = sorted(list(input_wav_path.glob('*.wav')))

        dict_params = convert_param_dict(all_params)

        self.output_csv_path = output_dict['output_csv_path']
        self.proc_log = output_dict['output_log_path']
        self.output_folder = output_dict['output_wav_path']

        self.bk_min_gain = dict_params['bk_gain_range'][0]
        self.bk_max_gain = dict_params['bk_gain_range'][1]
        self.bk_min_offset = dict_params['bk_offset_range'][0]
        self.bk_max_offset = dict_params['bk_offset_range'][1]

        self.bk_init_reduce = dict_params['bk_init_reduce']
        self.bk_inner_min = dict_params['bk_inner_gain_range'][0]
        self.bk_inner_max = dict_params['bk_inner_gain_range'][1]
        self.bk_inner_dur_min = dict_params['bk_inner_dur_range'][0] 
        self.bk_inner_dur_max = dict_params['bk_inner_dur_range'][1]
        self.bk_reverb = dict_params['bk_reverb']

        self.sp_init_reduce = dict_params['sp_init_reduce']
        self.sp_inner_min = dict_params['sp_inner_gain_range'][0]
        self.sp_inner_max = dict_params['sp_inner_gain_range'][1]
        self.sp_inner_dur_min = dict_params['sp_inner_dur_range'][0] 
        self.sp_inner_dur_max = dict_params['sp_inner_dur_range'][1]
        self.sp_reverb = dict_params['sp_reverb']


        self.ns_min_gain = dict_params['ns_gain_range'][0]
        self.ns_max_gain = dict_params['ns_gain_range'][1]

        self.bk_num = dict_params['bk_num']
        self.sr         = sr
        self.output_number = dict_params['output_samples_num']

        self.sp_gain = dict_params['sp_gain']
        self.ns_gain = dict_params['ns_gain']
        self.bk_gain = dict_params['bk_gain']

        self.bk_num_segments = dict_params['bk_num_segments'] 
        self.rnd_offset_secs = dict_params['rnd_offset_secs']

        self.length_min = dict_params['length_min']
        self.noise_dist = dict_params['ns_close_dist']
        self.min_offset_secs = dict_params['bk_ext_offset_range'][0]
        self.max_offset_secs = dict_params['bk_ext_offset_range'][1]
        self.store_sample = dict_params['debug_store_audio']
        self.number_noise_samples = dict_params['ns_number_samples'] 

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


    def audio_mod(self, signal, gain_value, offset_value,
                  length_current_audio, outmin_current, outmax_current, no_offset = False):
        """
        Modifies the signal with a gain_value (percentage) and
        offset_value(percentage) and applies a random gain variation at the output.
        """

        # counter for small audios
        global counter_small

        signal_length = signal.shape[0]
        signal_offset = np.zeros_like(signal)
        others_current_audio = np.zeros((length_current_audio),
                                        dtype='float32')

        # Init values
        factor1 = 1.0
        factor2 = 1.0

        if no_offset:
            others_current_audio = signal
        else:
            factor_length = np.min([length_current_audio, signal_length])
            
            # Calculate the offset factors at the start and end
            factor1 = int(factor_length*((abs(offset_value) - offset_value)/2))
            factor2 = int(factor_length*((abs(offset_value) + offset_value)/2))

            # Apply the offset factors
            signal_offset[factor1:(signal_length - factor2)] = \
                signal[factor2:(signal_length - factor1)]

            # Trim offset signal to the real length of the audio
            if signal_length > length_current_audio:
                others_current_audio = signal_offset[0:length_current_audio]
            else:
                others_current_audio[0:signal_length] = signal_offset
            
        # Apply gain value and convert to required output format
        signal_offset_norm  = norm_others_float32(others_current_audio,
                                                  gain_value = gain_value,
                                                  outmin = outmin_current,
                                                  outmax = outmax_current)

        # Verify the audio is large enough to withstand the offset
        audio_sum = np.sum(signal_offset_norm)
        if audio_sum == 0:
            if factor1 >= length_current_audio:
                counter_small = counter_small + 1
                msg = f"> small audio encountered: counter {counter_small} \n"
                log_message(msg, self.proc_log, 'a', True)
            else:
                msg = f"> Cero was found. Signal_offset {offset_value}. Factor {factor1} - {factor2} \n|"
                msg += f"INDEXS signal_offset {factor1} - {(signal_length - factor2)}. signal {factor2} - {(signal_length - factor1)}\n"
                log_message(msg, self.proc_log, 'a', True)


        # Verify the audio normalization did not failed
        if np.isnan(audio_sum):
            msg = f">> NaN found in Audio. Offset: {str(offset_value)}\n"
            log_message(msg, self.proc_log, 'a', True)

        signal_result = gain_variation(signal_offset_norm, init_reduce = self.bk_init_reduce,
                                       factor_min=self.bk_inner_min, factor_max=self.bk_inner_max,
                                       min_duration_ratio = self.bk_inner_dur_min, 
                                       max_duration_ratio = self.bk_inner_dur_max,
                                       verbose=False)

        return signal_result


    def noise_mod(self, noise, gain_value, length_current_audio):
        """
        Modifies the signal with a gain_value (percentage) and
        offset_value(percentage)
        """

        # Calculate the offset factors at the start
        noise_length = noise.shape[0]
        noise_current_audio = np.zeros((length_current_audio), dtype='float64')

        # Accomodate noise audios within the signal audio length
        if noise_length > length_current_audio:
            noise_current_audio = noise[0:length_current_audio]
        else:
            noise_current_audio[0:noise_length] = noise

        # Apply gain value and convert to required output format
        signal_offset_norm  = norm_noise_f32(noise_current_audio,
                                          gain_value)

        sum_noise = np.sum(signal_offset_norm)
        if np.isnan(sum_noise):
            msg = f">> NaN found in Noise\n"
            log_message(msg, self.proc_log, 'a', True)

        return signal_offset_norm


    def prepare_bk_audio(self, current_audio_info, no_offset = False, return_indx = False):

        length_current_audio, outmin_current, outmax_current = current_audio_info 

        indx_others = random.randint(0, len(self.list_audio_paths)-1)
        others_audio, _ = read_audio_name(self.list_audio_paths, indx_others)
        others_audio = others_audio.astype('float32')
        # others_audio = apply_reverb(others_audio, reverb_vals=self.bk_reverb)
        others_audio = np.trim_zeros(others_audio)

        offset_value = gen_random_on_range(self.bk_min_offset, self.bk_max_offset)

        gain_value = gen_random_gaussian(self.bk_min_gain, self.bk_max_gain)
        
        audio_bk_ready = self.audio_mod(others_audio, gain_value,
                                       offset_value, length_current_audio,
                                       outmin_current, outmax_current,
                                       no_offset=no_offset)

        if return_indx:
            return audio_bk_ready, indx_others
        else:
            return audio_bk_ready


    def prepare_noise_audio(self, current_audio_info, mic_dict, 
                            noise_gain_low, noise_gain_high,
                            dist_e1, dist_e2, verbose = False):

        length_current_audio, outmin_current, outmax_current = current_audio_info 

        if random.choice([True, False]):
            indx_others = random.randint(0, len(self.noiseE1_data)-1)
            # print(f'Random E1 int: {indx_others}')
            others_audio = self.noiseE1_data[indx_others].astype('float32')

            noise_x, noise_y, noise_z = mic_dict["mic_0"]
            noise_coords = [noise_x + dist_e1, noise_y + dist_e1, noise_z]
        else:
            indx_others = random.randint(0, len(self.noiseE2_data)-1)
            others_audio = self.noiseE2_data[indx_others].astype('float32')

            noise_x, noise_y, noise_z = mic_dict["mic_0"]
            noise_coords = [noise_x + dist_e2, noise_y + dist_e2, noise_z]

        others_audio = np.trim_zeros(others_audio)
        gain_value = gen_random_on_range(noise_gain_low, noise_gain_high)
        if verbose:
            print(f'Noise gain: {gain_value}')

        audio_bk_ready = self.noise_mod(others_audio, gain_value,
                                       length_current_audio)

        return noise_coords, audio_bk_ready


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

            noise_ready  = norm_noise_f32(others_audio,
                                        gain_value)

            selected_arrays.append(noise_ready)
            total_elements += np.size(noise_ready)
            if verbose:
                print(f'len total: {length_current_audio} \t noise_gain: {gain_value} \t cnt: {np.size(noise_ready)}')

        return np.concatenate(selected_arrays)


    def create_long_audio_others(self, current_audio_info, num_segments = 20, length_min = 3, verbose = False):

        # Generate initial offset
        offset_value = gen_random_on_range(self.min_offset_secs*self.sr, self.max_offset_secs*self.sr)

        if verbose:
            print(f'Offset in secs: {round(offset_value/self.sr, 2)}')

        # Divide the n minutes into segments and randomly assign each segment to an audio file
        segments = np.linspace(0 + offset_value, length_min*60*self.sr + offset_value, num_segments + 1)
        segment_limits = [(segments[i], segments[i+1]) for i in range(num_segments)]
        random.shuffle(segment_limits)

        list_of_audios = []

        # Call the extend_audio function for each input audio file with its corresponding limit_lower and limit_upper values
        for idx in range(0, num_segments):

            # Apply mods to audio here:
            current_audio, random_idx = self.prepare_bk_audio(current_audio_info, no_offset = True, return_indx=True)

            
            limit_lower, limit_upper = segment_limits[idx]

            if verbose:
                print(f'\nIndex selected: {random_idx} \t {limit_lower} \
                      ({round(limit_lower/self.sr, 2)}) - {limit_upper} \
                      ({round(limit_upper/self.sr, 2)})')

            ext_audio_raw, _ = extend_audio_bk(current_audio, limit_lower, limit_upper, 
                                                        length_min = length_min,
                                                        offset_samples = offset_value,
                                                        verbose = False)

            ext_audio = ext_audio_raw[0:length_min*60*self.sr]

            if verbose:
                print(f'Current segment extended {idx}: {len(ext_audio_raw)}')
            list_of_audios.append(ext_audio)

        result_audio = sum_arrays(list_of_audios)
        return result_audio


    def create_long_audio_main(self, max_offset = 4, length_min = 3, verbose = False):

        GT_log = []

        seg_idx = 0

        total_ext_length = sr * (length_min * 60)
        index_list = list(range(len(self.list_audio_paths)))
        list_of_audios = []

        prev_stop_time = 0

        while 1:
            # Choose a random element from the list
            random_idx = random.choice(index_list)
            index_list.remove(random_idx)

            # Read wav file

            raw_audio, current_audio_name = read_audio_name(self.list_audio_paths, 
                                                            random_idx)
            # raw_audio = apply_reverb(raw_audio, reverb_vals=self.sp_reverb)

            total_length_audio = len(raw_audio)

            # # Uniform distribution random
            # rnd_offset = int(max_offset*self.sr*random.random())

            # Gaussian distribution std_dev = 1.5 
            rnd_offset = int(gen_random_gaussian(1, max_offset)*self.sr)

            start_index = seg_idx + rnd_offset 
            end_index = start_index + total_length_audio

            if (total_ext_length - end_index) < total_length_audio:
                print(f'Last iteration: {start_index} - {end_index} | audio_len: {total_length_audio} | left: {total_ext_length - end_index}')
                break

            if start_index < 0:
                print(f'Error! negative start time: {start_index}')

            if verbose:
                print(f'\nIndex selected: {random_idx} \t {start_index} - {end_index} | offset: {rnd_offset} \t new length X_data: {len(index_list)}')


            seg_idx = end_index
            # Apply mods to audio here:
            current_audio = gain_variation(raw_audio, 
                                           init_reduce = self.sp_init_reduce,
                                           factor_min=self.sp_inner_min, 
                                           factor_max=self.sp_inner_max,
                                           min_duration_ratio = self.sp_inner_dur_min, 
                                           max_duration_ratio = self.sp_inner_dur_max,
                                           verbose=False)
            
            packed_extended_audio_result = extend_audio(current_audio, 
                                                        start_index, end_index, 
                                                        total_ext_length, verbose=False)

            ext_audio = packed_extended_audio_result[0]
            current_start_time, current_stop_time = packed_extended_audio_result[1] 

            if current_start_time <= prev_stop_time:
                print(f'Error! there should be no overlaps! {current_start_time} - {current_stop_time}')
            
            prev_stop_time = current_stop_time

            GT_log.append((current_audio_name, current_start_time, current_stop_time))

            if len(ext_audio) > length_min*60*self.sr:
                print(f'Warning, main audio length is {len(ext_audio)}, larger than {len(ext_audio)}')
                ext_audio = ext_audio[0:length_min*60*self.sr]

            list_of_audios.append(ext_audio)

        result_audio = sum_arrays(list_of_audios)
        return result_audio, GT_log


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

    def sim_single_signal(self, indx=0):
        """
        Pyroomacoustics simulation with 3 random other audios
        from the same x_data + 1 noise from AudioSet
        """

        shoebox_vals, mic_dict, lower_left_point, upper_right_point, \
        lower_left_table, upper_right_table, abs_coeff, fs = cfg_info

        # Create 3D room and add sources
        room = pra.ShoeBox(shoebox_vals,
                           fs=fs,
                           materials=pra.Material(abs_coeff),
                           max_order=12)

        # randomly select the coordinates for the main speaker:
        main_speaker_coords, tuple_outside_inside = gen_rand_table_coords(lower_left_point,
                                upper_right_point,  lower_left_table,
                                upper_right_table, table_margin = 0.3)


        result_audio, GT_log = self.create_long_audio_main(max_offset = self.rnd_offset_secs,
                                                            length_min = self.length_min, 
                                                            verbose = False)

        generate_csv_file(GT_log, self.output_csv_path,
                          indx, name_mapping_TTS2,
                          self.audio_name, only_speaker=True) 

        length_current_audio = len(result_audio)
        outmin_current = result_audio.min()
        outmax_current = result_audio.max()

        current_audio_info = [length_current_audio, outmin_current, outmax_current]

        if self.speaker_flag:
            room.add_source(main_speaker_coords,
                            signal=result_audio*self.sp_gain)
            if self.store_sample and (indx == 0):
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
                                                                num_segments = self.bk_num_segments,
                                                                length_min = self.length_min,
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
            noise_audio_long = self.gen_long_noise(current_audio_info, self.ns_min_gain, self.ns_max_gain)

            if self.store_sample and (indx == 0):
                output_path = self.proc_log.parent.joinpath("long_noise.wav")
                sf.write(output_path, noise_audio_long*self.ns_gain, self.sr, subtype='FLOAT')
                print(f'{indx}-Length of noise: {len(noise_audio_long)}({round(len(noise_audio_long)/self.sr, 2)})')

            noise_x, noise_y, noise_z = mic_dict["mic_0"]
            noise_coords = [noise_x + self.noise_dist, noise_y + self.noise_dist, noise_z]

            single_cfg['noise'] = (noise_coords[0], noise_coords[1])
            
            room.add_source(noise_coords,
                            signal = noise_audio_long*self.ns_gain)
        
            # Distance E3 Noise
            rand_coordinates_noise_E3 = gen_rand_coordinates(shoebox_vals[0], shoebox_vals[1],
                                                            lower_left_point,
                                                            upper_right_point)

            long_noise_E3 = self.long_distance_noise(number_samples=self.number_noise_samples,
                                                    length_min = self.length_min)
            if self.store_sample and (indx == 0):
                output_path = self.proc_log.parent.joinpath("long_noise_E3_X.wav")
                sf.write(output_path, long_noise_E3*self.ns_gain, self.sr, subtype='FLOAT')
                print(f'{indx}-Length of distance noise: {len(long_noise_E3)}({round(len(long_noise_E3)/self.sr, 2)})')

            room.add_source(rand_coordinates_noise_E3,
                            signal = long_noise_E3*self.ns_gain)

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

        return sim_audio


    def sim_vad_dataset(self):
        prev_time = time.process_time()
        for indx in range(0, self.output_number):
            single_x_DA = self.sim_single_signal(indx=indx)

            single_x_DA_trimmed = eliminate_noise_start_ending(single_x_DA, 0.00001)

            # Apply a high-pass filter to remove the DC component 
            filtered_audio = remove_dc_component(single_x_DA_trimmed)
            amp_audio = amplify_audio_to_0db(filtered_audio)

            new_name_wav = f'{self.audio_name}_{indx}.wav'
            output_path_wav = self.output_folder.joinpath(new_name_wav) 
            sf.write(str(output_path_wav), amp_audio, self.sr, subtype='FLOAT')

            if indx%20 == 0:
                current_time_100a = time.process_time()
                time_100a = current_time_100a - prev_time

                msg = f"{indx} | time for 100 audios: {time_100a:.2f}\n"
                log_message(msg, self.proc_log, 'a', True)
                prev_time = current_time_100a

        print("These number of small audios:" + str(counter_small))
