#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 10:48:43 2021

Latest version 5: October 10th, 2023

@author: Luis Sanchez Tapia
"""

import scipy.signal as signal
import random
import numpy as np
import pyroomacoustics as pra
import time
import librosa
import csv
import soundfile as sf

from cfg_pyroom_VAD import cfg_info, gen_rand_coordinates, \
    gen_rand_table_coords, sr

from plot_configuration_room import plot_single_configuration

from DA_pyroom_utils import norm_others_float32, log_message, gain_variation, \
    gen_random_on_range, norm_noise_f32, gen_random_on_range, gen_random_on_range, \
    sum_arrays, eliminate_noise_start_ending, extend_audio, convert_param_dict


class DAwithPyroom(object):
    """
    Class for audio simulation using pyroom.
    input signal + 4 random crosstalk + background noise
    """

    def __init__(self, input_path, input_path_names,
                 noise_dict, output_dict, 
                 all_params):
        """
        Start the class with the dataset path and turn the float32 flag for
        output format, False for int16 format
        """
        noise_path1 = noise_dict['noise_npy_folder'].joinpath(noise_dict['noise_e1_soft'])
        noise_path2 = noise_dict['noise_npy_folder'].joinpath(noise_dict['noise_e2_loud'])
        noise_path3 = noise_dict['noise_npy_folder'].joinpath(noise_dict['noise_e3_distance'])

        output_csv_path = output_dict['output_csv_path']
        output_log_path = output_dict['output_log_path']
        output_wav_path = output_dict['output_wav_path']

        self.x_data = np.load(input_path, allow_pickle=True)
        self.x_data_names = np.load(input_path_names, allow_pickle=True)
        self.noiseE1_data = np.load(noise_path1, allow_pickle=True)
        self.noiseE2_data = np.load(noise_path2, allow_pickle=True)
        self.data_E3 = np.load(noise_path3, allow_pickle=True)

        self.output_csv_path = output_csv_path
        self.proc_log = output_log_path
        self.output_folder = output_wav_path

        dict_params = convert_param_dict(all_params)

        self.min_gain = dict_params['bk_gain_range'][0]
        self.max_gain = dict_params['bk_gain_range'][1]
        self.min_offset = dict_params['bk_offset_range'][0]
        self.max_offset = dict_params['bk_offset_range'][1]
        self.bk_num = dict_params['bk_num']
        self.sr         = sr

        self.num_segments = 50
        self.num_min = 3
        self.noise_dist = 0.4
        self.min_offset_secs = 1
        self.max_offset_secs = 4

        # 

        if dict_params['sp_gain'] <= 0:
            self.speaker_flag = False
        else:
            self.speaker_flag = True

        if dict_params['ns_gain'] <= 0:
            self.noise_flag = False
        else:
            self.noise_flag = True

        if dict_params['bk_gain'] <= 0:
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
        offset_value(percentage) and converts output to int16
        """
        global counter_small

        signal_length = signal.shape[0]
        signal_offset = np.zeros_like(signal)
        others_current_audio = np.zeros((length_current_audio),
                                        dtype='float32')

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


        if np.isnan(audio_sum):
            msg = f">> NaN found in Audio. Offset: {str(offset_value)}\n"
            log_message(msg, self.proc_log, 'a', True)

        signal_result = gain_variation(signal_offset_norm, init_reduce = 0.4, factor_min=2.4, factor_max=2.5, verbose=False)
        # return signal_result*1.6
        # signal_result = self.gain_variation(signal_offset_norm, init_reduce = 0.4, factor_min=2.0, factor_max=2.4, verbose=False)
        # return signal_result*1.8
        # return signal_result
        return signal_result

    def noise_mod(self, noise, gain_value, length_current_audio):
        """
        Modifies the signal with a gain_value (percentage) and
        offset_value(percentage) and converts output to int16
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

        indx_others = random.randint(0, len(self.x_data)-1)
        others_audio = self.x_data[indx_others].astype('float32')
        others_audio = np.trim_zeros(others_audio)
        offset_value = gen_random_on_range(self.min_offset, self.max_offset)

        gain_value = gen_random_on_range(self.min_gain, self.max_gain)
        min_value = self.min_gain
        max_value = self.max_gain

        mu = (min_value + max_value) / 2   # mean
        sigma = (max_value - mu) / 2.5       # standard deviation

        # Generate a list of 1000 random numbers between min_value and max_value
        nums = []
        for i in range(1000):
            num = random.gauss(mu, sigma)
            while num < min_value or num > max_value:
                num = random.gauss(mu, sigma)
        
        gain_value = num

        audio_bk_ready = self.audio_mod(others_audio, gain_value,
                                       offset_value, length_current_audio,
                                       outmin_current, outmax_current,
                                       no_offset=no_offset)

        # audio_bk_ready  = self.norm_others_float32(others_audio,
        #                                                 gain_value = gain_value,
        #                                                 outmin = outmin_current,
        #                                                 outmax = outmax_current)
        if return_indx:
            return audio_bk_ready, indx_others
        else:
            return audio_bk_ready


    def prepare_noise_audio(self, current_audio_info, mic_dict, 
                            noise_gain_low, noise_gain_high,
                            dist_e1, dist_e2):

        length_current_audio, outmin_current, outmax_current = current_audio_info 

        if random.choice([True, False]):
            indx_others = random.randint(0, len(self.noiseE1_data)-1)
            # print(f'Random E1 int: {indx_others}')
            others_audio = self.noiseE1_data[indx_others].astype('float32')

            noise_x, noise_y, noise_z = mic_dict["mic_0"]
            noise_coords = [noise_x + dist_e1, noise_y + dist_e1, noise_z]
        else:
            indx_others = random.randint(0, len(self.noiseE2_data)-1)
            # indx_others = 1367
            # print(f'Random E2 int: {indx_others}')
            others_audio = self.noiseE2_data[indx_others].astype('float32')

            noise_x, noise_y, noise_z = mic_dict["mic_0"]
            noise_coords = [noise_x + dist_e2, noise_y + dist_e2, noise_z]

        others_audio = np.trim_zeros(others_audio)
        gain_value = gen_random_on_range(noise_gain_low, noise_gain_high)
        # print(f'gain: {gain_value}')

        audio_bk_ready = self.noise_mod(others_audio, gain_value,
                                       length_current_audio)

        return noise_coords, audio_bk_ready


    def gen_long_noise(self, current_audio_info, 
                            noise_gain_low, noise_gain_high):

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
            # print(f'gain: {gain_value}')

            noise_ready  = norm_noise_f32(others_audio,
                                                    gain_value)

            selected_arrays.append(noise_ready)
            total_elements += np.size(noise_ready)
            # print(f'len total: {length_current_audio} \t noise_gain: {gain_value} \t cnt: {np.size(noise_ready)}')

        return np.concatenate(selected_arrays)


    def create_long_audio_others(self, current_audio_info, num_segments = 20, num_min = 3, verbose = False):

        # Generate initial offset
        offset_value = gen_random_on_range(self.min_offset_secs*self.sr, self.max_offset_secs*self.sr)

        # print(f'Offset in secs: {round(offset_value/self.sr, 2)}')
        # Divide the 8 minutes into segments and randomly assign each segment to an audio file
        segments = np.linspace(0 + offset_value, num_min*60*self.sr + offset_value, num_segments + 1)
        # segments = np.linspace(0, num_min*60*self.sr, num_segments + 1)
        segment_limits = [(segments[i], segments[i+1]) for i in range(num_segments)]
        random.shuffle(segment_limits)

        index_list = list(range(len(self.x_data)))
        list_of_audios = []

        # Call the extend_audio function for each input audio file with its corresponding limit_lower and limit_upper values
        for idx in range(0, num_segments):

            # Apply mods to audio here:
            current_audio, random_idx = self.prepare_bk_audio(current_audio_info, no_offset = True, return_indx=True)

            
            limit_lower, limit_upper = segment_limits[idx]

            if verbose:
                print(f'\nIndex selected: {random_idx} \t {limit_lower}({round(limit_lower/self.sr, 2)}) - {limit_upper}({round(limit_upper/self.sr, 2)})')
            ext_audio_raw, ext_length, _ = extend_audio(current_audio, limit_lower, limit_upper, idx, num_min = num_min, offset_samples = offset_value)
            # ext_audio_raw, ext_length = self.extend_audio(current_audio, limit_lower, limit_upper, idx, num_min = num_min)

            ext_audio = ext_audio_raw[0:num_min*60*self.sr]

            # print(f'Current segment extended {idx}: {len(ext_audio_raw)}')
            list_of_audios.append(ext_audio)

        result_audio = sum_arrays(list_of_audios)
        return result_audio, ext_length


    def create_long_audio_main(self, num_segments = 20, num_min = 3, verbose = False):

        # Divide the 8 minutes into segments and randomly assign each segment to an audio file
        segments = np.linspace(0, num_min*60*self.sr, num_segments + 1)
        segment_limits = [(segments[i], segments[i+1]) for i in range(num_segments)]
        random.shuffle(segment_limits)
        GT_log = []

        index_list = list(range(len(self.x_data)))
        list_of_audios = []

        # Call the extend_audio function for each input audio file with its corresponding limit_lower and limit_upper values
        for idx in range(0, num_segments):

            # Choose a random element from the list
            random_idx = random.choice(index_list)
            index_list.remove(random_idx)

            raw_audio = self.x_data[random_idx]
            current_audio_name = self.x_data_names[random_idx]
            # Apply mods to audio here:
            current_audio = gain_variation(raw_audio, init_reduce = 0.6, factor_min=1.2, factor_max=1.5, verbose=False)
            
            limit_lower, limit_upper = segment_limits[idx]

            if verbose:
                print(f'\nIndex selected: {random_idx} \t {limit_lower} - {limit_upper} \t new length X_data: {len(index_list)}')
            packed_extended_audio_result = extend_audio(current_audio, limit_lower, limit_upper, idx, num_min=num_min)
            ext_audio = packed_extended_audio_result[0]
            ext_length = packed_extended_audio_result[1] 
            current_start_time, current_stop_time = packed_extended_audio_result[2] 
            GT_log.append((current_audio_name, current_start_time, current_stop_time))

            if len(ext_audio) > num_min*60*self.sr:
                print(f'ERROR!! Main audio length is {len(ext_audio)}')
                ext_audio = ext_audio[0:num_min*60*self.sr]

            list_of_audios.append(ext_audio)

        result_audio = sum_arrays(list_of_audios)
        return result_audio, ext_length, GT_log


    def long_distance_noise(self, n=60, t=3):
        audio_samples = random.sample(list(self.data_E3), n)

        # Calculate the total length of all audio samples
        total_audio_length = sum(len(audio) for audio in audio_samples)

        while(True):
            if (t*60*self.sr) < total_audio_length:
                audio_samples.pop()
                total_audio_length = sum(len(audio) for audio in audio_samples)
                # print(f'Value {n} is too big -> n-1')
                continue
            else:
                # Calculate the maximum length of silence to insert between audio samples
                max_silence_length = ((t * 60 * self.sr) - total_audio_length) // (n + 1)

                # Create the final audio array
                audio = np.zeros(t * 60 * self.sr)

                # Insert each audio sample into the final audio array at a random position with silence in between
                start = random.randint(0, max_silence_length)
                for audio_sample in audio_samples:
                    audio_length = len(audio_sample)
                    end = start + audio_length
                    audio[start:end] += audio_sample
                    start = end + random.randint(0, max_silence_length)

                # increased gain
                audio_final_gained = audio
                return audio_final_gained

    def sim_single_signal(self, input_signal, indx=0):
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
                                upper_right_table, table_margin = 0.3,
                                verbose = False)


        result_audio, ext_length ,GT_log = self.create_long_audio_main(num_segments = self.num_segments, num_min = self.num_min, verbose = False)

        GT_log = sorted(GT_log, key = lambda x: x[1])
        # specify filename for CSV file
        filename = self.output_csv_path.joinpath(f'GT_{indx}.csv')
        # open file for writing with tab delimiter
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')

            # write each tuple to a row in the file
            for audio_name, start, end in GT_log:
                writer.writerow([audio_name, round(start/self.sr,2), round(end/self.sr,2)])

        length_current_audio = len(result_audio)
        outmin_current = result_audio.min()
        outmax_current = result_audio.max()

        current_audio_info = [length_current_audio, outmin_current, outmax_current]

        harmonic_exciter = librosa.effects.harmonic(result_audio, margin=8)
        # result_audio += 0.8 * harmonic_exciter

        if self.speaker_flag:
            room.add_source(main_speaker_coords,
                            signal=result_audio*1.1)

        single_cfg = {'mic': [mic_dict['mic_0'][0], mic_dict['mic_0'][1]],
                    'main': [main_speaker_coords[0], main_speaker_coords[1]],
                    'others': [],
                    'regions': [(lower_left_point, upper_right_point),
                                (lower_left_table, upper_right_table),
                                tuple_outside_inside[0],
                                tuple_outside_inside[1]],
                    'room': [shoebox_vals[0], shoebox_vals[1]]}

        if self.bk_flag:
            for i in range(0, self.bk_num):

                # result_audio, _ = self.create_long_audio_others(current_audio_info, num_segments = int(np.ceil(self.num_segments*2.5)), num_min = self.num_min, verbose = False)
                result_audio, _ = self.create_long_audio_others(current_audio_info, num_segments = self.num_segments, num_min = self.num_min, verbose = False)

                # Randomly select the coordinates based on position
                rand_coordinates_other = gen_rand_coordinates(shoebox_vals[0], shoebox_vals[1],
                                                            lower_left_point,
                                                            upper_right_point, False)

                # if i == 1:
                #     output_path = f"others_long_audio_sample.wav"
                #     sf.write(output_path, result_audio, self.sr, subtype='FLOAT')

                # print(f'Length of {i}: {len(result_audio)}({round(len(result_audio)/self.sr, 2)})')

                single_cfg['others'].append((rand_coordinates_other[0], rand_coordinates_other[1]))

                room.add_source(rand_coordinates_other,
                                signal=result_audio)

        if self.noise_flag:
            noise_audio_long = self.gen_long_noise(current_audio_info, 0.9, 1.0)
                                        # 0.7, 0.8, 
                                        
            # output_path = f"long_noise_X.wav"
            # sf.write(output_path, noise_audio_long, self.sr, subtype='FLOAT')

            # print(f'Length of noise: {len(noise_audio_long)}({round(len(noise_audio_long)/self.sr, 2)})')

            noise_x, noise_y, noise_z = mic_dict["mic_0"]
            noise_coords = [noise_x + self.noise_dist, noise_y + self.noise_dist, noise_z]

            single_cfg['noise'] = (noise_coords[0], noise_coords[1])
            
            room.add_source(noise_coords,
                            signal = noise_audio_long)
        
        # Distance E3 Noise
        rand_coordinates_noise_E3 = gen_rand_coordinates(shoebox_vals[0], shoebox_vals[1],
                                                          lower_left_point,
                                                          upper_right_point, False)

        long_noise_E3 = self.long_distance_noise(n=45, t = self.num_min)

        # output_path = f"long_noise_E3_X.wav"
        # sf.write(output_path, long_noise_E3, self.sr, subtype='FLOAT')

        room.add_source(rand_coordinates_noise_E3,
                        signal = long_noise_E3)

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
        # for indx in range(0, self.x_data.shape[0]):
        for indx in range(0, 2):
            single_signal = self.x_data[indx]
            single_signal_trimmed = np.trim_zeros(single_signal)

            if single_signal_trimmed.dtype.kind != 'f':
                raise TypeError("'dtype' must be a floating point type")
                
                
            # single_signal_trimmed = librosa.effects.time_stretch(single_signal_trimmed, 1.1)
                
            single_x_DA = self.sim_single_signal(single_signal_trimmed
                                                 .astype('float32'),
                                                 indx)

            

            single_x_DA_trimmed = eliminate_noise_start_ending(single_x_DA, 0.00001)

            # Calculate the filter coefficients
            nyquist_freq = 0.5 * self.sr
            cutoff_freq = 60.0 / nyquist_freq
            b, a = signal.butter(1, cutoff_freq, 'highpass', analog=False, output='ba')

            # Apply the filter to the audio signal
            filtered_audio = signal.filtfilt(b, a, single_x_DA_trimmed)

            new_name_wav = f'DA_long_{indx}.wav'
            output_path_wav = self.output_folder.joinpath(new_name_wav) 
            sf.write(str(output_path_wav), filtered_audio, 16000, subtype='FLOAT')

            # new_name_wav = f'DA_long-{indx}_PCM16.wav'
            # output_path_wav = self.output_folder.joinpath(new_name_wav) 
            # sf.write(str(output_path_wav), single_x_DA_trimmed, 16000, subtype='PCM_16')

            if indx%20 == 0:

                current_time_100a = time.process_time()
                time_100a = current_time_100a - prev_time

                msg = f"{indx} | time for 100 audios: {time_100a:.2f}\n"
                log_message(msg, self.proc_log, 'a', True)
                prev_time = current_time_100a

        print("These number of small audios:" + str(counter_small))
        # return self.x_data_DA