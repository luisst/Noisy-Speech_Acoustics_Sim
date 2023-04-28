    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 10:48:43 2021

@author: luis
"""

import random
import numpy as np
import pyroomacoustics as pra
import time
import librosa
import inspect
import pdb
import soundfile as sf

from cfg_pyroom_VAD import cfg_info, gen_rand_coordinates, gen_rand_table_coords
from plot_configuration_room import plot_single_configuration


def log_message(msg, log_file, mode, both=True):
    '''Function that prints and/or adds to log'''
    #Always log file
    with open(log_file, mode) as f:
        f.write(msg)

    #If {both} is true, print to terminal as well
    if both:
        print(msg, end='')

class DAwithPyroom(object):
    """
    Class for audio simulation using pyroom.
    input signal + 4 random crosstalk + background noise
    """

    def __init__(self, input_path, noise_path1, noise_path2, proc_log, noise_flag = False,
                 min_gain = 0.15, max_gain = 0.25,
                 min_offset = -0.4, max_offset = 0.4, bk_num = 10):
        """
        Start the class with the dataset path and turn the float32 flag for
        output format, False for int16 format
        """

        self.x_data = np.load(input_path, allow_pickle=True)
        self.noiseE1_data = np.load(noise_path1, allow_pickle=True)
        self.noiseE2_data = np.load(noise_path2, allow_pickle=True)
        self.proc_log = proc_log
        self.noise_flag = noise_flag

        self.min_gain = min_gain
        self.max_gain = max_gain
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.bk_num = bk_num


        # Numpy output array according to the desired output
        self.x_data_DA = []

        global counter_small
        counter_small = 0


    def gen_random_on_range(self, lower_value, max_value):
        """
        generates a random value between lower_value and max_value.
        """
        return round(lower_value + random.random()*(max_value - lower_value),
                     2)


    def eliminate_noise_start_ending(self, signal, th):
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

    def norm_noise_f32(self, noise_current_audio, gain_value = 1):
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
            log_message(msg, self.proc_log, 'a', True)
        
        nm_sum = np.sum(audio_gained)
        if np.isnan(nm_sum):
            try_inspect = inspect.stack()
            msg = f">> NAN in audio_gained\nVin: {vmin} | {vmax} --- Vout: {vmin_gained} | {vmax_gained}\
                function {try_inspect[2].function} | code_context {try_inspect[2].code_context} \
                    || function {try_inspect[1].function} | context{try_inspect[1].code_context}\n"
            log_message(msg, self.proc_log, 'a', True)

        return audio_gained


    def norm_others_float32(self, audio_float32, gain_value, outmin, outmax):
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
            log_message(msg, self.proc_log, 'a', True)
        
        nm_sum = np.sum(audio_gained)
        if np.isnan(nm_sum):
            try_inspect = inspect.stack()
            msg = f">> NAN in audio_gained\nVin: {vmin} | {vmax} --- Vout: {vmin_gained} | {vmax_gained}\
                function {try_inspect[2].function} | code_context {try_inspect[2].code_context} \
                    || function {try_inspect[1].function} | context{try_inspect[1].code_context}\n"
            log_message(msg, self.proc_log, 'a', True)



        return audio_gained


    def gain_variation(self, original_audio, init_reduce = 0.6, factor_min=2.0, factor_max=2.5, verbose=False):
        # Reduce the gain to 30%
        audio_data = init_reduce * original_audio

        # Define the minimum and maximum duration of the gain increase
        min_duration = int(len(audio_data) * 0.2)
        max_duration = int(len(audio_data) * 0.4)

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



    def audio_mod(self, signal, gain_value, offset_value,
                  length_current_audio, outmin_current, outmax_current):
        """
        Modifies the signal with a gain_value (percentage) and
        offset_value(percentage) and converts output to int16
        """
        global counter_small

        signal_length = signal.shape[0]
        signal_offset = np.zeros_like(signal)
        others_current_audio = np.zeros((length_current_audio),
                                        dtype='float32')

    
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
        
        if np.sum(others_current_audio) == 0:
            msg = f">> Audio offset failed!! Index: {self.num_ite}\n"
            log_message(msg, self.proc_log, 'a', True)

        # Apply gain value and convert to required output format
        signal_offset_norm  = self.norm_others_float32(others_current_audio,
                                                  gain_value = gain_value,
                                                  outmin = outmin_current,
                                                  outmax = outmax_current)

        audio_sum = np.sum(signal_offset_norm)
        if audio_sum == 0:
            if factor1 >= length_current_audio:
                counter_small = counter_small + 1
                msg = f"> small audio encountered: counter {counter_small} | Index: {self.num_ite}\n"
                log_message(msg, self.proc_log, 'a', True)
            else:
                msg = f"> Cero was found. Signal_offset {offset_value}. Factor {factor1} - {factor2} \n|"
                msg += f"INDEXS signal_offset {factor1} - {(signal_length - factor2)}. signal {factor2} - {(signal_length - factor1)}\n| Index: {self.num_ite}\n"
                log_message(msg, self.proc_log, 'a', True)


        if np.isnan(audio_sum):
            msg = f">> NaN found in Audio. Offset: {str(offset_value)} | Index: {self.num_ite}\n"
            log_message(msg, self.proc_log, 'a', True)

            pdb.set_trace()

        # signal_result = self.gain_variation(signal_offset_norm, init_reduce = 0.4, factor_min=2.4, factor_max=2.5, verbose=False)
        # return signal_result*1.6
        signal_result = self.gain_variation(signal_offset_norm, init_reduce = 0.4, factor_min=2.0, factor_max=2.4, verbose=False)
        return signal_result*1.8

    def noise_mod(self, noise, gain_value, length_current_audio,
                  outmin_current, outmax_current):
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
        signal_offset_norm  = self.norm_noise_f32(noise_current_audio,
                                          gain_value)

        sum_noise = np.sum(signal_offset_norm)
        if np.isnan(sum_noise):
            msg = f">> NaN found in Noise | Index: {self.num_ite}\n"
            log_message(msg, self.proc_log, 'a', True)

        return signal_offset_norm

    def prepare_bk_audio(self, current_audio_info):

        length_current_audio, outmin_current, outmax_current = current_audio_info 

        indx_others = random.randint(0, len(self.x_data)-1)
        others_audio = self.x_data[indx_others].astype('float32')
        others_audio = np.trim_zeros(others_audio)
        offset_value = self.gen_random_on_range(self.min_offset, self.max_offset)
        gain_value = self.gen_random_on_range(self.min_gain, self.max_gain)

        audio_bk_ready = self.audio_mod(others_audio, gain_value,
                                       offset_value, length_current_audio,
                                       outmin_current, outmax_current)

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
        gain_value = self.gen_random_on_range(noise_gain_low, noise_gain_high)
        # print(f'gain: {gain_value}')

        audio_bk_ready = self.noise_mod(others_audio, gain_value,
                                       length_current_audio,
                                       outmin_current, outmax_current)


        return noise_coords, audio_bk_ready


    def sim_single_signal(self, input_signal, indx=0):
        """
        Pyroomacoustics simulation with 3 random other audios
        from the same x_data + 1 noise from AudioSet
        """

        shoebox_vals, mic_dict, lower_left_point, upper_right_point, \
        lower_left_table, upper_right_table, abs_coeff, fs = cfg_info

        # print(f' shoebox_vals: {shoebox_vals}')
        # print(f' inner square points {lower_left_point} - {upper_right_point}')
        # print(f' table points: {lower_left_table} - {upper_right_table}')

        # Create 3D room and add sources
        room = pra.ShoeBox(shoebox_vals,
                           fs=fs,
                           absorption=abs_coeff,
                           max_order=12)

        length_current_audio = len(input_signal)
        outmin_current = input_signal.min()
        outmax_current = input_signal.max()

        current_audio_info = [length_current_audio, outmin_current, outmax_current]

        audio_original = input_signal[0:length_current_audio]

        # randomly select the coordinates for the main speaker:
        main_speaker_coords, tuple_outside_inside = gen_rand_table_coords(lower_left_point,
                                upper_right_point,  lower_left_table,
                                upper_right_table, table_margin = 0.3,
                                verbose = False)

        audio_gained = self.gain_variation(audio_original, init_reduce = 0.6, factor_min=1.2, factor_max=1.5, verbose=False)

        # room.add_source(main_speaker_coords,
        #                 signal=audio_gained)

        # print(f'First AD: {tuple_outside_inside[0]} \t BC: {tuple_outside_inside[1]}')
        # print(f'Main speaker: {main_speaker_coords}')

        single_cfg = {'mic': [mic_dict['mic_0'][0], mic_dict['mic_0'][1]],
                    'main': [main_speaker_coords[0], main_speaker_coords[1]],
                    'others': [],
                    'regions': [(lower_left_point, upper_right_point),
                                (lower_left_table, upper_right_table),
                                tuple_outside_inside[0],
                                tuple_outside_inside[1]],
                    'room': [shoebox_vals[0], shoebox_vals[1]]}

        for i in range(0, self.bk_num):
            # Randomly select audio to place as background noise
            ready_audio = self.prepare_bk_audio(current_audio_info)

            # Randomly select the coordinates based on position
            rand_coordinates_other = gen_rand_coordinates(shoebox_vals[0], shoebox_vals[1],
                                                          lower_left_point,
                                                          upper_right_point, False)


            single_cfg['others'].append((rand_coordinates_other[0], rand_coordinates_other[1]))

            room.add_source(rand_coordinates_other,
                            signal=ready_audio)

        if self.noise_flag:
            noise_coords, noise_audio = self.prepare_noise_audio(current_audio_info, mic_dict, 
                                        # 0.7, 0.8, 
                                        0.8, 0.9, 
                                        0.3, 0.4)
            
            single_cfg['noise'] = (noise_coords[0], noise_coords[1])
            
            room.add_source(noise_coords,
                            signal = noise_audio)

        # Define microphone array
        R = np.c_[mic_dict["mic_0"], mic_dict["mic_1"]]
        room.add_microphone_array(pra.MicrophoneArray(R, room.fs))


        # plot_single_configuration(single_cfg, filename=f'minitest_{indx}.png')   

        # Compute image sources
        room.image_source_model()
        room.simulate()

        # Simulate audio
        sim_audio = room.mic_array.signals[0, :]

        sum_noise = np.sum(sim_audio)
        if np.isnan(sum_noise):
            msg = f">>> NaN found after pyroom | Index: {self.num_ite}\n"
            msg += f"{indx} -- Len {length_current_audio}\n"
            log_message(msg, self.proc_log, 'a', True)


        return sim_audio*1.


    def sim_vad_dataset(self):
        # global counter_small

        prev_time = time.process_time()

        for indx in range(0, self.x_data.shape[0]):
        # for indx in range(0, 20):
            single_signal = self.x_data[indx]
            single_signal_trimmed = np.trim_zeros(single_signal)

            if single_signal_trimmed.dtype.kind != 'f':
                raise TypeError("'dtype' must be a floating point type")
                
                
            # single_signal_trimmed = librosa.effects.time_stretch(single_signal_trimmed, 1.1)
                
            single_x_DA = self.sim_single_signal(single_signal_trimmed
                                                 .astype('float32'),
                                                 indx)

            single_x_DA_trimmed = self.eliminate_noise_start_ending(single_x_DA, 0.00001)
            self.x_data_DA.append(single_x_DA_trimmed)

            if indx%100 == 0:

                current_time_100a = time.process_time()
                time_100a = current_time_100a - prev_time

                msg = f"{indx} | time for 100 audios: {time_100a:.2f}\n"
                log_message(msg, self.proc_log, 'a', True)
                prev_time = current_time_100a

        print("These number of small audios:" + str(counter_small))
        return self.x_data_DA
