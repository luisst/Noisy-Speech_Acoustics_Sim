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

from cfg_pyroom import abs_coeff, fs, i_list

def log_message(msg, log_file, mode, both=True):
    '''Function that prints and/or adds to log'''
    #Always log file
    with open(log_file, mode) as f:
        f.write(msg)

    #If {both} is true, print to terminal as well
    if both:
        print(msg, end='')

class DAwithPyroom2(object):
    """
    Class for audio simulation using pyroom.
    input signal + 4 random crosstalk + background noise
    """

    def __init__(self, noise_path,
                 room_cfg, proc_log, DA_number, float_flag=True, ds_name='0',
                 total_ite=1, num_ite = 0):
        """
        Start the class with the dataset path and turn the float32 flag for
        output format, False for int16 format
        """
        self.noise_data = np.load(noise_path, allow_pickle=True)
        self.float_flag = float_flag
        self.ds_name = ds_name
        self.room_cfg = room_cfg
        self.proc_log = proc_log

        # Numpy output array according to the desired output
        self.x_data_DA = []
        self.num_ite = num_ite
        self.DA_number = DA_number

        global counter_small
        counter_small = 0

    def gen_random_on_range(self, lower_value, max_value):
        """
        generates a random value between lower_value and max_value.
        """
        return round(lower_value + random.random()*(max_value - lower_value),
                     2)


    def eliminate_noise_ending(self, signal):
        """
        Count using a non-optimized python alg the number of zeros
        at the end of the numpy array
        """

        # real_length is initialized with the total length of the signal
        real_length = int(len(signal))
        while abs(signal[real_length - 1]) < 0.0001:
            real_length = real_length - 1

        signal_trimmed = signal[0:real_length]
        return signal_trimmed

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

        return signal_offset_norm

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
        signal_offset_norm  = self.norm_others_float32(noise_current_audio,
                                          gain_value = gain_value,
                                          outmin = outmin_current,
                                          outmax = outmax_current)

        sum_noise = np.sum(signal_offset_norm)
        if np.isnan(sum_noise):
            msg = f">> NaN found in Noise | Index: {self.num_ite}\n"
            log_message(msg, self.proc_log, 'a', True)

        return signal_offset_norm


    def sim_same_speaker(self, input_signal, position=0, indx=0, noise_flag = False):
        """
        Pyroomacoustics simulation manually.
        Input signal as .astype('float32') from a NPY.
        Noise from AudioSet or AOLME, select this at class init ABOVE.
        """
        length_current_audio = len(input_signal)
        outmin_current = input_signal.min()
        outmax_current = input_signal.max()

        if noise_flag:        
            indx_noise_4 = random.randint(0, len(self.noise_data)-1)

            noise_audio4 = self.noise_data[indx_noise_4].astype('float32')
            # noise_audio4 = self.noise_data[indx_noise_4, :].astype('float32')

            noise_audio4 = np.trim_zeros(noise_audio4)

            gain_value4 = self.gen_random_on_range(0.08, 0.11)


            audio_offset4 = self.noise_mod(noise_audio4, gain_value4,
                                            length_current_audio,
                                            outmin_current, outmax_current)

        audio_original = input_signal[0:length_current_audio]

        # Create 3D room and add sources
        room = pra.ShoeBox(self.room_cfg[0],
                           fs=fs,
                           absorption=abs_coeff,
                           max_order=12)

        src_dict = self.room_cfg[2]

        # print(f'Current position {position}')
        room.add_source(src_dict["src_{}".format(i_list[position])],
                        signal=audio_original)
        if noise_flag:
            room.add_source([4, 4, 2],
                            signal=audio_offset4)

        # Define microphone array
        mic_dict = self.room_cfg[1]
        R = np.c_[mic_dict["mic_0"], mic_dict["mic_0"]]
        room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

        # Compute image sources
        room.image_source_model()
        room.simulate()

        # Simulate audio
        raw_sim_audio = room.mic_array.signals[0, :]

        sum_noise = np.sum(raw_sim_audio)
        if np.isnan(sum_noise):
            msg = f">>> NaN found after pyroom | Index: {self.num_ite}\n"
            msg += f"{indx} -- {self.ds_name}. noise {indx_noise_4}. Len {length_current_audio}\n"
            log_message(msg, self.proc_log, 'a', True)

        return raw_sim_audio


    def sim_interviews_dataset(self, list_audios_pth, noise_flag = True):
        # global counter_small
        prev_time = time.process_time()
        for indx in range(0, len(list_audios_pth)):
            single_signal, samplerate = sf.read(list_audios_pth[indx]) 
            single_signal_trimmed = np.trim_zeros(single_signal)

            if single_signal_trimmed.dtype.kind != 'f':
                raise TypeError("'dtype' must be a floating point type")
                
                
            # single_signal_trimmed = librosa.effects.time_stretch(single_signal_trimmed, 1.1)

            single_x_DA = self.sim_same_speaker(single_signal_trimmed
                                                 .astype('float32'),
                                                 position=random.choice([0,1,2]),
                                                 indx=indx,
                                                 noise_flag = noise_flag)


            single_x_DA_trimmed = self.eliminate_noise_start_ending(single_x_DA, 0.00001)
            self.x_data_DA.append(single_x_DA_trimmed)


            current_percentage = 100*(self.num_ite/len(list_audios_pth))

            if indx%10 == 0:

                current_time_10a = time.process_time()
                time_10a = current_time_10a - prev_time

                msg = f"DA{self.DA_number}: {indx} - {self.num_ite} - {current_percentage:.2f}% | time for 100 audios: {time_10a:.2f}\n"
                log_message(msg, self.proc_log, 'a', True)
                prev_time = current_time_10a

            self.num_ite += 1

        print("These number of small audios:" + str(counter_small))
        return self.x_data_DA, self.num_ite