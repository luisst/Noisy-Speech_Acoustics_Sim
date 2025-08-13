all_params = { # Set to 0 to have only bk_speakers
                'sp_gain' :     (0.8, True), # Gain of active speaker 
                'rnd_offset_secs' : (1, False), # max silence between speech segments 
                'bk_num' :      (6, True), # bk_speakers_number
                'output_samples_num' : (1, False), # Number of long audios generated
                'length_minutes' :     (3, False), # Length of output audios (min)
                'pattern_flag': (True, False), # Use pattern for conversation
                'double_talk_flag': (False, False), # Use double talk simulation
                'ordered_sp_flag': (False, False), # Use ordered speech segments

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
run_name = 'g5-easy4'


all_params = { # Set to 0 to have only bk_speakers
                'sp_gain' :     (0.8, True), # Gain of active speaker 
                'rnd_offset_secs' : (1, False), # max silence between speech segments 
                'bk_num' :      (6, True), # bk_speakers_number
                'output_samples_num' : (1, False), # Number of long audios generated
                'length_minutes' :     (3, False), # Length of output audios (min)
                'pattern_flag': (True, False), # Use pattern for conversation
                'double_talk_flag': (False, False), # Use double talk simulation
                'ordered_sp_flag': (False, False), # Use ordered speech segments

                'gain_var_flag' : (True, False), # Apply gain_variation 
                'sp_init_reduce' : (0.7, False), # Init reduction before inner gain_variation
                'sp_inner_gain_range':([1.2, 1.3], False), # Gain variation inside the sample
                'sp_inner_dur_range' : ([0.4, 0.6], False), # Min & max percentage of gain inside sample

                # Set to 0 to remove background speakers
                'bk_gain' :     (0.6, True), # Gain of all bk speakers together 
                'bk_num_segments' : (60, True), # Number of bk samples in the final output
                'bk_gain_range' :     ([0.3, 0.4], True), # Gain range bk speakers 
                'bk_init_reduce' : (0.5, False), # Init reduction before inner gain_variation
                'bk_inner_gain_range' : ([2,4, 2,5], False), # Gain variation inside bk sample
                'bk_inner_dur_range' : ([0.2, 0.4], False), # Duration percentage for gain inside bk sample
                'bk_ext_offset_range' : ([1, 4], False), # Offset of extend bk audio (secs)

                # Set to 0 to remove noises
                'ns_gain' :           (0.6, True), 
                'ns_gain_away' :      (0.6, False),
                'ns_gain_range' :     ([0.4, 0.6], True),
                'ns_number_samples':   (50, False), # Number of long distance noise samples

                # Store in log folder audios from inner stages
                'debug_store_audio':   (True, False) 
                }
run_name = 'TTS4-med'


all_params = { # Set to 0 to have only bk_speakers
                'sp_gain' :     (0.7, True), # Gain of active speaker 
                'rnd_offset_secs' : (1, False), # max silence between speech segments 
                'bk_num' :      (6, True), # bk_speakers_number
                'output_samples_num' : (1, False), # Number of long audios generated
                'length_minutes' :     (3, False), # Length of output audios (min)
                'pattern_flag': (True, False), # Use pattern for conversation
                'double_talk_flag': (False, False), # Use double talk simulation
                'ordered_sp_flag': (False, False), # Use ordered speech segments

                'gain_var_flag' : (True, False), # Apply gain_variation 
                'sp_init_reduce' : (0.7, False), # Init reduction before inner gain_variation
                'sp_inner_gain_range':([1.2, 1.3], False), # Gain variation inside the sample
                'sp_inner_dur_range' : ([0.4, 0.6], False), # Min & max percentage of gain inside sample

                # Set to 0 to remove background speakers
                'bk_gain' :     (0.6, True), # Gain of all bk speakers together 
                'bk_num_segments' : (50, True), # Number of bk samples in the final output
                'bk_gain_range' :     ([0.4, 0.5], True), # Gain range bk speakers 
                'bk_init_reduce' : (0.5, False), # Init reduction before inner gain_variation
                'bk_inner_gain_range' : ([2,4, 2,5], False), # Gain variation inside bk sample
                'bk_inner_dur_range' : ([0.2, 0.4], False), # Duration percentage for gain inside bk sample
                'bk_ext_offset_range' : ([1, 4], False), # Offset of extend bk audio (secs)

                # Set to 0 to remove noises
                'ns_gain' :           (0.7, True), 
                'ns_gain_away' :      (0.7, False),
                'ns_gain_range' :     ([0.4, 0.7], True),
                'ns_number_samples':   (50, False), # Number of long distance noise samples

                # Store in log folder audios from inner stages
                'debug_store_audio':   (True, False) 
                }
run_name = 'TTS4-hard'