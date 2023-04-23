import sys
import os
import shutil

# ### Allan setup (table)
# shoebox_vals = [1.6, 1.4, 1.2]

# mic_dict = {"mic_0": [0.8, 0.7, 0],
#             "mic_1": [0.81, 0.7, 0]}

# src_dict = {"src_0": [0, 0.622, 0.33], # d = 
#             "src_1": [0.381, 0, 0.432], # d = 
#             "src_2": [1, 1.35, 0.33], # d = 
#             "src_3": [0.762, 0, 0.3]} # d =


### Allan setup (extended by +4 +4)
shoebox_vals = [5.6, 5.4, 2.2]

mic_dict = {"mic_0": [2.8, 2.7, 0],
            "mic_1": [2.81, 2.7, 0]}

src_dict = {"src_0": [2, 2, 0.33], # d = 
            "src_1": [2.5, 3.6, 0.432], # d = 
            "src_2": [3.4, 2.9, 0.33]} # d = 

# ### Phuong setup (table)
# shoebox_vals = [1.6, 1.4, 1.2]

# mic_dict = {"mic_0": [0.8, 0.7, 0],
#             "mic_1": [0.81, 0.7, 0]}

# src_dict = {"src_0": [0.05, 0.533, 0.355], # d = 
#             "src_1": [0.025, 0.92, 0.33], # d = 
#             "src_2": [0.685, 0, 0.381], # d = 
#             "src_3": [0.431, 1.295, 0.33]} # d = 
# ### 0.9144, 1.244, 0.25

# ### Phuong setup (extended +4 +4)
# shoebox_vals = [5.6, 5.4, 2.2]

# mic_dict = {"mic_0": [2.8, 2.7, 0],
#             "mic_1": [2.81, 2.7, 0]}

# src_dict = {"src_0": [2.05, 2.533, 0.355], # d = 
#             "src_1": [2.025, 2.92, 0.33], # d = 
#             "src_2": [2.685, 2, 0.381], # d = 
#             "src_3": [2.431, 3.295, 0.33]} # d = 
# ### 2.9144, 3.244, 0.25

all_cfg = {"AllanTable":[shoebox_vals, mic_dict, src_dict] }
single_cfg = [shoebox_vals, mic_dict, src_dict]

# shoebox_general = [5.5, 5.2, 2.2]
# mic_dict_general = {"mic_0": [2.75, 2.6, 0],
#                     "mic_1": [2.85, 2.6, 0]}

# src_list = {"src_0": [2.508, 2,000, 0.355],
#             "src_0": [2.482, 2.705, 0.357],
#             "src_0": [2.000, 2.622, 0.33],
#             "src_0": [2.050, 2.533, 0.355],



#             "src_1": [2, 2.787, 0.432],
#             "src_2": [2, 3.219, 0.469],
#             "src_3": [2.787, 3.371, 0.457],
#             }

# src_dict_pred_ext  = {
#             "src_1": [2, 2.705, 0.516], # d =
#             "src_2": [2.177, 3.328, 0.510], # d =
#             "src_3": [2.898, 3.328, 0.334]} # d =

# src_dict = {
#             "src_1": [2.381, 2, 0.432], # d = 
#             "src_2": [3, 3.4, 0.33], # d = 
#             "src_3": [2.762, 2, 0.3]} # d =
# src_dict = {
#             "src_1": [2.025, 2.92, 0.33], # d = 
#             "src_2": [2.685, 2, 0.381], # d = 
#             "src_3": [2.431, 3.295, 0.33]} # d = 

# multiple_speakers_cfg = {"coverage2022": {shoebox_general, mic_dict_general, src_list}}


# all_cfg = {"pred_tab": [shoebox_vals_pred_tab, mic_dict_pred_tab, src_dict_pred_tab],
#            "pred_ext": [shoebox_vals_pred_ext, mic_dict_pred_ext, src_dict_pred_ext],
#            "bs_tab": [shoebox_vals_bs_tab, mic_dict_bs_tab, src_dict_bs_tab],
#            "bs_ext": [shoebox_vals_bs_ext, mic_dict_bs_ext, src_dict_bs_ext],
#            "gt_tab":[shoebox_vals_gt_tab, mic_dict_gt_tab, src_dict_gt_tab],
#            "gt_ext":[shoebox_vals_gt_ext, mic_dict_gt_ext, src_dict_gt_ext]}

i_list = [0,1,2,3]

abs_coeff = 0.7

fs = 16000
