import sys
import os
import shutil
import random
import numpy as np
import math
import pdb

def find_center(lower_left_point, upper_right_point):
    dist_x = upper_right_point[0] - lower_left_point[0]
    dist_y = upper_right_point[1] - lower_left_point[1]

    if dist_x < 0 or dist_y < 0:
        print(f'Error! find center is negative {dist_x} - {dist_y}')
        pdb.set_trace()

    return [dist_x/2 + lower_left_point[0], dist_y/2 + lower_left_point[1]]

def place_table_middle(lower_left_point, upper_right_point, table_x, table_y):
    center_inner_square = find_center(lower_left_point, upper_right_point)

    lower_left_table = (center_inner_square[0] - table_x/2, center_inner_square[1] - table_y/2)
    upper_right_table = (center_inner_square[0] + table_x/2, center_inner_square[1] + table_y/2)
    return lower_left_table, upper_right_table


def gen_rand_coordinates(room_x, room_y, lower_left_point, upper_right_point, verbose = False):

    rand_z = random.randint(int(0.6*100), int(1.8*100))/100

    while True:
        # Randomly select coord from outside the inner square
        rand_x = random.randint(0, int(room_x*100) -1)/100
        rand_y = random.randint(0, int(room_y*100) -1)/100

        # print(f' rand {rand_x} - {rand_y}')

        # Discard if inside inner square
        if lower_left_point[0] <= rand_x <= upper_right_point[0]:
            if verbose:
                print(f'x: {rand_x} \t inside the inner square: [{lower_left_point[0]} - {upper_right_point[0]}]')
            continue
        elif lower_left_point[1] <= rand_y <= upper_right_point[1]:
            if verbose:
                print(f'y: {rand_y} \t inside the inner square: [{lower_left_point[1]} - {upper_right_point[1]}]')
            continue
        else:
            if verbose:
                print(f'Accepted! Outside the area: {rand_x} - {rand_y}')
            return [rand_x, rand_y, rand_z]


def gen_rand_table_coords(lower_left_point, upper_right_point, 
                          lower_left_table, upper_right_table, 
                          table_margin = 0.2,
                          verbose = False):

    rand_z = random.randint(int(0.8*100), int(1.4*100))/100

    # Define points A and B
    pA_x = lower_left_table[0] - table_margin
    if pA_x < 0:
        pA_x = 0
    
    pA_y = lower_left_table[1] - table_margin
    if pA_y < 0:
        pA_y = 0
    
    point_A = (pA_x, pA_y)
    point_B = (lower_left_table[0] + table_margin, lower_left_table[1] + table_margin)

    # Define points C and D

    pD_x = upper_right_table[0] + table_margin
    if pD_x > upper_right_point[0]:
        pD_x = upper_right_point[0]
    
    pD_y = upper_right_table[1] + table_margin
    if pD_y > upper_right_point[1]:
        pD_y = upper_right_point[1]

    point_C = (upper_right_table[0] - table_margin, upper_right_table[1] - table_margin)
    point_D = (pD_x, pD_y)

    while True:
        # Randomly select coord from outside the inner square
        rand_x = random.randint(int(point_A[0]*100), int(point_D[0]*100) -1)/100
        rand_y = random.randint(int(point_A[1]*100), int(point_D[1]*100) -1)/100

        if rand_x < 0 or rand_y < 0:
            print(f'ERROR! Rand is negative: {rand_x} \t {rand_y}')
            pdb.set_trace()

        if verbose:
            print(f'Candidates rand {rand_x} - {rand_y}')

        if ((point_B[0] < rand_x < point_C[0]) and (point_B[1] < rand_y < point_C[1])):
            if verbose:
                print('the candidate point is too much in the center of the table!')
            continue
        else:
            if verbose:
                print(f'Accepted! {rand_x} - {rand_y}')
        
            # print(f'x:{point_B[0]} <= {rand_x} <= {point_C[0]}  AND  y:{point_B[1]} <= {rand_y} <= {point_C[1]}')
            return [rand_x, rand_y, rand_z], [(point_A, point_D), (point_B, point_C)]



# ### Define all room dimensions
# room_x = 10
# room_y = 11

# ### Define the inner square
# lower_left_point = (3., 2.)
# upper_right_point = (6.,5.)

### Define all room dimensions
room_x = 15
room_y = 13

### Define the inner square
lower_left_point = (3., 2.)
upper_right_point = (6.,5.)

### Define table dimensions
# Place it in the center of the inner square
lower_left_table, upper_right_table = place_table_middle(lower_left_point, 
                                                         upper_right_point, 
                                                         1.6, 1.4)

## Place microphone in the center of inner sqaure
mic_point_x, mic_point_y = find_center(lower_left_table, upper_right_table)

for i in range(0,20):
    print(f'\n\n')
#     gen_rand_coordinates(room_x, room_y, lower_left_point, upper_right_point, True)

# print(f'\n\n\nNow the speaker:')
    gen_rand_table_coords(lower_left_point, upper_right_point, 
                            lower_left_table, upper_right_table, 
                            table_margin = 0.2,
                            verbose = True)

# corners = np.array([[2.,0.], [2.,3.], [0.,3.],
# [0.,8.],[14.,8.],[14.,3.],[12.,3.],[12.,0.]]).T # [x,y]
# room = pra.Room.from_corners(corners)
# room.extrude(2.5)


shoebox_vals = [room_x, room_y, 2.5]

abs_coeff = 0.6

fs = 16000
sr = fs

mic_dict = {"mic_0": [mic_point_x, mic_point_y, 0.75],
            "mic_1": [mic_point_x + 0.1, mic_point_y, 0.75]}

cfg_info = [shoebox_vals, mic_dict ,
            lower_left_point, upper_right_point,
            lower_left_table, upper_right_table,
            abs_coeff, fs]

