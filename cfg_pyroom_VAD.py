import random
import numpy as np
import pdb


def find_center(lower_left_point, upper_right_point):
    dist_x = upper_right_point[0] - lower_left_point[0]
    dist_y = upper_right_point[1] - lower_left_point[1]

    if dist_x < 0 or dist_y < 0:
        print(f'Error! find center is negative {dist_x} - {dist_y}')
        pdb.set_trace()

    return [dist_x/2 + lower_left_point[0], dist_y/2 + lower_left_point[1]]


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


def gen_room_config():

    ### Define all room dimensions
    room_x = random.randint(7, 17)  # Room width
    room_y = random.randint(7, 15)  # Room height
    room_z = random.randint(2, 4)  # Room height

    mic_point_z = random.randint(1, 2)*0.5  # Microphone height

    ### Define the inner square
    # original points (3., 2.)
    lower_left_point = random.uniform(2.5, 4.0), random.uniform(1.5, 3.0)  # Lower left point of inner square

    # original points (6.,5.)
    upper_right_point = random.uniform(5.5, 6.5), random.uniform(4.5, 5.5)

    table_x = random.uniform(0.8, 1.2)  # Table width
    table_y = random.uniform(0.8, 1.8)  # Table 

    center_inner_square = find_center(lower_left_point, upper_right_point)

    lower_left_table = (center_inner_square[0] - table_x/2, center_inner_square[1] - table_y/2)
    upper_right_table = (center_inner_square[0] + table_x/2, center_inner_square[1] + table_y/2)

    ## Place microphone in the center of inner sqaure
    mic_point_x, mic_point_y = find_center(lower_left_table, upper_right_table)

    shoebox_vals = [room_x, room_y, room_z]

    abs_coeff = random.uniform(0.4, 0.9)  # Absorption coefficient 

    fs = 16000

    mic_dict = {"mic_0": [mic_point_x, mic_point_y, mic_point_z],
                "mic_1": [mic_point_x + 0.1, mic_point_y, mic_point_z]}

    # Define noise coordinates between 0.4 and 1.5 meters away from the mic
    noise_in_room = True
    while noise_in_room:
        noise_offset = random.uniform(0.4, 1.5)
        noise_coords = [mic_point_x + noise_offset, mic_point_y + noise_offset, mic_point_z]

        # Verify that the noise coordinates are within the room dimensions
        if noise_coords[0] > 0 or noise_coords[0] < room_x or \
        noise_coords[1] > 0 or noise_coords[1] < room_y:
            noise_in_room = False

    # Define ai_noise coordinates between 0.4 and 1.5 meters away from the mic
    ai_noise_in_room = True
    while ai_noise_in_room:
        noise_offset = random.uniform(0.4, 1.5)
        ai_noise_coords = [mic_point_x + noise_offset, mic_point_y + noise_offset, mic_point_z]

        # Verify that the noise coordinates are within the room dimensions
        if ai_noise_coords[0] > 0 or ai_noise_coords[0] < room_x or \
        ai_noise_coords[1] > 0 or ai_noise_coords[1] < room_y:
            ai_noise_in_room = False

    cfg_info = [shoebox_vals, mic_dict ,
                lower_left_point, upper_right_point,
                lower_left_table, upper_right_table,
                noise_coords, ai_noise_coords, 
                abs_coeff, fs]

    return cfg_info

if __name__ == "__main__":
    # Example usage
    room_config = gen_room_config()
    print("Room Configuration:", room_config)
    print("Mic Position:", room_config[1])
    print("Inner Square Lower Left Point:", room_config[2])
    print("Inner Square Upper Right Point:", room_config[3])
    print("Table Lower Left Point:", room_config[4])
    print("Table Upper Right Point:", room_config[5])

    shoebox_vals, mic_dict, lower_left_point, upper_right_point, \
    lower_left_table, upper_right_table, abs_coeff, fs = room_config

    main_speaker_coords, tuple_outside_inside = gen_rand_table_coords(lower_left_point,
                            upper_right_point,  lower_left_table,
                            upper_right_table, table_margin = 0.3)

    print("Main Speaker Coordinates:", main_speaker_coords)
    print("Outside-Table Coordinates:", tuple_outside_inside)


    # Generate 5 random coordinates for other speakers
    rand_coordinates_other = []
    for _ in range(5):
        coord = gen_rand_coordinates(shoebox_vals[0], shoebox_vals[1],
                                      lower_left_point,
                                      upper_right_point)
        rand_coordinates_other.append(coord)

    print("Random Coordinates for Other Speakers:", rand_coordinates_other)

    # Plotting the room configuration
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.plot([lower_left_point[0], upper_right_point[0]], [lower_left_point[1], lower_left_point[1]], 'k-')  # Bottom wall
    plt.plot([lower_left_point[0], upper_right_point[0]], [upper_right_point[1], upper_right_point[1]], 'k-')  # Top wall
    plt.plot([lower_left_point[0], lower_left_point[0]], [lower_left_point[1], upper_right_point[1]], 'k-')  # Left wall
    plt.plot([upper_right_point[0], upper_right_point[0]], [lower_left_point[1], upper_right_point[1]], 'k-')  # Right wall

    # Plot the table
    plt.plot([lower_left_table[0], upper_right_table[0]], [lower_left_table[1], lower_left_table[1]], 'b-')  # Bottom of table
    plt.plot([lower_left_table[0], upper_right_table[0]], [upper_right_table[1], upper_right_table[1]], 'b-')  # Top of table
    plt.plot([lower_left_table[0], lower_left_table[0]], [lower_left_table[1], upper_right_table[1]], 'b-')  # Left of table
    plt.plot([upper_right_table[0], upper_right_table[0]], [lower_left_table[1], upper_right_table[1]], 'b-')  # Right of table


    # Plotting only 1 mic position
    plt.plot(mic_dict["mic_0"][0], mic_dict["mic_0"][1], 'ro')  # Mic position in red
    plt.text(mic_dict["mic_0"][0], mic_dict["mic_0"][1], 'Mic', fontsize=12, ha='right')
    
    # Plotting main speaker position
    plt.plot(main_speaker_coords[0], main_speaker_coords[1], 'bo')  # Main speaker in blue
    plt.text(main_speaker_coords[0], main_speaker_coords[1], 'Main Speaker', fontsize=12, ha='right')

    # Plotting other random coordinates
    for coord in rand_coordinates_other:
        plt.plot(coord[0], coord[1], 'go')  # Other speakers in green
        plt.text(coord[0], coord[1], 'Other Speaker', fontsize=10, ha='right')

    plt.xlim(0, shoebox_vals[0])
    plt.ylim(0, shoebox_vals[1])
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Room Configuration")
    plt.grid()
    plt.show()