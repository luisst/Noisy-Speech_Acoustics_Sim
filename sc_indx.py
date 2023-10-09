import random

x_data = [11,22,33,44,55,66]
index_list = list(range(len(x_data)))



random_idx = random.choice(index_list)
index_list.remove(random_idx)