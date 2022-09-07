import numpy as np

def get_file_as_np_int_array(path):
    array = []
    with open(path, 'r') as f:
        for line in f:
            array.append([int(x) for x in line.split()])
    array = np.array(array)
    return array

def save_np_array_array_to_file(path, array):
    with open(path, 'a+') as f:
        f.truncate(0)
        for row in array:
            line = ' '.join(str(x) for x in row) + '\n'
            f.write(line)