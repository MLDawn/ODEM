import os
import numpy as np
def save(array, filename, directory):
    file_path = os.path.join(directory, f"{filename}.npy")
    np.save(file_path, array)