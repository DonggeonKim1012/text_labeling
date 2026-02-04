import os
import h5py
import numpy as np

ZERO = np.zeros(14)

def min_change_interval(sequence: np.ndarray):
    T = len(sequence)
    change_indices = []
    change_values = []

    for t in range(1, T):
        if np.any(sequence[t] != sequence[t - 1]):
            change_indices.append(t)
            change_values.append(sequence[t])

    if not change_indices:
        return None

    valid_intervals = []

    for i in range(len(change_indices) - 1):
        if np.all(change_values[i] == ZERO):
            continue
        valid_intervals.append(change_indices[i + 1] - change_indices[i])

    if not valid_intervals:
        return None

    return min(valid_intervals)


def process_folder(folder_path):
    global_min = None
    global_min_file = None
    global_min_key = None  # numeric / numeric2 구분용

    for fname in os.listdir(folder_path):
        if not fname.endswith(".hdf5"):
            continue

        path = os.path.join(folder_path, fname)
        with h5py.File(path, "r") as f:
            for key in ["prompts/numeric", "prompts/numeric2"]:
                if key not in f:
                    continue

                seq = f[key][:]
                m = min_change_interval(seq)

                if m is not None:
                    if global_min is None or m < global_min:
                        global_min = m
                        global_min_file = fname
                        global_min_key = key

    return global_min, global_min_file, global_min_key


if __name__ == "__main__":
    folder = "/home/robros/labelmaker_test/results"
    value, fname, key = process_folder(folder)

    print("최소 interval :", value)
    print("파일명        :", fname)
    print("dataset key   :", key)
