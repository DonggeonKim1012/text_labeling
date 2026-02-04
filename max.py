import os
import h5py
import numpy as np

ZERO = np.zeros(14)

def max_change_interval(sequence: np.ndarray):
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

    return max(valid_intervals)


def process_folder(folder_path):
    global_max = None
    global_max_file = None
    global_max_key = None
    a=0
    all_max_intervals = []  # ★ 추가

    for fname in os.listdir(folder_path):
        if not fname.endswith(".hdf5"):
            continue

        path = os.path.join(folder_path, fname)
        with h5py.File(path, "r") as f:
            for key in ["prompts/numeric", "prompts/numeric2"]:
                if key not in f:
                    continue

                seq = f[key][:]
                M = max_change_interval(seq)

                if M is not None:
                    all_max_intervals.append(M)  # ★ 수집
                    if M > 230:
                        a+=1

                    # 기존 글로벌 최대(사실상 최소 interval) 로직 유지
                    if global_max is None or M > global_max:
                        global_max = M
                        global_max_file = fname
                        global_max_key = key

    avg_max_interval = (
        np.mean(all_max_intervals) if all_max_intervals else None
    )

    return global_max, global_max_file, global_max_key, avg_max_interval, a


if __name__ == "__main__":
    folder = "/home/robros/labelmaker_test/results"
    value, fname, key, avg_value, a = process_folder(folder)

    print("최대 interval (best) :", value)
    print("파일명              :", fname)
    print("dataset key         :", key)
    print("최대 interval 평균  :", avg_value)
    print("평균보다 많이 기다린 에피소드 수:", a)
