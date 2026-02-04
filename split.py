import os
import h5py
import numpy as np


def split_hdf_by_intervals(file_path, index_list):
    """
    Given a list of indices, split the HDF5 file into multiple HDF5 files,
    each containing slices of datasets according to the intervals.

    Example:
        index_list = [3, 80, 96]
        Output intervals = [:3], [3:80], [80:96], [96:]
    """

    # Ensure sorted indices
    index_list = sorted(index_list)

    # Open original file
    with h5py.File(file_path, "r") as f:
        # Determine length T by reading the first (T, ...) dataset
        def find_T(group):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    data = item
                    if data.shape[0] > 1:
                        return data.shape[0]
                elif isinstance(item, h5py.Group):
                    t = find_T(item)
                    if t is not None:
                        return t
            return None

        T = find_T(f)
        if T is None:
            raise ValueError("Could not infer T dimension from datasets.")

        # Build intervals
        # Example: [3,80,96] → [(0,3), (3,80), (80,96), (96,T)]
        boundaries = [0] + index_list + [T]
        intervals = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

        print("Intervals to export:", intervals)

        # Process each interval
        for idx, (start, end) in enumerate(intervals):
            out_path = file_path.replace(".hdf5", f"_{start}_{end}.hdf5")
            print(f"\nCreating split: {out_path}  →  [{start}:{end}]")

            with h5py.File(out_path, "w") as out_f:

                def copy_group(in_group, out_group):
                    for key in in_group.keys():
                        item = in_group[key]

                        if isinstance(item, h5py.Dataset):
                            data = item

                            # Special case: compressed_image_len (shape [3, T])
                            if data.shape[0] == 3 and data.ndim == 2:
                                sliced = data[:, start:end]

                            # Default case: assume first dimension is time (T, ...)
                            elif data.shape[0] == T:
                                sliced = data[start:end]

                            else:
                                # Non-time datasets → copy fully
                                sliced = data[...]

                            out_group.create_dataset(
                                key,
                                data=sliced,
                                compression="gzip",
                                compression_opts=4
                            )

                        elif isinstance(item, h5py.Group):
                            new_group = out_group.create_group(key)
                            copy_group(item, new_group)

                copy_group(f, out_f)

            print(f"✔ Saved: {out_path}")


# Example usage:
split_hdf_by_intervals(
    "episode_209.hdf5",
    [68,198, 280,672, 738,896, 977,1110, 1174,1337, 1399]
)
