import os
import h5py
import numpy as np

def reduce_timesteps(file_path, N, subsample=False):
    print(f"Processing (in-place): {file_path}")

    with h5py.File(file_path, "r+") as f:
        def process_group(group):
            for key in list(group.keys()):
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    data = item[:]
                    if data.shape[0] > N:
                        if subsample:
                            idx = np.linspace(0, data.shape[0]-1, N, dtype=int)
                            new_data = data[idx]
                        else:
                            new_data = data[:N]

                        del group[key]
                        group.create_dataset(
                            key,
                            data=new_data,
                            compression="gzip",
                            compression_opts=4
                        )

                elif isinstance(item, h5py.Group):
                    process_group(item)

        process_group(f)

    print(f"  ✔ Reduced to {N} timesteps (overwritten in {file_path})")

reduce_timesteps("/mnt/ddrive/Downloads/with_mask_teleop/with_mask_teleop/vertical/episode_48.hdf5", N=434)
