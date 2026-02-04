import os
import h5py
import zlib
import numpy as np

MASK_SHAPE = (480, 640)
DTYPE = np.uint8
NO_MASK_THRESHOLD = 10
MAX_INTERVAL = 50


def find_no_mask_intervals(h5_path):
    """Return no-mask intervals (> NO_MASK_THRESHOLD) and file length T"""
    intervals = []
    start = None

    with h5py.File(h5_path, 'r') as f:
        masks_ds = f['prompts/masks/head_camera']
        T = len(masks_ds)

        for t in range(T):
            raw = zlib.decompress(masks_ds[t])
            mask = np.frombuffer(raw, dtype=DTYPE).reshape(MASK_SHAPE)

            has_mask = np.any(mask)

            if not has_mask:
                if start is None:
                    start = t
            else:
                if start is not None and t - start > NO_MASK_THRESHOLD:
                    intervals.append((start, t))
                    start = None
        if start is not None and T - start > NO_MASK_THRESHOLD:
            intervals.append((start, T))

    return intervals, T


def copy_slice(src, dst, start, end):
    """Copy datasets, slicing axis 0 or 1 depending on dataset"""
    for key in src.keys():
        obj = src[key]

        if isinstance(obj, h5py.Group):
            grp = dst.create_group(key)
            copy_slice(obj, grp, start, end)
        else:
            data = obj
            if key == 'compressed_image_len':
                sliced = data[:, start:end]
            else:
                sliced = data[start:end]

            dset = dst.create_dataset(
                key,
                data=sliced,
                compression=data.compression,
                dtype=data.dtype
            )
            # copy attributes
            for attr in data.attrs:
                dset.attrs[attr] = data.attrs[attr]


def process_file(h5_path, out_dir):
    intervals, T = find_no_mask_intervals(h5_path)
    base = os.path.splitext(os.path.basename(h5_path))[0]
    prev_start_idx = 0
    file_count = 0

    # handle first interval starting at 0
    if intervals and intervals[0][0] == 0 and intervals[0][1] - intervals[0][0] > MAX_INTERVAL:
        prev_start_idx = MAX_INTERVAL

    with h5py.File(h5_path, 'r') as src:
        for s, e in intervals:
            if e - s > MAX_INTERVAL:
                new_start = prev_start_idx
                new_end = min(s + MAX_INTERVAL, T)
                if new_start < new_end:
                    out_path = os.path.join(out_dir, f"{base}_cut_{file_count:02d}.hdf5")
                    with h5py.File(out_path, 'w') as dst:
                        copy_slice(src, dst, new_start, new_end)
                    print(f"Created: {out_path} [{new_start}:{new_end}]")
                    file_count += 1
                prev_start_idx = e  # move prev_start_idx to end of interval

        # final segment
        if prev_start_idx < T:
            out_path = os.path.join(out_dir, f"{base}_cut_{file_count:02d}.hdf5")
            with h5py.File(out_path, 'w') as dst:
                copy_slice(src, dst, prev_start_idx, T)
            print(f"Created final segment: {out_path} [{prev_start_idx}:{T}]")


def process_folder(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname in sorted(os.listdir(in_dir)):
        if not fname.endswith((".h5", ".hdf5")):
            continue
        print(f"\nProcessing {fname}")
        process_file(os.path.join(in_dir, fname), out_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cut HDF5 files based on long no-mask intervals (>80)"
    )

    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing input HDF5 files"
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder to write new HDF5 files"
    )

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)

