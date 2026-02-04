import os
import h5py


GROUPS_TO_DELETE = ["prompts", "rewards", "labels"]


def clean_hdf5_file(h5_path):
    with h5py.File(h5_path, "r+") as f:
        for grp in GROUPS_TO_DELETE:
            if grp in f:
                del f[grp]
                print(f"  deleted /{grp}")
            else:
                print(f"  /{grp} not found")


def clean_folder(folder):
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith((".h5", ".hdf5")):
            continue

        print(f"\nCleaning {fname}")
        clean_hdf5_file(os.path.join(folder, fname))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove unused groups from HDF5 files (in place)"
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Folder containing HDF5 files to clean"
    )

    args = parser.parse_args()
    clean_folder(args.folder)

