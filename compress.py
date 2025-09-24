import os
import h5py
import numpy as np
import cv2
import shutil

def compress_and_save_to_new_file(directory):
    datasets_to_cut = [
        'prompts/text/attention_mask',
        'prompts/text/input_ids',
        'prompts/text/token_type_ids'
    ]
    for filename in os.listdir(directory):
        if not (filename.endswith('.h5') or filename.endswith('.hdf5')):
            continue

        src_path = os.path.join(directory, filename)
        dst_path = os.path.join(directory, filename.replace('.h5', '_compressed.h5').replace('.hdf5', '_compressed.h5'))

        print(f"Processing: {src_path}")
        
        # Copy original file to new file first
        shutil.copy2(src_path, dst_path)

        with h5py.File(dst_path, 'r+') as f:
            dataset_path = 'prompts/masks/head_camera'
            if dataset_path not in f:
                print(f"  --> Dataset '{dataset_path}' not found. Skipping.")
                continue

            # Read original masks
            mask_ds = f[dataset_path]
            # Delete original dataset
            del f[dataset_path]

            f.create_dataset(dataset_path, data=mask_ds, compression='gzip', compression_opts=9)
            
            for dset_path in datasets_to_cut:
                if dset_path in f:
                    dset = f[dset_path]
                    data = dset[:]  # read full data
                    
                    # Cut to first half of sequence dimension (assuming shape (T, 32))
                    new_data = data[:, :16]
                    
                    # Delete old dataset
                    del f[dset_path]
                    
                    # Create new dataset with smaller size, same dtype, optionally compress
                    f.create_dataset(dset_path, data=new_data, compression="gzip", compression_opts=9)
                else:
                    print(f"Dataset {dset_path} not found in file.")

            

            print(f"  --> Saved compressed dataset to '{dst_path}'")

# Example usage
compress_and_save_to_new_file("/mnt/ddrive/Downloads/with_mask_teleop/with_mask_teleop/vertical")
